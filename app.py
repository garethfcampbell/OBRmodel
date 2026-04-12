from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import openai
import json
import uuid
import threading
import traceback
import re
from datetime import datetime, timedelta
from contextlib import contextmanager
import time
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
_session_secret = os.environ.get("SESSION_SECRET")
if not _session_secret:
    raise RuntimeError("SESSION_SECRET environment variable must be set")
app.secret_key = _session_secret

# Restrict to same origin only
CORS(app, origins=[], supports_credentials=False)

# Rate limiting: 10 AI requests per minute per IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
        "font-src cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

def require_json_csrf(f):
    """Verify the request is a genuine AJAX JSON call (mitigates CSRF)."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "POST":
            if not request.is_json:
                return jsonify({"error": "Invalid request format"}), 400
            if request.headers.get("X-Requested-With") != "XMLHttpRequest":
                return jsonify({"error": "Invalid request"}), 403
        return f(*args, **kwargs)
    return decorated

def sanitize_user_input(text):
    """Strip control characters and prompt injection attempts."""
    text = text.strip()
    # Remove null bytes and other non-printable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Basic prompt injection guardrail: flag repeated instruction-style patterns
    injection_patterns = [
        r'ignore (all |previous |above |prior )?instructions',
        r'you are now',
        r'disregard your',
        r'new instructions:',
        r'system prompt',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[removed]', text, flags=re.IGNORECASE)
    return text

model_code = None
model_variables = None

tasks = {}
task_cleanup_thread = None

# PostgreSQL connection pool
db_pool = None
db_pool_lock = threading.Lock()

def get_db_pool():
    global db_pool
    if db_pool is None:
        with db_pool_lock:
            if db_pool is None:
                db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    dsn=os.environ.get("DATABASE_URL")
                )
    return db_pool

@contextmanager
def get_db_conn():
    conn = get_db_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_db_pool().putconn(conn)

def load_model_data():
    global model_code, model_variables
    if model_code is None:
        with open('Macroeconomic_model_code_March_2024.txt', 'r') as f:
            model_code = f.read()
    if model_variables is None:
        with open('OBR_Model_Variables_March_2024.csv', 'r') as f:
            model_variables = f.read()
    return model_code, model_variables

client = openai.OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

openai_client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gpt-5-nano"
MAX_SESSION_RESPONSE_LENGTH = 500

def call_gemini(messages, timeout=600):
    """Call Gemini API with automatic fallback to GPT-5-nano on rate limit errors."""
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        error_str = str(e).lower()
        is_rate_limit = (
            "429" in str(e) or
            "rate limit" in error_str or
            "quota" in error_str or
            "resource_exhausted" in error_str
        )
        if is_rate_limit:
            logging.warning(f"Rate limit hit on {PRIMARY_MODEL}, falling back to {FALLBACK_MODEL}: {e}")
            response = openai_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                timeout=timeout
            )
            return response.choices[0].message.content
        raise

def call_gemini_stream(messages, timeout=600):
    """Call Gemini API with streaming, yielding text chunks. Falls back to GPT-5-nano on rate limit."""
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            timeout=timeout,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_str = str(e).lower()
        is_rate_limit = (
            "429" in str(e) or
            "rate limit" in error_str or
            "quota" in error_str or
            "resource_exhausted" in error_str
        )
        if is_rate_limit:
            logging.warning(f"Rate limit hit on {PRIMARY_MODEL}, falling back to {FALLBACK_MODEL}: {e}")
            response = openai_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                timeout=timeout,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            raise

def pg_save_task(task_id, status, user_id=None, message=None, result=None, error=None, completed_at=None):
    """Upsert a task record in PostgreSQL."""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tasks (task_id, status, user_id, message, result, error, completed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        error = EXCLUDED.error,
                        completed_at = EXCLUDED.completed_at
                """, (task_id, status, user_id, message, result, error, completed_at))
    except Exception as e:
        logging.error(f"Failed to save task {task_id} to PostgreSQL: {e}")
        raise

def pg_get_task(task_id):
    """Fetch a task record from PostgreSQL."""
    try:
        with get_db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM tasks WHERE task_id = %s", (task_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logging.error(f"Failed to get task {task_id} from PostgreSQL: {e}")
        return None

def pg_delete_task(task_id):
    """Delete a task record from PostgreSQL."""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tasks WHERE task_id = %s", (task_id,))
    except Exception as e:
        logging.error(f"Failed to delete task {task_id} from PostgreSQL: {e}")

def get_conversation_history(user_id):
    """Get conversation history for a user from PostgreSQL."""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT messages FROM conversation_history WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                return row[0] if row else []
    except Exception as e:
        logging.error(f"Failed to get conversation history for {user_id}: {e}")
        return []

def save_conversation_history(user_id, history):
    """Save conversation history for a user to PostgreSQL."""
    try:
        trimmed = history[-20:] if len(history) > 20 else history
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_history (user_id, messages, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        messages = EXCLUDED.messages,
                        updated_at = NOW()
                """, (user_id, json.dumps(trimmed)))
    except Exception as e:
        logging.error(f"Failed to save conversation history for {user_id}: {e}")

def delete_conversation_history(user_id):
    """Delete conversation history for a user from PostgreSQL."""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversation_history WHERE user_id = %s", (user_id,))
    except Exception as e:
        logging.error(f"Failed to delete conversation history for {user_id}: {e}")

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def cleanup_old_tasks():
    """Background thread to clean up old completed tasks from both memory and DB."""
    while True:
        try:
            cutoff = datetime.now() - timedelta(hours=1)
            tasks_to_remove = [
                tid for tid, tdata in list(tasks.items())
                if tdata.get('created_at', datetime.now()) < cutoff
            ]
            for task_id in tasks_to_remove:
                tasks.pop(task_id, None)
                pg_delete_task(task_id)

            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM tasks WHERE created_at < NOW() - INTERVAL '2 hours'")
        except Exception as e:
            logging.error(f"Cleanup thread error: {e}")
        time.sleep(600)

def start_cleanup_thread():
    global task_cleanup_thread
    if task_cleanup_thread is None or not task_cleanup_thread.is_alive():
        task_cleanup_thread = threading.Thread(target=cleanup_old_tasks, daemon=True)
        task_cleanup_thread.start()

def run_background_task(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()

def _build_initial_messages(user_message):
    """Build the messages list for an initial chat request."""
    model_code, model_variables = load_model_data()
    prompt = f"""You are an expert economist analyzing the UK economy using the OBR's macroeconomic model. The user will ask about the impact of an economic shock. Your task is to trace the impact of this shock through the model, explaining the steps in words.

Here is the model code:
{model_code}

Here are the model variables:
{model_variables}

User's question: What would happen if: {user_message}

CRITICAL FORMATTING REQUIREMENTS:
- Do NOT provide any introductory text or preamble
- Begin immediately with "### EXECUTIVE SUMMARY" as the first line
- Use EXACTLY these section headings in this order with proper markdown formatting:
  1. ### EXECUTIVE SUMMARY
  2. ### STEP BY STEP ANALYSIS  
  3. ### LONG-RUN OUTCOMES
- Use proper markdown formatting for headings (### for main headings, #### for step headings)
- For equations, use simple mathematical notation that can be displayed as plain text
- Structure your response clearly with proper paragraph breaks
- For each step in STEP BY STEP ANALYSIS, the three sub-sections MUST each start on their own new line, formatted exactly like this (including the blank line between each):

**a) Economic Logic:**
[text here]

**b) Model Transmission:**
[text here]

**c) Equations:**
[equations here]

Never run a), b) and c) together on the same line or in the same paragraph.

Please provide:

EXECUTIVE SUMMARY

STEP BY STEP ANALYSIS
Analysis of the shock's impact, explaining for each step:

**a) Economic Logic:**
[overview of this step]

**b) Model Transmission:**
[equations explained in plain English]

**c) Equations:**
[relevant equations listed]

LONG-RUN OUTCOMES

"""
    logging.info(f"Prompt length: {len(prompt)} characters")
    return [
        {"role": "system", "content": "You are an expert economist."},
        {"role": "user", "content": prompt}
    ]

def _build_contextual_messages(user_message, conversation_history):
    """Build the messages list for a follow-up chat request."""
    model_code, model_variables = load_model_data()
    system_content = f"""You are an expert economist analyzing the UK economy using the OBR's macroeconomic model. The user has asked about the impact of an economic shock and this has been analysed. The user may have additional questions or comments about the analysis which you should respond to.

CRITICAL FORMATTING REQUIREMENTS:
- Provide clear, well-structured responses using proper markdown formatting
- Use proper markdown headings (# ##) for structure
- For equations, use simple mathematical notation that can be displayed as plain text
- Structure your response with clear paragraph breaks and bullet points where helpful
- Be direct and concise in your responses
- If referencing specific model equations or variables, explain them clearly

Here is the model code:
{model_code}

Here are the model variables:
{model_variables}
"""
    recent_history = conversation_history[-10:]
    system_content += "\n\nPrevious Conversation Context:\n"
    for msg in recent_history:
        role_label = "User" if msg["role"] == "user" else "AI Economist"
        system_content += f"{role_label}: {msg['content']}\n"
    system_content += "\nPlease use this conversation history to provide a contextually relevant response to the user's new message below."
    logging.info(f"Total message length: {len(system_content) + len(user_message)} characters")
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"The latest follow up question is: {user_message}"}
    ]

def _handle_initial_chat_sync(user_message, task_id):
    try:
        tasks[task_id]['status'] = 'processing'
        pg_save_task(task_id, 'processing', user_id=tasks[task_id]['user_id'], message=user_message)

        messages = _build_initial_messages(user_message)
        ai_response = call_gemini(messages)

        user_id = tasks[task_id]['user_id']
        history = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response[:MAX_SESSION_RESPONSE_LENGTH]}
        ]
        save_conversation_history(user_id, history)

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = ai_response
        tasks[task_id]['completed_at'] = datetime.now()
        pg_save_task(task_id, 'completed', user_id=user_id, message=user_message,
                     result=ai_response, completed_at=tasks[task_id]['completed_at'])

    except Exception as e:
        logging.error(f"Error in initial chat: {e}\n{traceback.format_exc()}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['completed_at'] = datetime.now()
        pg_save_task(task_id, 'error', error=str(e), completed_at=tasks[task_id]['completed_at'])

def _handle_contextual_chat_sync(user_message, conversation_history, task_id):
    try:
        tasks[task_id]['status'] = 'processing'
        pg_save_task(task_id, 'processing', user_id=tasks[task_id]['user_id'], message=user_message)

        messages = _build_contextual_messages(user_message, conversation_history)
        ai_response = call_gemini(messages)

        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": ai_response[:MAX_SESSION_RESPONSE_LENGTH]})
        user_id = tasks[task_id]['user_id']
        save_conversation_history(user_id, conversation_history)

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = ai_response
        tasks[task_id]['completed_at'] = datetime.now()
        pg_save_task(task_id, 'completed', user_id=user_id, message=user_message,
                     result=ai_response, completed_at=tasks[task_id]['completed_at'])

    except Exception as e:
        logging.error(f"Error in contextual chat: {e}\n{traceback.format_exc()}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['completed_at'] = datetime.now()
        pg_save_task(task_id, 'error', error=str(e), completed_at=tasks[task_id]['completed_at'])

@app.route('/')
def index():
    user_id = get_user_id()
    delete_conversation_history(user_id)
    start_cleanup_thread()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
@require_json_csrf
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = sanitize_user_input(data['message'])
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    if len(user_message) > 2000:
        return jsonify({'error': 'Message too long (max 2000 characters)'}), 400

    task_id = str(uuid.uuid4())
    user_id = get_user_id()

    try:
        conversation_history = get_conversation_history(user_id)

        tasks[task_id] = {
            'status': 'started',
            'user_id': user_id,
            'created_at': datetime.now(),
            'message': user_message
        }

        pg_save_task(task_id, 'started', user_id=user_id, message=user_message)
        logging.info(f"Task {task_id} persisted to PostgreSQL")

        if not conversation_history:
            run_background_task(_handle_initial_chat_sync, user_message, task_id)
        else:
            run_background_task(_handle_contextual_chat_sync, user_message, conversation_history, task_id)

        return jsonify({'task_id': task_id})

    except Exception as e:
        logging.error(f"Error starting chat task: {e}")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/chat-stream', methods=['POST'])
@limiter.limit("10 per minute")
@require_json_csrf
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = sanitize_user_input(data['message'])
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    if len(user_message) > 2000:
        return jsonify({'error': 'Message too long (max 2000 characters)'}), 400

    user_id = get_user_id()
    conversation_history = get_conversation_history(user_id)

    if not conversation_history:
        messages = _build_initial_messages(user_message)
    else:
        messages = _build_contextual_messages(user_message, conversation_history)

    def generate():
        full_response = []
        try:
            for chunk in call_gemini_stream(messages):
                full_response.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Save conversation history after stream completes
            ai_response = ''.join(full_response)
            if not conversation_history:
                history = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ai_response[:MAX_SESSION_RESPONSE_LENGTH]}
                ]
            else:
                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": ai_response[:MAX_SESSION_RESPONSE_LENGTH]})
                history = conversation_history
            save_conversation_history(user_id, history)

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logging.error(f"Streaming error: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/poll/<task_id>', methods=['GET'])
def poll_task(task_id):
    task_data = tasks.get(task_id)

    if not task_data:
        row = pg_get_task(task_id)
        if row:
            logging.info(f"Found task {task_id} in PostgreSQL, restoring to memory")
            task_data = {
                'status': row['status'],
                'user_id': row.get('user_id'),
                'message': row.get('message'),
                'result': row.get('result'),
                'error': row.get('error'),
                'created_at': row['created_at'] if isinstance(row['created_at'], datetime) else datetime.fromisoformat(str(row['created_at'])),
                'completed_at': row.get('completed_at')
            }
            tasks[task_id] = task_data
        else:
            logging.warning(f"Task {task_id} not found in memory or PostgreSQL")
            return jsonify({'status': 'not_found', 'error': 'Task not found'}), 200

    response = {
        'status': task_data['status'],
        'created_at': task_data['created_at'].isoformat() if isinstance(task_data['created_at'], datetime) else str(task_data['created_at'])
    }

    if task_data['status'] == 'completed':
        response['result'] = task_data['result']
        completed_at = task_data.get('completed_at')
        if completed_at:
            response['completed_at'] = completed_at.isoformat() if isinstance(completed_at, datetime) else str(completed_at)
        cutoff = datetime.now() - timedelta(hours=2)
        if completed_at and (completed_at if isinstance(completed_at, datetime) else datetime.fromisoformat(str(completed_at))) < cutoff:
            tasks.pop(task_id, None)
            pg_delete_task(task_id)

    elif task_data['status'] == 'error':
        response['error'] = task_data.get('error', 'Unknown error')
        completed_at = task_data.get('completed_at')
        if completed_at:
            response['completed_at'] = completed_at.isoformat() if isinstance(completed_at, datetime) else str(completed_at)

    return jsonify(response)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'})

    user_message = data['message']
    task_id = str(uuid.uuid4())

    try:
        tasks[task_id] = {
            'status': 'started',
            'created_at': datetime.now(),
            'message': user_message
        }
        pg_save_task(task_id, 'started', message=user_message)

        def calculate_sync():
            try:
                tasks[task_id]['status'] = 'processing'
                pg_save_task(task_id, 'processing', message=user_message)

                model_code, model_variables = load_model_data()

                prompt = f"""You are an expert economist analyzing the UK economy using the OBR's macroeconomic model. The user wants to see the calculations for an economic shock. Your task is to trace the impact of this shock through the model, showing the equations in LaTeX format.

                Here is the model code:
                {model_code}

                Here are the model variables:
                {model_variables}

                User's question: {user_message}

                CRITICAL FORMATTING REQUIREMENTS:
                - When referring to variable names in prose text, use markdown backticks (e.g. `IBUS`, `KGAP`, `GDPMPS`), NOT LaTeX $ symbols
                - For equations, ALWAYS use display math mode with double dollar signs ($$) on their own line, never inline single $ signs. Each equation must be on its own line with a blank line before and after. For example:

                The capital gap is defined as:

                $$KGAP = KSTAR - KBUS$$

                And business investment responds to:

                $$IBUS = f(KGAP, RREAL, PROFITS)$$

                - Never put multiple equations on the same line
                - Use proper markdown headings (# ## ###) for clear section structure
                - Structure your response with clear step-by-step analysis
                - Make sure all LaTeX equations are properly formatted for MathJax rendering
                - Combine markdown formatting with LaTeX for the best presentation

                Please provide a step-by-step analysis of the shock's impact, showing the equations in LaTeX format, interpreted by MathJax.
                """
                logging.info(f"Calculate prompt length: {len(prompt)} characters")

                messages = [
                    {"role": "system", "content": "You are an expert economist."},
                    {"role": "user", "content": prompt}
                ]
                ai_response = call_gemini(messages)

                tasks[task_id]['status'] = 'completed'
                tasks[task_id]['result'] = ai_response
                tasks[task_id]['completed_at'] = datetime.now()
                pg_save_task(task_id, 'completed', message=user_message,
                             result=ai_response, completed_at=tasks[task_id]['completed_at'])

            except Exception as e:
                logging.error(f"Error in calculate: {e}\n{traceback.format_exc()}")
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error'] = str(e)
                tasks[task_id]['completed_at'] = datetime.now()
                pg_save_task(task_id, 'error', error=str(e), completed_at=tasks[task_id]['completed_at'])

        run_background_task(calculate_sync)
        return jsonify({'task_id': task_id})

    except Exception as e:
        logging.error(f"Error starting calculate task: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
