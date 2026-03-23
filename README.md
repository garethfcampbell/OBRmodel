# OBR Model Analyst

A web application for exploring the Office for Budget Responsibility's (OBR) macroeconomic model. Ask questions about the impact of economic shocks and trace how they flow through the model step by step.

## What it does

- Accepts a description of an economic shock (e.g. "increase income tax by 1%")
- Uses AI to trace the shock through the OBR's published macroeconomic model
- Returns a structured analysis with an executive summary, step-by-step transmission, and long-run outcomes
- Supports follow-up questions on the same scenario

## Tech stack

- **Backend**: Python / Flask, served via Gunicorn
- **Database**: PostgreSQL (for task and conversation storage)
- **AI**: Google Gemini (primary), OpenAI GPT-5-nano (rate limit fallback)
- **Frontend**: Bootstrap 5, marked.js, DOMPurify

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/garethfcampbell/OBRmodel.git
cd OBRmodel
```

### 2. Obtain the OBR model data files

This application requires two data files from the OBR's published macroeconomic model documentation. These are not included in the repository and must be downloaded separately.

1. Go to the OBR website: **[obr.uk](https://obr.uk)**
2. Navigate to **About > Macroeconomic model** (or search for "macroeconomic model documentation")
3. Download the **March 2024** model documentation package
4. From the downloaded files, place the following in the root of this project:
   - `Macroeconomic_model_code_March_2024.txt` — the model equation code
   - `OBR_Model_Variables_March_2024.csv` — the variable definitions

Direct link: [https://obr.uk/data/](https://obr.uk/data/)

### 3. Install dependencies

```bash
pip install flask gunicorn openai psycopg2-binary
```

### 4. Set environment variables

The following environment variables are required:

| Variable | Description |
|----------|-------------|
| `SESSION_SECRET` | A long random string used to sign session cookies |
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `OPENAI_API_KEY` | Your OpenAI API key (used as rate limit fallback) |
| `DATABASE_URL` | PostgreSQL connection string |

You can get a Gemini API key at [aistudio.google.com](https://aistudio.google.com) and an OpenAI key at [platform.openai.com](https://platform.openai.com).

### 5. Set up the database

Connect to your PostgreSQL database and run:

```sql
CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR(36) PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'started',
    user_id VARCHAR(36),
    message TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_history (
    user_id VARCHAR(36) PRIMARY KEY,
    messages JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 6. Run the application

```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Notes

- The application uses an async task queue so long-running AI requests do not block other users
- AI responses fall back automatically from Gemini to GPT-5-nano if rate limits are hit
- Conversation history is stored per session and cleared when the page is reloaded
