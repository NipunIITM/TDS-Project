# Data Analyst Agent (Gemini)

A token-efficient API that accepts `questions.txt` and optional data files, plans with an LLM, generates Python code, executes it safely, and returns the requested answers and plots (as base64 data URIs).

- Endpoint: `POST /api/`
- Health: `GET /`

## Features
- Handles `questions.txt` (required) and optional files: CSV/TSV/Parquet/JSON/images
- Token-efficient prompt (only previews of data are sent)
- LLM outputs raw Python code; server executes it in a sandbox with a time limit
- Supports scraping (requests/BeautifulSoup) and plotting (matplotlib)
- Ensures image outputs under ~100kB (prompted)

## Local Setup
1. Create virtualenv and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy env template and set key:
   ```bash
   cp .env.example .env
   # edit .env to add GEMINI_API_KEY
   ```
3. Run the server:
   ```bash
   python app.py
   # or
   flask --app app run --host 0.0.0.0 --port 8080
   ```
4. Test:
   ```bash
   echo "Scrape the list of highest grossing films from Wikipedia: https://en.wikipedia.org/wiki/List_of_highest-grossing_films\n\nReturn a JSON array with:\n1) How many $2 bn movies before 2000?\n2) Earliest film over $1.5 bn?\n3) Correlation between Rank and Peak.\n4) Scatterplot Rank vs Peak with dotted red regression line under 100kB." > question.txt

   curl -X POST http://localhost:8080/api/ \
     -F "questions.txt=@question.txt"
   ```

## Deploy on Render
- Create a new Web Service from this repo
- Environment variables:
  - `GEMINI_API_KEY`
  - Optional: `GEMINI_MODEL`, `GEMINI_MAX_TOKENS`, `EXEC_TIMEOUT_SECONDS`
- Build command:
  ```
  pip install -r requirements.txt
  ```
- Start command:
  ```
  gunicorn app:app --workers 2 --threads 4 --timeout 180
  ```

## Notes
- The sandbox whitelists common analytics libs. If you need DuckDB for S3 queries, it is installed; set `DISABLE_DUCKDB=1` to block it.
- Execution is limited by `EXEC_TIMEOUT_SECONDS` (default 90s).