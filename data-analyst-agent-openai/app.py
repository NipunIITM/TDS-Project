import os
import io
import base64
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# --- Flask App Initialization ---
load_dotenv()
app = Flask(__name__)

# --- OpenAI Client Initialization ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# --- System Prompt for the LLM ---
SYSTEM_PROMPT = """
You are a world-class Python data analyst agent. Your goal is to help users by writing a Python script to answer their questions about data.

You will be given a user's question and, if they uploaded a file, the first 5 rows of their data in a pandas DataFrame named `df`.

Your task is to **write a complete, self-contained Python script** that performs the necessary steps to answer the user's questions.

**Instructions & Rules:**
1. **Read the User's Query**: Understand what they are asking for (e.g., calculations, plots, data scraping).
2. **Data Source**:
   - If a pandas DataFrame named `df` is provided, use it as the primary data source.
   - If the user provides a URL, you **MUST** write code to scrape the data from that URL (e.g., using `pd.read_html` or `requests` and `BeautifulSoup`).
   - If no data or URL is provided, and the question requires external data, state that you cannot answer without the data.
3. **Code Generation**:
   - Your output **MUST** be only a Python script. Do not add any explanations, comments, or markdown formatting like ```python ... ```. Just the raw code.
   - The script must be self-contained and use libraries like `pandas`, `matplotlib`, `numpy`, `sklearn`, `requests`, `bs4`, etc.
   - Import all necessary libraries at the beginning of your script.
4. **Answering Questions**:
   - Perform all calculations, data cleaning, analysis, and plotting within the script.
   - For plots, you **MUST** generate a plot using `matplotlib`, save it to an in-memory buffer (`io.BytesIO`), encode it to a Base64 string, and format it as a `data:image/png;base64,...` data URI. The final image must be under 100,000 bytes. To achieve this, use arguments like `dpi=80` and `figsize=(8,6)` in `plt.savefig`.
   - Always call `plt.close()` after saving each plot to free memory.
5. **Final Output**:
   - Your script **MUST** create a variable called `final_answers`.
   - This variable will hold the results. It should be a **list** or a **dictionary**, matching the format requested by the user.
   - Place all answers (numbers, strings, and the Base64 data URI for plots) into this `final_answers` variable.
   - **This is the most important step.** The execution environment will capture this variable to get the result.

**Example for Wikipedia scraping task:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

# Scrape Wikipedia data
url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
tables = pd.read_html(url)
df = tables[0]  # Get the main table

# Clean and process data as needed
# ... data processing code ...

# Answer questions
answer1 = len(df[df['Worldwide gross'] >= 2000000000])
answer2 = df[df['Worldwide gross'] >= 1500000000]['Title'].iloc[0]
correlation = df['Rank'].corr(df['Peak'])

# Create plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Rank'], df['Peak'], alpha=0.6)
# Add regression line
X = df['Rank'].values.reshape(-1, 1)
y = df['Peak'].values
reg = LinearRegression().fit(X, y)
plt.plot(df['Rank'], reg.predict(X), 'r--', linewidth=2)
plt.xlabel('Rank')
plt.ylabel('Peak')
plt.title('Rank vs Peak Correlation')

# Save plot
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
buf.seek(0)
plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
plot_uri = f"data:image/png;base64,{plot_base64}"
buf.close()
plt.close()

final_answers = [answer1, answer2, correlation, plot_uri]
```

Remember: Your output must be ONLY the Python code, no explanations or markdown formatting.
"""

def safe_exec(code_string: str, df: pd.DataFrame = None):
    """
    Executes the LLM-generated Python code in a controlled environment.
    """
    # Create a safe execution environment with necessary imports
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'print': print,
        }
    }
    
    local_scope = {
        'df': df.copy() if df is not None else None,
        'pd': pd,
        'np': np,
        'plt': plt,
        'io': io,
        'base64': base64,
        'requests': requests,
        'BeautifulSoup': BeautifulSoup,
        'LinearRegression': LinearRegression,
        'final_answers': None
    }
    
    try:
        exec(code_string, safe_globals, local_scope)
        result = local_scope.get('final_answers', "Error: 'final_answers' variable not found in executed script.")
        return result
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error executing generated code:\n{tb_str}")
        return {"error": "Failed to execute the generated analysis script.", "details": str(e), "traceback": tb_str}

@app.route("/api/", methods=["POST"])
def analyze_data():
    """
    Main API endpoint to receive data analysis tasks.
    """
    if not client:
        return jsonify({"error": "OpenAI client is not initialized. Please check your API key."}), 500

    if 'questions.txt' not in request.files:
        return jsonify({"error": "'questions.txt' file is required."}), 400

    try:
        # Read the mandatory questions file
        questions_file = request.files['questions.txt']
        user_questions = questions_file.read().decode('utf-8')

        # Read optional data files
        df = None
        data_context = "No data file was provided. If the task requires data from a URL, the script should fetch it."
        
        # Check for CSV file
        if 'data.csv' in request.files:
            data_file = request.files['data.csv']
            try:
                df = pd.read_csv(data_file)
                data_context = f"A CSV file has been provided with {len(df)} rows and {len(df.columns)} columns. Here are the first 5 rows:\n\n{df.head().to_string()}\n\nColumn names: {list(df.columns)}"
            except Exception as e:
                return jsonify({"error": f"Failed to read CSV file: {e}"}), 400
        
        # Check for other file types (images, etc.)
        other_files = []
        for key in request.files:
            if key not in ['questions.txt', 'data.csv']:
                other_files.append(key)
        
        if other_files:
            data_context += f"\n\nAdditional files uploaded: {', '.join(other_files)}"

        # Construct the prompt for the LLM
        full_prompt = f"""**User Questions:**
{user_questions}

**Data Context:**
{data_context}

Please write a Python script to answer these questions. Remember to put your final answers in a variable called 'final_answers'."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        
        generated_code = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if present
        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        elif generated_code.startswith("```"):
            generated_code = generated_code[3:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        
        generated_code = generated_code.strip()
        
        print("--- Generated Code by OpenAI ---")
        print(generated_code)
        print("--------------------------------")

        # Execute the generated code
        result = safe_exec(generated_code, df)

        # Handle execution errors
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Server error: {tb_str}")
        return jsonify({"error": "Internal server error occurred.", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "Data Analyst Agent (OpenAI) is running", "version": "1.0"})

@app.route("/health", methods=["GET"])
def health():
    """Additional health check endpoint."""
    return jsonify({"status": "healthy", "provider": "openai"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)