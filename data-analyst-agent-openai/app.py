import os
import io
import base64
import traceback
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from bs4 import BeautifulSoup
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
SYSTEM_PROMPT = """You are a world-class Python data analyst agent. Your task is to write a complete, self-contained Python script that answers the user's questions about data.

CRITICAL RULES:
1. Your output must be ONLY Python code - no explanations, no markdown, no ```python blocks
2. The script must create a variable called `final_answers` containing the results
3. For plots: Use matplotlib, save to BytesIO buffer, encode as base64, format as "data:image/png;base64,..."
4. Keep images under 100KB by using dpi=80 or lower
5. If URL is provided, scrape the data using pandas.read_html() or requests+BeautifulSoup
6. Handle missing data gracefully
7. Import all required libraries at the top

Available libraries: pandas, numpy, matplotlib, sklearn, requests, bs4, io, base64

EXAMPLE OUTPUT FORMAT:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import requests
from bs4 import BeautifulSoup

# Your analysis code here
# ...

# Save plot if needed
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
buf.seek(0)
plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
plot_uri = f"data:image/png;base64,{plot_b64}"
buf.close()
plt.close()

# Final results
final_answers = [answer1, answer2, plot_uri]
```

Remember: Only return the Python script, nothing else."""

def safe_exec(code_string: str, df: pd.DataFrame = None):
    """Execute LLM-generated Python code safely"""
    # Prepare execution environment
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
        # Execute the generated code
        exec(code_string, {'__builtins__': {}}, local_scope)
        result = local_scope.get('final_answers')
        
        if result is None:
            return {"error": "Script did not set 'final_answers' variable"}
        
        return result
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Execution error:\n{tb_str}")
        return {"error": f"Script execution failed: {str(e)}"}

@app.route("/api/", methods=["POST"])
def analyze_data():
    """Main API endpoint for data analysis"""
    if not client:
        return jsonify({"error": "OpenAI client not initialized. Check OPENAI_API_KEY."}), 500

    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt file is required"}), 400

    try:
        # Read questions
        questions_file = request.files['questions.txt']
        user_questions = questions_file.read().decode('utf-8')

        # Handle optional data file
        df = None
        data_context = "No data file provided."
        
        for file_key in request.files:
            if file_key != 'questions.txt' and file_key.endswith('.csv'):
                try:
                    data_file = request.files[file_key]
                    df = pd.read_csv(data_file)
                    data_context = f"CSV file provided with {len(df)} rows and {len(df.columns)} columns.\nFirst 5 rows:\n{df.head().to_string()}\nColumns: {list(df.columns)}"
                    break
                except Exception as e:
                    return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

        # Prepare prompt
        full_prompt = f"""USER QUESTIONS:
{user_questions}

DATA CONTEXT:
{data_context}

Generate a Python script to answer these questions."""

        # Call OpenAI
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
        
        # Clean up markdown formatting
        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        if generated_code.startswith("```"):
            generated_code = generated_code[3:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        
        print("=== GENERATED CODE ===")
        print(generated_code)
        print("=====================")

        # Execute code
        result = safe_exec(generated_code, df)
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "Data Analyst Agent (OpenAI) is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)