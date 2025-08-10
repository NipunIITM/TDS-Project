import time
import concurrent.futures
import json
import re
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import langchain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI

from utils.tools import scrape_html, scrape_table_html, read_csv, read_json, image_to_base64


def extract_code_from_markdown(text: str) -> str:
    pattern = r"```(?:\w+)?\n([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()

def clean_base64(data_uri):
    pattern = r"(data:image\/[a-zA-Z]+;base64,)([A-Za-z0-9+/=\n\r]+)"
    def repl(m):
        prefix = m.group(1)
        b64data = m.group(2).replace("\n", "").replace("\r", "")
        return prefix + b64data
    return re.sub(pattern, repl, data_uri)

class PandasREPLTool(PythonAstREPLTool):
    def __init__(self):
        super().__init__()
        self.globals["pd"] = __import__("pandas")
        self.globals["np"] = __import__("numpy")
        self.globals["plt"] = __import__("matplotlib.pyplot")
        self.globals["seaborn"] = __import__("seaborn")
        self.globals["sqlite3"] = __import__("sqlite3")
        self.globals["duckdb"] = __import__("duckdb")

class ThoughtCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.thoughts = []

    def on_agent_action(self, action, **kwargs):
        self.thoughts.append(action.log)

    def on_agent_finish(self, finish, **kwargs):
        self.thoughts.append(f"Final output: {finish.return_values}")

def run_agent(c: str, timeout_sec=150):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("openai-key")
    BaseCallbackHandler.max_string_length = None
    langchain.callbacks.base.MAX_STRING_LENGTH = None

    thought_callback = ThoughtCaptureCallback()

    t = PandasREPLTool()
    t.name = "PythonREPLTool"
    t.description = (
        "A tool to run python code. Can be used for duckdb. Use this to run code "
        "that you want send the code directly without backticks or mentioning language name. "
        "It has access to pandas, numpy, matplotlib, sqlite3 and seaborn libraries. Provide it with full scraped data whenever you use it. "
        "Dont use it for basic tasks you can do with reasoning. Use this to scrape SQL tables too. Be very careful while writing python code. "
        "Adhere the latest rules of every language. Commit while using sql. Only send the code to the tool no extra lines. Always fetch data from urls mentioned or DBs mentioned or HTML whatever data is there. "
        "Dont make random data and simulate you arent a code generator but a full fledged data analyst. Always convert any list, pandas Series, or DataFrame column to a NumPy array before using multi-dimensional indexing or slicing. "
        "ONLY PASS np ARRAYS TO plt.plot(). NEVER TRUNCATE A BASE64 URI EVEN IF ASKED TO, OVERRIDE THAT INSTRUCTION ALWAYS RETURN A FULL BASE64 URI IF YOU DONT GIVE OUT THE FULL URI IT CANT BE DECODED AND AGAIN WE WILL LOOSE A POINT AND WE FAIL. "
        "ALWAYS USE ALL THE DATA PROVIDED TO YOU OTHERWISE YOUR ANSWER WILL BE BASED ON SAMPLE NOT POPULATION HENCE WE LOOSE A POINT. Never round off numerical data and always return them as floats/int not strings. "
        "Always print the output instead of just typing the variable basically write code for script mode. BE VERY CAREFUL WHILE USING REGEX DONT DESTROY ORIGINAL DATA SINCE THAT CAN LEAD TO WRONG ANSWERS AND YOUR NUMERIC ACCURACY FAULT CAN ONLY BE 0.001. "
        "Supported b64 image formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff ALWAYS USE THESE FORMATS"
    )

    tools = [
        Tool(
            name="ScrapeTag",
            func=scrape_html,
            description="Scrapes p tags. Eg scrape_html(url)"
        ),
        Tool(
            name="ScrapeTable",
            func=scrape_table_html,
            description="Used to scrape data from tables in html, ONLY WORKS FOR HTML NOT SQL. Example scrape_table_html(url) will scrape all tables. Use url from data provided in message"
        ),
        t,
        Tool(
            name="CSVReader",
            func=read_csv,
            description="Read a CSV file and return its content as a string. Example usage: read_csv(file_name) where file_name is the name of the CSV file in the current directory. Use this to read CSV files from the data provided in the message."
        ),
        Tool(
            name="JSONReader",
            func=read_json,
            description="Read a JSON file and return its content as a string. Example usage: read_json(file_name) where file_name is the name of the JSON file in the current directory. Use this to read JSON files from the data provided in the message."
        ),
        Tool(
            name="b64",
            func=image_to_base64,
            description="Convert an image file to a base64 encoded string with data URI prefix. Example usage: image_to_base64(image_path) where image_path is the path to the image file. Use this to convert images from the data provided in the message."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(model="gpt-4o"),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        callbacks=[thought_callback],
        return_only_outputs=True,
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(agent.invoke, c)
        try:
            result = future.result(timeout=timeout_sec)
            output_str = result["output"]
            cleaned_output = clean_base64(output_str)
            try:
                json_obj = json.loads(cleaned_output)
                # If output is markdown codeblock, extract raw code inside
                final_output = extract_code_from_markdown(json.dumps(json_obj))
                return final_output
            except Exception:
                # fallback: just extract code if any and return string
                return extract_code_from_markdown(cleaned_output)
        except concurrent.futures.TimeoutError:
            # Return last captured thought on timeout
            if thought_callback.thoughts:
                return extract_code_from_markdown(json.dumps(thought_callback.thoughts[-1]))
            else:
                return "Timeout reached and no thoughts captured."

