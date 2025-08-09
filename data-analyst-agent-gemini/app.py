import os
import io
import re
import json
import base64
import signal
import types
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import multiprocessing as mp

import requests
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# LLM provider: Gemini
import google.generativeai as genai

# -----------------------------
# App init and config
# -----------------------------
load_dotenv()
app = Flask(__name__)

# Render/production: bind to PORT
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")
_gemini_ready = False
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_ready = True
except Exception as e:
    print(f"[WARN] Failed initializing Gemini client: {e}")

# -----------------------------
# Utility: safe exec sandbox (identical to OpenAI version)
# -----------------------------
ALLOWED_IMPORTS = {
    "io": io,
    "json": json,
    "re": re,
    "time": time,
    "math": __import__("math"),
    "datetime": __import__("datetime"),
    "base64": base64,
    "numpy": np,
    "np": np,
    "pandas": pd,
    "pd": pd,
    "matplotlib": matplotlib,
    "matplotlib.pyplot": plt,
    "seaborn": sns,
    "sns": sns,
    "sklearn": __import__("sklearn"),
    "sklearn.linear_model": __import__("sklearn.linear_model", fromlist=["LinearRegression"]),
    "requests": requests,
    "bs4": __import__("bs4", fromlist=["BeautifulSoup"]),
    "duckdb": None if os.environ.get("DISABLE_DUCKDB") == "1" else __import__("duckdb"),
}

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "print": print,
}


def _whitelist_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
    if name in ALLOWED_IMPORTS and ALLOWED_IMPORTS[name] is not None:
        return ALLOWED_IMPORTS[name]
    for allowed in list(ALLOWED_IMPORTS.keys()):
        if name.startswith(allowed + ".") and ALLOWED_IMPORTS.get(allowed) is not None:
            return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{name}' is not allowed")


class Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        def _handle(signum, frame):
            raise TimeoutError("Execution time limit exceeded")
        signal.signal(signal.SIGALRM, _handle)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        return False


def _child_exec(code_string: str, df: Optional[pd.DataFrame], conn):
    try:
        import io as _io
        import json as _json
        import re as _re
        import time as _time
        import base64 as _base64
        import numpy as _np
        import pandas as _pd
        import matplotlib as _matplotlib
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from sklearn.linear_model import LinearRegression as _LinReg
        import requests as _requests
        from bs4 import BeautifulSoup as _BeautifulSoup
        import seaborn as _sns

        def _wl_import(name, globals=None, locals=None, fromlist=(), level=0):
            allowed = {
                "io": _io,
                "json": _json,
                "re": _re,
                "time": _time,
                "base64": _base64,
                "numpy": _np,
                "np": _np,
                "pandas": _pd,
                "pd": _pd,
                "matplotlib": _matplotlib,
                "matplotlib.pyplot": _plt,
                "seaborn": _sns,
                "sns": _sns,
                "sklearn": __import__("sklearn"),
                "sklearn.linear_model": __import__("sklearn.linear_model", fromlist=["LinearRegression"]),
                "requests": _requests,
                "bs4": __import__("bs4", fromlist=["BeautifulSoup"]),
            }
            if name in allowed:
                return allowed[name]
            for k in list(allowed.keys()):
                if name.startswith(k + "."):
                    return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of '{name}' is not allowed")

        safe_globals = {
            "__builtins__": {**SAFE_BUILTINS, "__import__": _wl_import},
            "io": _io,
            "json": _json,
            "re": _re,
            "time": _time,
            "base64": _base64,
            "np": _np,
            "numpy": _np,
            "pd": _pd,
            "pandas": _pd,
            "plt": _plt,
            "LinearRegression": _LinReg,
            "requests": _requests,
            "BeautifulSoup": _BeautifulSoup,
            "matplotlib": _matplotlib,
            "df": (df.copy() if df is not None else None),
        }
        safe_locals = {"final_answers": None}
        exec(code_string, safe_globals, safe_locals)
        result = safe_locals.get("final_answers", None)
        if result is None:
            conn.send({"error": "Generated script did not set 'final_answers'."})
        else:
            conn.send(result)
    except Exception as e:
        conn.send({"error": "Failed to execute generated script", "details": str(e)})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _execute_in_subprocess(code_string: str, df: Optional[pd.DataFrame], timeout_seconds: int):
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_child_exec, args=(code_string, df, child_conn))
    p.daemon = True
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return {"error": "Execution time limit exceeded"}
    if parent_conn.poll(1):
        try:
            return parent_conn.recv()
        except EOFError:
            return {"error": "No result received from execution process"}
    return {"error": "No result received from execution process"}


def safe_exec(code_string: str, df: Optional[pd.DataFrame]) -> Any:
    timeout_seconds = int(os.environ.get("EXEC_TIMEOUT_SECONDS", "90"))
    if hasattr(signal, "SIGALRM"):
        try:
            class _Timeout:
                def __enter__(self_inner):
                    def _handle(signum, frame):
                        raise TimeoutError("Execution time limit exceeded")
                    signal.signal(signal.SIGALRM, _handle)
                    signal.alarm(timeout_seconds)
                def __exit__(self_inner, exc_type, exc, tb):
                    signal.alarm(0)
                    return False
            df_copy = df.copy() if df is not None else None
            safe_globals: Dict[str, Any] = {
                "__builtins__": {**SAFE_BUILTINS, "__import__": _whitelist_import},
                "io": io,
                "json": json,
                "re": re,
                "time": time,
                "base64": base64,
                "np": np,
                "numpy": np,
                "pd": pd,
                "pandas": pd,
                "plt": plt,
                "LinearRegression": LinearRegression,
                "requests": requests,
                "BeautifulSoup": BeautifulSoup,
                "matplotlib": matplotlib,
                "df": df_copy,
            }
            safe_locals: Dict[str, Any] = {"final_answers": None}
            with _Timeout():
                exec(code_string, safe_globals, safe_locals)
            result = safe_locals.get("final_answers", None)
            if result is None:
                return {"error": "Generated script did not set 'final_answers'."}
            return result
        except Exception:
            return _execute_in_subprocess(code_string, df, timeout_seconds)
    else:
        return _execute_in_subprocess(code_string, df, timeout_seconds)


# -----------------------------
# Prompt construction (identical content)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a senior Python data analyst. You will ONLY output raw Python code. "
    "No explanations, no markdown, no comments. The environment provides a pandas DataFrame named df "
    "when a tabular file is uploaded. Your job is to write a COMPLETE script that:\n"
    "- Imports needed libraries at the top (pandas as pd, numpy as np, matplotlib.pyplot as plt, io, base64, requests, bs4)\n"
    "- If a URL is mentioned, scrape/load the data (prefer pandas.read_html for simple tables)\n"
    "- Clean and transform the data as needed to answer the questions\n"
    "- Create any plots with matplotlib. Ensure file size under 100kB by using small figure size and dpi<=90.\n"
    "- Put ALL final results in a variable named final_answers at the end.\n"
    "- Match the exact output format requested (list or dict).\n"
    "- If required columns are missing or external data is not available, provide best-effort results or state that the data is insufficient.\n"
)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def df_preview(df: pd.DataFrame, max_rows: int = 5) -> str:
    try:
        info = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {c: str(df[c].dtype) for c in df.columns[:50]},
        }
        head_str = df.head(max_rows).to_string(index=False)[:4000]
        return (
            f"shape: {info['shape']}\n"
            f"columns: {info['columns']}\n"
            f"dtypes: {info['dtypes']}\n"
            f"head:\n{head_str}\n"
        )
    except Exception:
        return df.head(max_rows).to_string(index=False)


def load_primary_dataframe(incoming_files) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
    meta: List[Dict[str, Any]] = []
    primary_df: Optional[pd.DataFrame] = None

    for name, storage in incoming_files.items():
        try:
            filename = getattr(storage, "filename", name)
            content_type = getattr(storage, "content_type", "")
            size = storage.content_length or 0
            meta.append({"name": name, "filename": filename, "content_type": content_type, "size": size})

            lowered = (filename or name).lower()
            if primary_df is None and (lowered.endswith(".csv") or lowered.endswith(".tsv")):
                storage.stream.seek(0)
                sep = "," if lowered.endswith(".csv") else "\t"
                primary_df = pd.read_csv(storage.stream, sep=sep)
            elif primary_df is None and lowered.endswith(".parquet"):
                storage.stream.seek(0)
                primary_df = pd.read_parquet(storage.stream)
            elif primary_df is None and lowered.endswith(".json"):
                storage.stream.seek(0)
                try:
                    data_json = json.load(storage.stream)
                    if isinstance(data_json, list):
                        primary_df = pd.DataFrame(data_json)
                    elif isinstance(data_json, dict) and any(isinstance(v, list) for v in data_json.values()):
                        primary_df = pd.DataFrame(data_json)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Failed inspecting/reading file '{name}': {e}")
            continue

    return primary_df, meta


# -----------------------------
# HTTP handlers
# -----------------------------
@app.route("/", methods=["GET"])
def root_health():
    return "Data Analyst Agent (Gemini) is running. POST /api/"


@app.route("/api/", methods=["POST"])
def api_analyze():
    if not _gemini_ready:
        return jsonify({"error": "Gemini client not initialized. Set GEMINI_API_KEY"}), 500

    if "questions.txt" not in request.files:
        return jsonify({"error": "questions.txt is required"}), 400

    try:
        qf = request.files["questions.txt"]
        questions_text = qf.read().decode("utf-8", errors="ignore")
        questions_text = truncate_text(questions_text, 8000)

        df, files_meta = load_primary_dataframe(request.files)
        df_ctx = "No tabular file provided. If a URL is present in the question, fetch data from the web."
        if df is not None:
            df_ctx = df_preview(df)

        file_summaries = [
            {
                "filename": m.get("filename"),
                "type": (m.get("content_type") or ""),
                "size": int(m.get("size") or 0),
            }
            for m in files_meta
            if m.get("filename") != "questions.txt"
        ]

        user_prompt = (
            "System Instructions:\n" + SYSTEM_PROMPT + "\n\n" +
            "Questions:\n" + questions_text + "\n\n" +
            "DataFrame Context (if any):\n" + df_ctx + "\n\n" +
            "Other uploaded files (names/types only):\n" + json.dumps(file_summaries) + "\n"
        )

        # Gemini call
        gen_model = genai.GenerativeModel(GEMINI_MODEL)
        response = gen_model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "1200"))),
        )
        code = (response.text or "").strip()
        if code.startswith("```"):
            code = re.sub(r"^```[a-zA-Z0-9]*\n", "", code)
            if code.endswith("```"):
                code = code[:-3]
        code = code.strip()

        print("--- Generated code (Gemini) ---\n" + truncate_text(code, 3000) + "\n------------------------------")

        result = safe_exec(code, df)

        def _maybe_truncate_plot_uri(val: Any) -> Any:
            try:
                if isinstance(val, str) and val.startswith("data:image"):
                    return val
                return val
            except Exception:
                return val

        if isinstance(result, list):
            result = [_maybe_truncate_plot_uri(v) for v in result]
        elif isinstance(result, dict):
            result = {k: _maybe_truncate_plot_uri(v) for k, v in result.items()}

        return jsonify(result)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] /api/ failed:\n{tb}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=DEFAULT_PORT)