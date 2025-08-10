from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import uvicorn
from llm_open import run_agent

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome to the API!"})

@app.post("/api")
async def analyze(request: Request):

    form = await request.form()
    question_text = None
    saved_files = {}

    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:  # It's a file
            async with aiofiles.open(value.filename, "wb") as f:
                await f.write(await value.read())
            saved_files[field_name] = value.filename

            if field_name == "questions.txt":
                async with aiofiles.open(value.filename, "r") as f:
                    question_text = await f.read()
        else:
            saved_files[field_name] = value

    if question_text is None and saved_files:
        first_file = next(iter(saved_files.values()))
        async with aiofiles.open(first_file, "r") as f:
            question_text = await f.read()
    x=[key for key in saved_files.keys() if key != "questions"]
    resp=run_agent(question_text)
    return resp




