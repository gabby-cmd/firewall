from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
import time
import logging
import re

from firewall_lists import BLOCK_LIST, ALLOW_LIST

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

openai.api_key = OPENAI_API_KEY

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str

class RouteLLMResponse(BaseModel):
    response: str
    model_used: str
    latency: float
    input_tokens: int
    output_tokens: int

# -----------------------------------------------------------------------------
# Firewall Functions
# -----------------------------------------------------------------------------
def check_allowlist(text: str):
    for word in ALLOW_LIST:
        if word.lower() in text.lower():
            return True
    return False

def check_blocklist(text: str):
    for word in BLOCK_LIST:
        if word.lower() in text.lower():
            return True
    return False

def check_pii(text: str):
    patterns = {
        "Email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "Phone Number": r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "Credit Card": r"\b(?:\d[ -]*?){13,16}\b",
        "Name": r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b"
    }
    for pii_type, pattern in patterns.items():
        if re.search(pattern, text):
            return pii_type
    return None

def check_secrets(text: str):
    secret_patterns = [
        r"sk-[A-Za-z0-9]{20,40}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"ghp_[A-Za-z0-9]{36}",
        r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]*"
    ]
    for pattern in secret_patterns:
        if re.search(pattern, text):
            return True
    return False

def full_scan(prompt: str):
    if not check_allowlist(prompt):
        raise HTTPException(status_code=403, detail="Blocked by firewall: prompt not allowed (finance-related keywords missing).")

    if check_blocklist(prompt):
        raise HTTPException(status_code=403, detail="Blocked: banned word detected.")

    pii_type = check_pii(prompt)
    if pii_type:
        raise HTTPException(status_code=403, detail=f"Blocked: {pii_type} detected.")

    if check_secrets(prompt):
        raise HTTPException(status_code=403, detail="Blocked: potential secret detected.")

# -----------------------------------------------------------------------------
# OpenAI Call
# -----------------------------------------------------------------------------
def call_openai(prompt: str):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.time()

    message = response.choices[0].message.content
    latency = end_time - start_time

    return RouteLLMResponse(
        response=message,
        model_used="gpt-3.5-turbo",
        latency=latency,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )

# -----------------------------------------------------------------------------
# Testing Endpoints for Each Firewall Check
# -----------------------------------------------------------------------------
@app.post("/test/allowlist")
async def test_allowlist(request: PromptRequest):
    if not check_allowlist(request.prompt):
        raise HTTPException(status_code=403, detail="Blocked: not in allowlist")
    return {"message": "✅ Allowed based on allowlist"}

@app.post("/test/blocklist")
async def test_blocklist(request: PromptRequest):
    if check_blocklist(request.prompt):
        raise HTTPException(status_code=403, detail="Blocked: banned word detected")
    return {"message": "✅ No banned words found"}

@app.post("/test/pii")
async def test_pii(request: PromptRequest):
    pii_type = check_pii(request.prompt)
    if pii_type:
        raise HTTPException(status_code=403, detail=f"Blocked: {pii_type} detected")
    return {"message": "✅ No PII found"}

@app.post("/test/secrets")
async def test_secrets(request: PromptRequest):
    if check_secrets(request.prompt):
        raise HTTPException(status_code=403, detail="Blocked: secret detected")
    return {"message": "✅ No secrets found"}

# -----------------------------------------------------------------------------
# Final Combined Firewall + OpenAI Endpoint
# -----------------------------------------------------------------------------
@app.post("/process_prompt", response_model=RouteLLMResponse)
async def process_prompt(request: PromptRequest):
    logger.info(f"Received prompt: {request.prompt[:50]}...")

    full_scan(request.prompt)

    llm_response = call_openai(request.prompt)

    logger.info(f"Response returned in {llm_response.latency:.2f} seconds.")
    return llm_response

# -----------------------------------------------------------------------------
# Startup Event
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("✅ FastAPI Chatbot Server with Modular Firewall is running!")
