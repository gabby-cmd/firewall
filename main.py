from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
import re
import time
import logging
from openai import AsyncOpenAI

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
# Allowlist, Blocklist
# -----------------------------------------------------------------------------
ALLOW_LIST = [
    "stocks", "mutual funds", "investments", "equity", "debt", "finance", "economy", 
    "GDP", "wealth management", "savings", "401k", "retirement", "capital gains",
    "dividends", "index funds", "bonds", "commodities", "hedge fund", "venture capital",
    "angel investment", "portfolio", "fund manager", "pension", "real estate", 
    "stock exchange", "bull market", "bear market", "asset allocation", "inflation",
    "interest rates", "banking", "credit score", "financial planning", "net worth",
    "insurance", "annuities", "currency market", "crypto", "blockchain", "bitcoin",
    "private equity", "public offering", "treasury bonds", "forex", "financial literacy",
    "estate planning", "trust funds", "gold investment", "financial advisor", "market crash"
]

BLOCK_LIST = [
    "hack", "bomb", "attack", "terrorist", "kill", "weapon", "shoot", "kidnap", 
    "explosive", "drugs", "smuggle", "crime", "fraud", "assassinate", "poison",
    "blackmail", "hijack", "hostage", "murder", "cyberattack", "harassment",
    "money laundering", "bribe", "ransom", "extort", "illegal", "nuclear", 
    "bioweapon", "chemical weapon", "suicide", "genocide", "riot", "sedition",
    "treason", "arson", "anarchy", "sabotage", "espionage", "violence", "abduction",
    "counterfeit", "corruption", "smuggling", "prostitution", "drug trafficking",
    "child abuse", "hate crime", "gang violence", "pornography", "rape"
]

# -----------------------------------------------------------------------------
# Firewall Functions
# -----------------------------------------------------------------------------
def scan_for_allowlist(text: str):
    for word in ALLOW_LIST:
        if word.lower() in text.lower():
            return True
    return False

def scan_for_blocklist(text: str):
    for word in BLOCK_LIST:
        if word.lower() in text.lower():
            raise HTTPException(status_code=403, detail=f"Blocked by firewall: banned word detected")

def scan_for_pii(text: str):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    
    if re.search(email_pattern, text):
        raise HTTPException(status_code=403, detail="Blocked by firewall: Email detected")
    if re.search(phone_pattern, text):
        raise HTTPException(status_code=403, detail="Blocked by firewall: Phone number detected")
    if re.search(ssn_pattern, text):
        raise HTTPException(status_code=403, detail="Blocked by firewall: SSN detected")

def scan_for_secrets(text: str):
    secret_patterns = [
        r"sk-[A-Za-z0-9]{32,}",  # OpenAI API Key
        r"AKIA[0-9A-Z]{16}",     # AWS Key
        r"ghp_[A-Za-z0-9]{36}",  # GitHub Token
        r"AIza[0-9A-Za-z\-_]{35}", # Google API Key
        r"eyJ[a-zA-Z0-9-_=]+?\.[a-zA-Z0-9-_=]+\.?[a-zA-Z0-9-_.+/=]*$" # JWT
    ]
    for pattern in secret_patterns:
        if re.search(pattern, text):
            raise HTTPException(status_code=403, detail="Blocked by firewall: Secret detected")

# -----------------------------------------------------------------------------
# OpenAI Async Client
# -----------------------------------------------------------------------------
client = AsyncOpenAI()

async def call_openai(prompt: str):
    start_time = time.time()
    response = await client.chat.completions.create(
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
# API Endpoints
# -----------------------------------------------------------------------------

# Individual firewall checkers
@app.post("/test/allowlist")
async def test_allowlist(request: PromptRequest):
    if scan_for_allowlist(request.prompt):
        return {"detail": "allowed by firewall"}
    else:
        raise HTTPException(status_code=404, detail="Not Found")

@app.post("/test/blocklist")
async def test_blocklist(request: PromptRequest):
    try:
        scan_for_blocklist(request.prompt)
    except HTTPException:
        raise
    return {"detail": "Not Found"}

@app.post("/test/pii")
async def test_pii(request: PromptRequest):
    try:
        scan_for_pii(request.prompt)
    except HTTPException:
        raise
    return {"detail": "Not Found"}

@app.post("/test/secrets")
async def test_secrets(request: PromptRequest):
    try:
        scan_for_secrets(request.prompt)
    except HTTPException:
        raise
    return {"detail": "Not Found"}

# Main chatbot
@app.post("/process_prompt", response_model=RouteLLMResponse)
async def process_prompt(request: PromptRequest):
    logger.info(f"Received prompt: {request.prompt[:50]}...")

    # Firewall full scan but do not block unless critical
    try:
        scan_for_blocklist(request.prompt)
        scan_for_pii(request.prompt)
        scan_for_secrets(request.prompt)
    except HTTPException as e:
        logger.warning(f"Firewall blocked prompt: {e.detail}")
        raise

    llm_response = await call_openai(request.prompt)
    logger.info(f"Response returned in {llm_response.latency:.2f} seconds.")
    return llm_response

# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("✅ FastAPI Firewall Chatbot Server is running!")

