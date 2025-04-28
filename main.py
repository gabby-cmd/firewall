from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
import time
import logging

import httpx
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from detect_secrets import SecretsCollection

from firewall_lists import BLOCK_LIST, ALLOW_LIST

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

if not OPENAI_API_KEY or not PERSPECTIVE_API_KEY:
    raise EnvironmentError("Environment variables missing: OPENAI_API_KEY and/or PERSPECTIVE_API_KEY.")

openai.api_key = OPENAI_API_KEY

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Initialize Security Scanners
# -----------------------------------------------------------------------------
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ]
}

nlp_engine = SpacyNlpEngine()
nlp_engine.load(nlp_configuration)

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

PERSPECTIVE_ENDPOINT = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

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
def scan_for_blocklist(text: str):
    for word in BLOCK_LIST:
        if word.lower() in text.lower():
            raise HTTPException(status_code=403, detail=f"Banned word detected: {word}")

def scan_for_allowlist(text: str):
    for word in ALLOW_LIST:
        if word.lower() in text.lower():
            return True
    return False

def scan_for_pii(text: str):
    results = analyzer.analyze(text=text, language='en')
    if results:
        raise HTTPException(status_code=403, detail=f"Blocked: PII detected ({[e.entity_type for e in results]})")

def scan_for_secrets(text: str):
    secrets = SecretsCollection()
    secrets.scan_text(text)
    if secrets.data:
        raise HTTPException(status_code=403, detail="Blocked: Potential secrets detected.")

def scan_for_toxicity(text: str):
    headers = {"Content-Type": "application/json"}
    data = {
        "comment": {"text": text},
        "requestedAttributes": {"TOXICITY": {}},
        "languages": ["en"]
    }
    params = {"key": PERSPECTIVE_API_KEY}

    response = httpx.post(PERSPECTIVE_ENDPOINT, headers=headers, json=data, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Perspective API call failed.")

    scores = response.json()
    toxicity_score = scores["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

    if toxicity_score >= 0.8:
        raise HTTPException(status_code=403, detail="Blocked: Toxic content detected.")

def scan_input(prompt: str):
    scan_for_blocklist(prompt)
    scan_for_pii(prompt)
    scan_for_secrets(prompt)
    scan_for_toxicity(prompt)

    if not scan_for_allowlist(prompt):
        raise HTTPException(status_code=403, detail="Blocked by firewall: prompt not allowed.")

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
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/process_prompt", response_model=RouteLLMResponse)
async def process_prompt(request: PromptRequest):
    logger.info(f"Received prompt: {request.prompt[:50]}...")

    scan_input(request.prompt)

    llm_response = call_openai(request.prompt)

    logger.info(f"Response returned in {llm_response.latency:.2f} seconds.")
    return llm_response

# -----------------------------------------------------------------------------
# Startup Event
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("✅ FastAPI Chatbot Server with Full Firewall is running!")
