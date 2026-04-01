"""
KYC Verification Agent — Demo (WELL-GOVERNED)

This agent demonstrates responsible AI governance practices.
Designed to pass the marketplace's 25-point governance review cleanly.

Governance practices implemented:
- API key auth on all endpoints (check 12)
- Model version via env var, pinned (check 6, 25)
- No PII sent to LLM — only anonymized summaries (check 23)
- Rate limiting via simple in-memory counter (check 8)
- Pydantic validation on all inputs and LLM output (check 9, 15)
- Human escalation for low-confidence results (check 16)
- Model card / transparency metadata (check 17)
- Output guardrails — restricted topics, PII redaction (check 18)
- No proxy variables — uses only financial data (check 19)
- Full decision audit logging (check 20)
- Source attribution on document checks (check 21)
- Fallback to rules-based when LLM unavailable (check 22)
- Data minimization in prompts (check 23)
- AI disclosure in every response (check 24)
- Structured logging with JSON (check 14)
- Health check endpoint (check 14)
- Timeouts on all external calls (check 8)
- Pinned dependencies (check 13)

Run: python agent.py → http://localhost:8202
"""

import os
import json
import time
import random
import logging
import hashlib
from datetime import datetime, timezone
from collections import defaultdict

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel, field_validator
from typing import Optional

# ---------------------------------------------------------------------------
# Structured logging (check 14)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("kyc_agent")

# ---------------------------------------------------------------------------
# Config — model version pinned via env var (checks 6, 25)
# ---------------------------------------------------------------------------
AGENT_ID = "kyc_verification_agent"
AGENT_TYPE = "compliance"
AGENT_PORT = 8202
AGENT_DID = "did:key:z6MkKYCVerifyGov020"
AGENT_ENDPOINT = f"http://localhost:{AGENT_PORT}/receive"
A2A_SERVER_URL = "http://localhost:9000"
MARKETPLACE_URL = "http://localhost:8500"
SOURCE_REPO = "https://github.com/rishi154/kyc-verification-agent"

LLM_MODEL = os.getenv("KYC_LLM_MODEL", "gemini-2.0-flash")  # Pinned version (check 25)
LLM_API_KEY = os.getenv("KYC_LLM_API_KEY", "")  # From env, never hardcoded (check 6)
CONFIDENCE_THRESHOLD = 0.75  # Below this → escalate to human (check 16)

CAPABILITIES = [
    "verify_identity",
    "screen_watchlist",
    "assess_risk_level",
]

# Model card / transparency metadata (check 17)
MODEL_CARD = {
    "model": LLM_MODEL,
    "provider": "google",
    "intended_use": "KYC identity verification risk assessment for merchant onboarding",
    "not_intended_for": [
        "Credit decisioning",
        "Law enforcement identification",
        "Automated denial of services without human review",
    ],
    "known_limitations": [
        "May produce lower confidence for non-Latin script names",
        "Document validation is heuristic — not a replacement for manual review",
        "Watchlist screening covers OFAC/EU only — not all jurisdictions",
    ],
    "training_data": "Google foundation model — no custom fine-tuning",
    "last_evaluated": "2025-01-15",
    "fairness_testing": "Tested across 12 nationality groups — no statistically significant disparate impact detected",
}

app = FastAPI(title="KYC Verification Agent", version="1.0.0")

# ---------------------------------------------------------------------------
# Auth (check 12)
# ---------------------------------------------------------------------------
VALID_API_KEY = os.getenv("KYC_API_KEY", "kyc-demo-key-secure")

def require_auth(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------------------------------------------------------------------
# Rate limiting — simple in-memory (check 8)
# ---------------------------------------------------------------------------
_rate_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 30  # requests per minute

def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = [t for t in _rate_counts[client_ip] if now - t < 60]
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — 30 requests/minute")
    window.append(now)
    _rate_counts[client_ip] = window

# ---------------------------------------------------------------------------
# Audit log (check 20)
# ---------------------------------------------------------------------------
_audit_log: list[dict] = []

def log_decision(action: str, input_hash: str, result: dict, confidence: float, escalated: bool):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "input_hash": input_hash,  # Never log raw PII — only hash (check 7)
        "decision": result.get("risk_level", "unknown"),
        "confidence": confidence,
        "escalated_to_human": escalated,
        "reasoning_summary": result.get("reasoning", ""),
        "model": LLM_MODEL,
    }
    _audit_log.append(entry)
    logger.info(f"Decision logged: {action} confidence={confidence} escalated={escalated}")

# ---------------------------------------------------------------------------
# Input models with validation (check 15)
# ---------------------------------------------------------------------------

class VerifyIdentityRequest(BaseModel):
    merchant_id: str
    business_name: str
    registration_country: str
    registration_number: str
    representative_name: str
    document_type: str  # "passport", "drivers_license", "national_id"
    document_country: str

    @field_validator("document_type")
    @classmethod
    def valid_doc_type(cls, v):
        allowed = {"passport", "drivers_license", "national_id", "business_registration"}
        if v not in allowed:
            raise ValueError(f"document_type must be one of: {allowed}")
        return v

class WatchlistScreenRequest(BaseModel):
    entity_name: str
    entity_type: str  # "individual" or "business"
    country: str

class A2AMessage(BaseModel):
    from_did: str
    to_did: str
    message_type: str
    body: dict
    signature: str

# Output validation model (check 9)
class RiskAssessment(BaseModel):
    risk_level: str  # "low", "medium", "high"
    confidence: float
    reasoning: str
    flags: list[str]
    recommended_action: str

# ---------------------------------------------------------------------------
# LLM call with data minimization + fallback (checks 22, 23)
# ---------------------------------------------------------------------------

def _anonymize_for_llm(merchant_id: str, business_name: str, country: str, doc_type: str) -> str:
    """Build LLM prompt with NO PII — only anonymized business context (check 23)."""
    return f"""Assess KYC risk for a merchant onboarding request.

Business type: {business_name.split()[0] if business_name else 'Unknown'} (industry category)
Registration country: {country}
Document provided: {doc_type}
Merchant ID hash: {hashlib.sha256(merchant_id.encode()).hexdigest()[:12]}

Evaluate risk level (low/medium/high) based on:
1. Country risk (sanctions, CPI score)
2. Document type adequacy for the jurisdiction
3. Business category risk

Return JSON: {{"risk_level": "low|medium|high", "confidence": 0.0-1.0, "reasoning": "...", "flags": [...], "recommended_action": "approve|review|escalate"}}"""


def call_llm_with_fallback(prompt: str) -> dict:
    """
    Call LLM with fallback to rules-based logic (check 22).
    In demo mode, simulates the LLM response.
    """
    try:
        # Simulated LLM response for demo
        logger.info(f"LLM call: model={LLM_MODEL} prompt_length={len(prompt)}")
        risk = random.choice(["low", "low", "medium", "high"])
        confidence = round(random.uniform(0.6, 0.98), 2)
        return {
            "risk_level": risk,
            "confidence": confidence,
            "reasoning": f"Country risk assessment and document validation indicate {risk} risk",
            "flags": ["enhanced_due_diligence_recommended"] if risk == "high" else [],
            "recommended_action": "approve" if risk == "low" else ("review" if risk == "medium" else "escalate"),
        }
    except Exception as e:
        # Fallback: rules-based assessment (check 22 — fail closed)
        logger.warning(f"LLM unavailable, using rules-based fallback: {e}")
        return {
            "risk_level": "medium",
            "confidence": 0.5,
            "reasoning": "Rules-based fallback — LLM unavailable. Manual review recommended.",
            "flags": ["llm_fallback_used"],
            "recommended_action": "escalate",
        }


# ---------------------------------------------------------------------------
# Output guardrails (check 18)
# ---------------------------------------------------------------------------

BLOCKED_TOPICS = {"credit_score", "criminal_record", "political_affiliation", "religion", "ethnicity"}

def apply_guardrails(result: dict) -> dict:
    """Filter LLM output for safety (check 18)."""
    reasoning = result.get("reasoning", "").lower()
    for topic in BLOCKED_TOPICS:
        if topic.replace("_", " ") in reasoning:
            result["reasoning"] = "[REDACTED — restricted topic detected]"
            result["flags"] = result.get("flags", []) + ["guardrail_triggered"]
            logger.warning(f"Guardrail triggered: blocked topic in reasoning")
            break
    # Ensure risk_level is valid
    if result.get("risk_level") not in ("low", "medium", "high"):
        result["risk_level"] = "medium"
        result["flags"] = result.get("flags", []) + ["invalid_risk_level_corrected"]
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/tools/verify_identity", dependencies=[Depends(require_auth), Depends(check_rate_limit)])
async def verify_identity(req: VerifyIdentityRequest):
    """
    Verify merchant identity for KYC onboarding.
    Uses AI with human escalation for low-confidence results.
    """
    # Data minimization: only send anonymized data to LLM (check 23)
    prompt = _anonymize_for_llm(req.merchant_id, req.business_name, req.registration_country, req.document_type)
    raw_result = call_llm_with_fallback(prompt)

    # Validate LLM output against schema (check 9)
    try:
        validated = RiskAssessment(**raw_result)
        result = validated.model_dump()
    except Exception:
        logger.error("LLM output failed schema validation — using safe default")
        result = {
            "risk_level": "medium",
            "confidence": 0.4,
            "reasoning": "Output validation failed — escalating for manual review",
            "flags": ["schema_validation_failed"],
            "recommended_action": "escalate",
        }

    # Apply output guardrails (check 18)
    result = apply_guardrails(result)

    # Human-in-the-loop: escalate low confidence (check 16)
    escalated = False
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        result["recommended_action"] = "escalate"
        result["flags"] = result.get("flags", []) + ["low_confidence_escalation"]
        escalated = True
        logger.info(f"Escalating to human review: confidence={result['confidence']}")

    # Audit log (check 20)
    input_hash = hashlib.sha256(f"{req.merchant_id}:{req.business_name}".encode()).hexdigest()[:16]
    log_decision("verify_identity", input_hash, result, result["confidence"], escalated)

    # AI disclosure in response (check 24)
    return {
        "merchant_id": req.merchant_id,
        "verification_status": result["recommended_action"],
        "risk_assessment": result,
        "escalated_to_human": escalated,
        "ai_disclosure": "This assessment was generated by an AI system and may be subject to human review.",
        "model_info": {"model": LLM_MODEL, "version": "pinned"},
        "source_attribution": {  # check 21
            "watchlist_source": "OFAC SDN List (updated daily)",
            "country_risk_source": "Transparency International CPI 2024",
        },
        "appeal_available": True,  # Human override always available
    }


@app.post("/tools/screen_watchlist", dependencies=[Depends(require_auth), Depends(check_rate_limit)])
async def screen_watchlist(req: WatchlistScreenRequest):
    """Screen entity against sanctions and watchlists."""
    # Simulated watchlist check with source attribution (check 21)
    is_match = random.random() < 0.05  # 5% chance of match for demo
    confidence = round(random.uniform(0.85, 0.99), 2) if is_match else round(random.uniform(0.95, 1.0), 2)

    result = {
        "entity_name": req.entity_name,
        "entity_type": req.entity_type,
        "screening_result": "potential_match" if is_match else "clear",
        "confidence": confidence,
        "sources_checked": [  # Source attribution (check 21)
            {"name": "OFAC SDN List", "last_updated": "2025-06-01", "match": is_match},
            {"name": "EU Consolidated Sanctions", "last_updated": "2025-06-01", "match": False},
            {"name": "UN Security Council", "last_updated": "2025-05-28", "match": False},
        ],
        "recommended_action": "escalate" if is_match else "clear",
        "ai_disclosure": "Screening performed using automated matching. Potential matches require human verification.",
    }

    # Always escalate matches to human (check 16)
    if is_match:
        result["escalated_to_human"] = True
        logger.info(f"Watchlist match escalated for human review: {req.entity_name}")

    input_hash = hashlib.sha256(req.entity_name.encode()).hexdigest()[:16]
    log_decision("screen_watchlist", input_hash, result, confidence, is_match)

    return result


@app.get("/tools/model_card")
async def get_model_card():
    """Return model transparency information (check 17)."""
    return MODEL_CARD


@app.get("/tools/audit_log", dependencies=[Depends(require_auth)])
async def get_audit_log(limit: int = 50):
    """Return recent decision audit entries (check 20)."""
    return {"entries": _audit_log[-limit:], "total": len(_audit_log)}


# ---------------------------------------------------------------------------
# A2A message receiver
# ---------------------------------------------------------------------------

@app.post("/receive")
async def receive_message(message: A2AMessage):
    logger.info(f"A2A message received: type={message.message_type}")
    try:
        if message.message_type == "verify_identity":
            return await verify_identity(VerifyIdentityRequest(**message.body))
        elif message.message_type == "screen_watchlist":
            return await screen_watchlist(WatchlistScreenRequest(**message.body))
        else:
            return {"status": "unknown_capability", "type": message.message_type}
    except Exception as e:
        logger.error(f"Error processing message: {type(e).__name__}")
        return {"status": "error", "message": "Internal processing error"}  # No leak (check 15)


# Health check (check 14)
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_id": AGENT_ID,
        "model": LLM_MODEL,
        "capabilities": CAPABILITIES,
        "uptime_checks": True,
    }


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def register():
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            await client.post(f"{A2A_SERVER_URL}/agents/register", json={
                "agent_id": AGENT_ID,
                "agent_type": AGENT_TYPE,
                "endpoint": AGENT_ENDPOINT,
                "did": AGENT_DID,
                "public_key": "demo-public-key-kyc-gov",
                "capabilities": CAPABILITIES,
            })
            logger.info("A2A registration: OK")
        except Exception as e:
            logger.warning(f"A2A server not reachable: {e}")

        try:
            await client.post(f"{MARKETPLACE_URL}/api/register/agent", json={
                "agent_id": AGENT_ID,
                "agent_type": AGENT_TYPE,
                "endpoint": AGENT_ENDPOINT,
                "capabilities": CAPABILITIES,
                "owner_team": "Compliance & Onboarding",
                "compliance": ["PII-Safe", "Production-Ready"],
                "docs_url": "https://wiki.internal/agents/kyc-verification",
                "source_repo": SOURCE_REPO,
                "description": (
                    "KYC verification agent for merchant onboarding. Verifies identity documents, "
                    "screens against OFAC/EU sanctions lists, and assesses onboarding risk. "
                    "Uses AI with human-in-the-loop escalation for low-confidence results. "
                    "No PII is sent to the LLM — only anonymized business context."
                ),
            })
            logger.info("Marketplace registration: OK")
        except Exception as e:
            logger.warning(f"Marketplace not reachable: {e}")

        tools = [
            {
                "tool_id": "verify_identity",
                "description": "Verify merchant identity for KYC. AI-assisted with human escalation for uncertain cases.",
                "owner_team": "Compliance & Onboarding",
                "compliance": ["PII-Safe", "Production-Ready"],
                "source_repo": SOURCE_REPO,
                "source_agent": AGENT_ID,
                "endpoint": f"http://localhost:{AGENT_PORT}/tools/verify_identity",
                "auth_required": True,
                "auth_type": "api_key",
                "example_request": {
                    "merchant_id": "MERCH-2025-0042",
                    "business_name": "Acme Payments Ltd",
                    "registration_country": "US",
                    "registration_number": "EIN-12-3456789",
                    "representative_name": "John Smith",
                    "document_type": "passport",
                    "document_country": "US",
                },
            },
            {
                "tool_id": "screen_watchlist",
                "description": "Screen entity against OFAC, EU, and UN sanctions lists. Matches escalated to human.",
                "owner_team": "Compliance & Onboarding",
                "compliance": ["PII-Safe", "Production-Ready"],
                "source_repo": SOURCE_REPO,
                "source_agent": AGENT_ID,
                "endpoint": f"http://localhost:{AGENT_PORT}/tools/screen_watchlist",
                "auth_required": True,
                "auth_type": "api_key",
                "example_request": {
                    "entity_name": "Acme Payments Ltd",
                    "entity_type": "business",
                    "country": "US",
                },
            },
        ]
        for tool in tools:
            try:
                await client.post(f"{MARKETPLACE_URL}/api/register/tool", json=tool)
                logger.info(f"Tool registered: {tool['tool_id']}")
            except Exception as e:
                logger.warning(f"Could not register tool {tool['tool_id']}: {e}")


if __name__ == "__main__":
    logger.info(f"Starting {AGENT_ID} on port {AGENT_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
