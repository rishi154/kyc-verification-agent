# KYC Verification Agent

AI-assisted KYC verification agent for merchant onboarding with responsible AI governance practices.

## What it does

- Verifies merchant identity documents for KYC compliance
- Screens entities against OFAC, EU, and UN sanctions watchlists
- Assesses onboarding risk using AI with human-in-the-loop escalation
- No PII is sent to the LLM — only anonymized business context

## Governance Practices

| Practice | Implementation |
|----------|---------------|
| Human-in-the-loop | Low-confidence results escalated to human review |
| Model transparency | `/tools/model_card` endpoint with limitations and intended use |
| Output guardrails | Blocked topics filtered, PII redacted from reasoning |
| Bias mitigation | No proxy variables; fairness tested across 12 nationality groups |
| Explainability | Full decision audit log at `/tools/audit_log` |
| Source attribution | Watchlist sources cited with last-updated dates |
| Model fallback | Rules-based fallback when LLM unavailable (fail-closed) |
| Data minimization | Only anonymized business context sent to LLM |
| AI disclosure | Every response includes AI disclosure notice |
| Version pinning | Model version pinned via `KYC_LLM_MODEL` env var |

## Quick Start

```bash
pip install -r requirements.txt

# Optional: configure
export KYC_LLM_MODEL=gemini-2.0-flash
export KYC_API_KEY=your-secure-key

python agent.py
# → http://localhost:8202
```

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/tools/verify_identity` | POST | API Key | KYC identity verification |
| `/tools/screen_watchlist` | POST | API Key | Sanctions watchlist screening |
| `/tools/model_card` | GET | None | Model transparency metadata |
| `/tools/audit_log` | GET | API Key | Decision audit trail |
| `/health` | GET | None | Health check |
| `/receive` | POST | None | A2A message receiver |

## Registration

On startup, the agent self-registers with:
- A2A Server (localhost:9000)
- Agent Marketplace (localhost:8500)
