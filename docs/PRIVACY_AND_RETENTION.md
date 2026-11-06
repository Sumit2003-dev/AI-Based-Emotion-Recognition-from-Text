# 🔒 Privacy, Security & Data Retention Policy

**Platform:** BeyondWord — AI-Based Emotion Recognition from Text  
**Maintainer:** Sumit Kumar ([@Sumit2003-dev](https://github.com/Sumit2003-dev))  
**Effective Date:** November 2026  

---

## 1. Core Privacy Principles

BeyondWord is engineered with a **privacy-first, edge-capable architecture**. The platform operates under the following commitments:

1. **Zero Unsolicited Data Retention**: Inference requests sent to the standard `/api/predict` and `/api/batch` endpoints are processed entirely in memory. Raw user text is never saved to disk, logged into persistent databases, or used for model training without explicit consent.
2. **Local CPU/GPU Execution**: The primary NLP intelligence engine runs locally via FastAPI. Unlike SaaS emotion APIs that send payloads to third-party endpoints, BeyondWord keeps text within your execution boundary.
3. **No PII Collection**: The platform does not collect, store, or profile personally identifiable information (PII) such as real names, email addresses, phone numbers, or IP addresses.

---

## 2. Active Learning & Human-Reviewed Feedback Controls

When a user submits voluntary feedback through the `/api/feedback` endpoint:

- **Explicit Opt-In**: Feedback is only stored when the user explicitly clicks "Submit Feedback".
- **Sanitization & Redaction**: Text submitted through the feedback pipeline undergoes automatic regex stripping for email addresses, URLs, and phone numbers before ingestion.
- **Review Queue**: All stored feedback items are marked as `pending_human_review`. No feedback sample is ingested into the training pipeline without two-person review and label adjudication.
- **Retention Period**: Unreviewed feedback items are purged after 90 days. Reviewed samples that are approved for training are transferred to the versioned training corpus with anonymous identifiers.
- **Right to Erasure (GDPR / CCPA)**: Users can request the removal of any submitted sample by referencing the unique submission `feedback_id`.

---

## 3. Sealed Holdout Security & Integrity

To preserve strict scientific integrity and avoid benchmark contamination:
- The **Sealed Test Set** (600 samples) is permanently locked in private storage outside public version control.
- Only its cryptographic SHA-256 hash (`5bb656cadeae936e157c59e90180d625c5613d6746a17c4ea321fca608c135d5`) and evaluation audit reports are published.
- The sealed test set is never accessed during model training, hyperparameter optimization, or temperature calibration.

---

## 4. Third-Party Fallback Disclosure

In the event that the local FastAPI backend is unreachable, the client frontend can optionally route requests through configured Cloud LLM providers (Groq, Google Gemini, OpenAI):
- This fallback only triggers if the host environment has explicitly configured valid API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).
- When activated, text payloads are transmitted over TLS 1.3 to the chosen provider under their respective enterprise zero-data-retention APIs.
- If no cloud API keys are present, the client falls back to a 100% offline, local browser heuristic engine.
