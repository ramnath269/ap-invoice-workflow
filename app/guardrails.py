"""PII sanitization, mirroring n8n's "Guardrails1" node (sanitize operation,
entities CREDIT_CARD + EMAIL_ADDRESS by default - see PII_ENTITIES).

This is a lightweight regex-based stand-in for n8n's guardrails entity
detector. It's good enough to keep obvious PII out of the Gemini prompt, but
for production-grade detection swap in Google Cloud DLP or Microsoft
Presidio - the credit-card pattern in particular can false-positive on long
numeric identifiers (e.g. some PO/invoice numbers).
"""
import re

from .settings import settings

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_SANITIZERS = {
    "EMAIL_ADDRESS": (_EMAIL_RE, "[REDACTED_EMAIL]"),
    "CREDIT_CARD": (_CREDIT_CARD_RE, "[REDACTED_CREDIT_CARD]"),
}


def sanitize(text: str, entities: list[str] | None = None) -> str:
    for entity in entities or settings.PII_ENTITIES:
        pattern_and_replacement = _SANITIZERS.get(entity)
        if pattern_and_replacement:
            pattern, replacement = pattern_and_replacement
            text = pattern.sub(replacement, text)
    return text
