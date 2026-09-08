"""PII sanitization, mirroring n8n's "Guardrails1" node (sanitize operation,
entities CREDIT_CARD + EMAIL_ADDRESS by default - see PII_ENTITIES).

This is a lightweight regex-based stand-in for n8n's guardrails entity
detector. It's good enough to keep obvious PII out of the Gemini prompt, but
for production-grade detection swap in Google Cloud DLP or Microsoft
Presidio - the credit-card pattern in particular can false-positive on long
numeric identifiers (e.g. some PO/invoice numbers), which is why it's gated
on a Luhn checksum below rather than digit count alone.
"""
import logging
import re

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _sanitize_email(text: str) -> tuple[str, int]:
    return _EMAIL_RE.subn("[REDACTED_EMAIL]", text)


def _sanitize_credit_card(text: str) -> tuple[str, int]:
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"[ -]", "", match.group())
        if _luhn_valid(digits):
            count += 1
            return "[REDACTED_CREDIT_CARD]"
        return match.group()

    return _CREDIT_CARD_RE.sub(_replace, text), count


_SANITIZERS = {
    "EMAIL_ADDRESS": _sanitize_email,
    "CREDIT_CARD": _sanitize_credit_card,
}


def sanitize(text: str, entities: list[str] | None = None) -> str:
    entities = entities or settings.PII_ENTITIES
    redaction_counts = {}
    for entity in entities:
        sanitizer = _SANITIZERS.get(entity)
        if sanitizer:
            text, count = sanitizer(text)
            if count:
                redaction_counts[entity] = count
    logger.info(
        "guardrails.sanitize: entities=%s redactions=%s output_chars=%d",
        entities,
        redaction_counts,
        len(text),
    )
    return text
