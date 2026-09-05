"""Structured extraction via Gemini on Vertex AI.

Ports n8n's "Gemini Vertex AI1" (schema-constrained generateContent call)
and "Parse Gemini JSON1" (strip markdown fences, JSON.parse) nodes.
"""
import json
import logging
import re

import requests

from .google_auth import get_access_token
from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_INSTRUCTIONS = """You are an information extraction engine.
You must extract fields EXACTLY according to the provided JSON schema.

RULES:
1. Do NOT add any fields not present in the schema.
2. Do NOT rename fields.
3. Do NOT infer or create new keys.
4. If a field is missing in the document, return an empty string.
5. For arrays, each object MUST contain ONLY the fields defined in the schema.
6. Return the final output as VALID JSON that exactly matches the schema.
7. Never include explanations, comments, or text outside of the JSON.
8. If a value is numerical or date-like, still return it as string (schema requires string).
9. Unknown or unreadable fields -> return empty string.
10. For products array:
    - Only include: item, description, quantity, unit_price, amount, unit_of_measure or uom or u/m or unit (eg. FT, EA), line_number, supplier_item_number, purchase_order_number.
    - Remove amount, rate, tax, hsn, or any other extra keys.
11. If invoice number contains text character include that as well.
12. If total amount due includes currency type remove the currency and also special characters.
13. Get the currency type from total amount due.
14. Item means item_name and not description.
15. Customer order number and purchase order number are same.
16. If the value of purchase order contains additional suffixes, revisions, sequence identifiers, line-item indicators, or other segments separated by special characters (such as -, /, _, ., or spaces), return only the primary/base purchase order number that uniquely identifies the order in the source system.
STRING ESCAPING CONSTRAINT: All extracted text fields must be safe for automated parsing.
17. The table contains separate columns for MFR (Manufacturer Code) and Item #. Even if they appear together in the raw text, extract MFR into mfr_code and the item identifier into item_number. Do not concatenate them.

Here is the document:
"""


def extract(document_text: str, response_schema: dict) -> dict:
    url = (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{settings.VERTEX_PROJECT_ID}"
        f"/locations/{settings.VERTEX_LOCATION}/publishers/google/models/"
        f"{settings.GEMINI_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": _INSTRUCTIONS + document_text}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
            "responseSchema": response_schema,
        },
    }
    logger.info(
        "gemini_extract.extract: POST %s model=%s doc_chars=%d schema_fields=%s",
        url,
        settings.GEMINI_MODEL,
        len(document_text),
        list(response_schema.get("properties", {}).keys()),
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=body,
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info(
        "gemini_extract.extract: response received | usage=%s finish_reason=%s",
        data.get("usageMetadata"),
        data.get("candidates", [{}])[0].get("finishReason"),
    )
    return data


def parse_response(raw_response: dict) -> dict:
    """Returns {"output": {...}} on success, or
    {"parse_error": True, "error": ..., "raw_output": ...} on failure -
    same contract as n8n's "Parse Gemini JSON1" node.
    """
    text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        output = json.loads(cleaned)
        logger.info("gemini_extract.parse_response: parsed OK | fields=%s", list(output.keys()))
        return {"output": output}
    except json.JSONDecodeError as exc:
        logger.error("gemini_extract.parse_response: JSON parse failed | error=%s", exc)
        return {"parse_error": True, "error": str(exc), "raw_output": text}
