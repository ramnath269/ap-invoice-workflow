"""OCR + field extraction via Google Document AI.

Ports n8n's "Code in JavaScript" (base64 encode) and "HTTP Request3"
(Document AI :process call) nodes. DOCAI_PROCESSOR_ID now points at a custom
extractor processor that returns structured `document.entities` alongside
the raw OCR `document.text` in one call - see parse_entities() below for how
those entities are converted into the same canonical field shape
gemini_extract.parse_response() used to produce, so this replaces the old
OCR-only processor + separate Gemini extraction step without changing what
downstream graph nodes consume.
"""
import base64
import logging
import mimetypes
import re

import requests

from .google_auth import get_access_token
from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_MONEY_STRIP_RE = re.compile(r"[^\d.\-]")
_LEADING_SYMBOL_RE = re.compile(r"^([^\d\s.\-]+)")

# Entity types this processor's schema doesn't carry at all - defaulted the
# same way Gemini did when a field was absent from the source document.
_MISSING_TOP_LEVEL_DEFAULTS = {"purchase_order": "", "currency": "", "special_level_tax": "0.00"}
_MISSING_PRODUCT_DEFAULTS = {"unit_of_measure": "", "line_number": ""}

_TOP_LEVEL_MONEY_FIELDS = {
    "county_level_tax", "handling_fee", "sales_tax", "state_level_tax",
    "sub_total", "total_amount_due", "total_freight",
}
_TOP_LEVEL_TEXT_FIELDS = {"due_date", "invoice_date", "invoice_number", "payment_terms", "vendor"}

# DocAI's "item" entity carries the actual part/item number (JDE's
# first-choice identifier per jde_client.build_payload) -> canonical
# "item_number". Its "description" entity carries a short name (e.g.
# "SENSOR") -> canonical "item", matching the old Gemini convention where
# "item" was a short name and "item_number" carried the real identifier.
# "supplier_item_number" keeps its name - this processor populates it
# reliably (Gemini often left it blank).
_PRODUCT_MONEY_FIELDS = {"amount", "unit_price"}
_PRODUCT_ENTITY_TO_FIELD = {
    "item": "item_number",
    "description": "item",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "amount": "amount",
    "supplier_item_number": "supplier_item_number",
    "purchase_order_number": "purchase_order_number",
}


def _clean_text(value) -> str:
    if not value:
        return ""
    return value.replace("\n", " ").replace("\r", " ").strip()


def _clean_money(value) -> str:
    return _MONEY_STRIP_RE.sub("", _clean_text(value))


def _leading_symbol(value) -> str:
    match = _LEADING_SYMBOL_RE.match(_clean_text(value))
    return match.group(1) if match else ""


def encode_file(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        content = f.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "application/pdf"
    logger.info(
        "document_ai.encode_file: path=%s bytes=%d mime=%s", file_path, len(content), mime_type
    )
    return base64.b64encode(content).decode("utf-8"), mime_type


def process_document(base64_content: str, mime_type: str) -> dict:
    """Calls the Document AI processor and returns its full raw response
    (document.text plus, for a custom extractor processor, document.entities)."""
    url = (
        f"https://us-documentai.googleapis.com/v1/projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{settings.DOCAI_LOCATION}/processors/{settings.DOCAI_PROCESSOR_ID}:process"
    )
    logger.info(
        "document_ai.process_document: POST %s mime=%s encoded_len=%d", url, mime_type, len(base64_content)
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {get_access_token()}"},
        json={"rawDocument": {"content": base64_content, "mimeType": mime_type}},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    document = data.get("document", {})
    logger.info(
        "document_ai.process_document: received %d chars of OCR text, %d entities",
        len(document.get("text", "")),
        len(document.get("entities", [])),
    )
    return data


def run_ocr(base64_content: str, mime_type: str) -> str:
    return process_document(base64_content, mime_type)["document"]["text"]


def parse_entities(raw_response: dict) -> dict:
    """Converts this processor's structured document.entities into the same
    canonical field shape gemini_extract.parse_response()'s {"output": ...}
    used to produce - see the module docstring and the field-mapping notes
    above _PRODUCT_ENTITY_TO_FIELD for the specific renames/derivations."""
    entities = raw_response.get("document", {}).get("entities", [])

    output: dict = dict(_MISSING_TOP_LEVEL_DEFAULTS)
    products = []

    for entity in entities:
        etype = entity.get("type")
        if etype == "products":
            product = dict(_MISSING_PRODUCT_DEFAULTS)
            for prop in entity.get("properties", []):
                field = _PRODUCT_ENTITY_TO_FIELD.get(prop.get("type"))
                if not field:
                    continue
                text = _clean_text(prop.get("mentionText"))
                product[field] = _clean_money(text) if field in _PRODUCT_MONEY_FIELDS else text
            product.setdefault("description", "")
            products.append(product)
            continue

        text = _clean_text(entity.get("mentionText"))
        if etype in _TOP_LEVEL_MONEY_FIELDS:
            output[etype] = _clean_money(text)
            if etype == "total_amount_due":
                output["currency"] = _leading_symbol(text)
        elif etype in _TOP_LEVEL_TEXT_FIELDS:
            output[etype] = text

    output["products"] = products
    if products:
        output["purchase_order"] = products[0].get("purchase_order_number") or ""

    logger.info(
        "document_ai.parse_entities: parsed OK | fields=%s products=%d",
        list(output.keys()), len(products),
    )
    return {"output": output}
