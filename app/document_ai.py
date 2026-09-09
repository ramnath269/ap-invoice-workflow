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
# "item". Its "description" entity carries a short name (e.g. "SENSOR")
# -> canonical "description". "supplier_item_number" keeps its own name -
# this processor populates it reliably (Gemini often left it blank).
_PRODUCT_MONEY_FIELDS = {"amount", "unit_price"}
_PRODUCT_ENTITY_TO_FIELD = {
    "item": "item",
    "description": "description",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "amount": "amount",
    "supplier_item_number": "supplier_item_number",
    "purchase_order_number": "purchase_order_number",
}

# This processor's foundation-model backend isn't perfectly deterministic:
# the same bytes can come back on one call with every entity populated and
# on the next with a property silently absent (see graph.py's extract_fields
# retry). CONFIDENCE_THRESHOLD and the two sets below are what
# find_extraction_issues() checks before we trust a response - this is a
# manually-kept-in-sync snapshot of the processor's schema
# (projects/.../schemas/2fd6b5078c2734d4), not a live fetch, so re-check it
# here after any schema edit in the Document AI console. As of
# schemaVersions/524ee19051f45a55: top level is REQUIRED_ONCE for
# invoice_date/total_amount_due/invoice_number only (vendor was
# REQUIRED_ONCE, now OPTIONAL_ONCE - it was coming back empty on every call
# for at least one real vendor's invoice layout, making it a useless retry
# trigger); every other top-level field and, previously, every product
# property was OPTIONAL_ONCE - "item" is now REQUIRED_MULTIPLE (the schema
# itself now enforces the field that was actually flaking, on Google's
# side). _REQUIRED_PRODUCT_PROPERTY_TYPES stays deliberately broader than
# schema-required alone: description/quantity/unit_price/amount are still
# schema-optional but make up the substance of an invoice line (a line
# always has all four on a real invoice) and are what we directly observed
# a foundation-model call drop on an otherwise-clean rerun, whereas fields
# like unit_of_measure/line_number/supplier_item_number/purchase_order_number
# are routinely absent even on correctly-extracted invoices and would retry
# every single document if included here.
CONFIDENCE_THRESHOLD = 0.80
_REQUIRED_TOP_LEVEL_TYPES = {"invoice_date", "total_amount_due", "invoice_number"}
_REQUIRED_PRODUCT_PROPERTY_TYPES = {"item", "description", "quantity", "unit_price", "amount"}


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


def find_extraction_issues(raw_response: dict) -> list[str]:
    """Checks a raw process_document() response for required fields
    (_REQUIRED_TOP_LEVEL_TYPES / _REQUIRED_PRODUCT_PROPERTY_TYPES) that are
    either absent or below CONFIDENCE_THRESHOLD. Returns a list of
    human-readable reasons (empty if the response looks trustworthy) - the
    caller (graph.py's extract_fields) logs these before its one retry."""
    entities = raw_response.get("document", {}).get("entities", [])

    top_level = {}
    product_entities = []
    for entity in entities:
        if entity.get("type") == "products":
            product_entities.append(entity)
        else:
            top_level[entity.get("type")] = entity

    issues = []
    for field_type in _REQUIRED_TOP_LEVEL_TYPES:
        entity = top_level.get(field_type)
        if entity is None:
            issues.append(f"required field '{field_type}' missing")
        elif entity.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            issues.append(f"field '{field_type}' confidence {entity.get('confidence', 0):.2f} < {CONFIDENCE_THRESHOLD}")

    for line_num, entity in enumerate(product_entities, start=1):
        properties = {prop.get("type"): prop for prop in entity.get("properties", [])}
        for field_type in _REQUIRED_PRODUCT_PROPERTY_TYPES:
            prop = properties.get(field_type)
            if prop is None:
                issues.append(f"product line {line_num}: '{field_type}' missing")
            elif prop.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                issues.append(
                    f"product line {line_num}: '{field_type}' confidence {prop.get('confidence', 0):.2f} < {CONFIDENCE_THRESHOLD}"
                )

    return issues


def parse_entities(raw_response: dict) -> dict:
    """Converts this processor's structured document.entities into the same
    canonical field shape gemini_extract.parse_response()'s {"output": ...}
    used to produce - see the module docstring and the field-mapping notes
    above _PRODUCT_ENTITY_TO_FIELD for the specific renames/derivations.

    Also attaches a "confidence" dict (field name -> DocAI's 0-1 confidence)
    at the top level and on every product line, alongside the field values
    themselves - rides through state["extracted"] / pdf_fields unchanged
    since nothing downstream reads more than the specific field keys it
    already expects."""
    entities = raw_response.get("document", {}).get("entities", [])

    output: dict = dict(_MISSING_TOP_LEVEL_DEFAULTS)
    # Field name -> DocAI confidence (0-1) for whatever field it actually
    # extracted. A field left at its _MISSING_*_DEFAULTS value (DocAI never
    # returned that entity at all) has no key here - keep that distinct from
    # "extracted at 0.0 confidence" for the dashboard to render differently.
    output["confidence"] = {}
    products = []

    for entity in entities:
        etype = entity.get("type")
        if etype == "products":
            product = dict(_MISSING_PRODUCT_DEFAULTS)
            product["confidence"] = {}
            for prop in entity.get("properties", []):
                field = _PRODUCT_ENTITY_TO_FIELD.get(prop.get("type"))
                if not field:
                    continue
                text = _clean_text(prop.get("mentionText"))
                product[field] = _clean_money(text) if field in _PRODUCT_MONEY_FIELDS else text
                product["confidence"][field] = prop.get("confidence")
            product.setdefault("description", "")
            products.append(product)
            continue

        text = _clean_text(entity.get("mentionText"))
        if etype in _TOP_LEVEL_MONEY_FIELDS:
            output[etype] = _clean_money(text)
            output["confidence"][etype] = entity.get("confidence")
            if etype == "total_amount_due":
                output["currency"] = _leading_symbol(text)
        elif etype in _TOP_LEVEL_TEXT_FIELDS:
            output[etype] = text
            output["confidence"][etype] = entity.get("confidence")

    output["products"] = products
    if products:
        output["purchase_order"] = products[0].get("purchase_order_number") or ""

    logger.info(
        "document_ai.parse_entities: parsed OK | fields=%s products=%d",
        list(output.keys()), len(products),
    )
    return {"output": output}
