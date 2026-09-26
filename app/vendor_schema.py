"""Vendor-specific extraction-field overrides, stored in invoice-server's
"vendor-extract-schema" CouchDB database (see server/vendorSchemaDb.js /
vendorSchemaRoutes.js) and fetched over HTTP the same way invoice_server.py
talks to that service's other endpoints.

Each vendor doc holds ONLY the fields with a genuine, structurally-different
extraction rule for that vendor (~9 fields, out of the processor's 26) - not
a full copy of the schema. See document_ai.build_schema_override for how
these are merged with the processor's own field descriptions to build a
second-call processOptions.schemaOverride that targets just the fields a
first call got wrong.
"""
from __future__ import annotations

import logging

import requests

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

# Keyword -> canonical vendor_name (must match the "vendor_name" stored on
# the corresponding vendor-extract-schema doc). Mirrors the lesson learned
# building the scratch vendor-classifier processor: the foundation model's
# "vendor" field extracts literal text off the page (e.g. "Ford Motor
# Company", "GM Customer Care and Aftersales"), never an invented category -
# so mapping that literal text to a vendor key is done here in app code via
# keyword matching, not by asking Document AI for a category label.
_VENDOR_KEYWORDS: list[tuple[str, str]] = [
    ("ford", "Ford"),
    ("acdelco", "GM/ACDelco"),
    ("general motors", "GM/ACDelco"),
    ("advanced distribution", "ADS"),
    ("federal-mogul", "Federal-Mogul"),
    ("federal mogul", "Federal-Mogul"),
    ("federated", "Federated Auto Parts"),
]

# Canonical vendor_names whose line items always satisfy quantity x
# unit_price == amount and sum to the invoice total, so that arithmetic can
# be used to detect misread values (see document_ai.find_line_math_issues)
# and repair them (document_ai.repair_line_math). Opt-in per vendor, NOT
# universal: GM/ACDelco's "amount" is the net after a per-line discount, so
# quantity x unit_price legitimately differs from it on almost every line
# there, and other vendors can carry freight/tax/core lines outside the
# line-item sum.
LINE_MATH_VENDORS = {"Federated Auto Parts"}


def classify_vendor(vendor_text: str | None) -> str | None:
    """Maps extracted "vendor" text to one of the canonical vendor_name
    values stored in vendor-extract-schema, or None if it matches none of
    the known vendors (e.g. blank extraction, or a genuinely new vendor with
    no seeded overrides yet - callers should fall back to the processor's
    own generic field descriptions in that case, not fail)."""
    if not vendor_text:
        return None
    lowered = vendor_text.lower()
    for keyword, canonical_name in _VENDOR_KEYWORDS:
        if keyword in lowered:
            return canonical_name
    return None


def fetch_vendor_schemas() -> list[dict]:
    """GET all vendor-extract-schema docs. Not cached - these are edited live
    (e.g. via a future admin UI on top of the existing CRUD routes) and this
    is only called on the rare retry-with-override path, not per document."""
    resp = requests.get(f"{settings.INVOICE_SERVER_BASE_URL}/vendor-schemas", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_vendor_field_overrides(vendor_name: str) -> dict[str, dict]:
    """Returns {field_name: {valueType, occurrenceType, method, description}}
    for the given canonical vendor_name, or {} if no doc matches (unseeded
    vendor) or the lookup fails - a lookup failure here should fall back to
    the processor's generic field descriptions, not abort the retry."""
    try:
        docs = fetch_vendor_schemas()
    except Exception:
        logger.exception(
            "vendor_schema.get_vendor_field_overrides: lookup failed for vendor_name=%s - "
            "falling back to generic field descriptions", vendor_name,
        )
        return {}

    for doc in docs:
        if str(doc.get("vendor_name", "")).strip().lower() == vendor_name.strip().lower():
            return {field["name"]: field for field in doc.get("schema", [])}

    logger.info(
        "vendor_schema.get_vendor_field_overrides: no vendor-extract-schema doc for vendor_name=%s - "
        "using generic field descriptions only", vendor_name,
    )
    return {}
