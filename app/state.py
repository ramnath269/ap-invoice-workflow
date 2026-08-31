"""Shared graph state, threaded through every node.

Field-by-field, this is the union of everything that flowed between n8n
nodes as `$json` on the main path of the original workflow.
"""
from __future__ import annotations

from typing import TypedDict


class InvoiceState(TypedDict, total=False):
    # Local File Trigger1 / Set Start Time1
    file_path: str
    file_name: str
    execution_start_ms: float

    # Schema Fields1 + Array fields1 + Code in JavaScript3
    extraction_schema: dict

    # Code in JavaScript (base64 encode)
    base64_content: str
    mime_type: str

    # HTTP Request3 + Edit Fields1
    ocr_text: str

    # Guardrails1
    sanitized_text: str

    # Gemini Vertex AI1
    gemini_raw_response: dict

    # Parse Gemini JSON1
    extracted: dict
    parse_error: bool
    parse_error_message: str

    # Code in JavaScript2 + Voucher Match + Edit Fields2
    voucher_match_payload: dict
    voucher_match_response: dict
    erp_fields: dict
    po_receipt_valid: bool

    # HTTP Request5
    create_po_response: dict

    # Terminal status, for logging / metrics
    status: str
