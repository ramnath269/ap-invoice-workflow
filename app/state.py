"""Shared graph state, threaded through every node.

Field-by-field, this is the union of everything that flowed between n8n
nodes as `$json` on the main path of the original workflow.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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

    # Execute Command (mv to processed_files)
    processed_file_path: str

    # resolve_item_numbers: lines JDE's voucher-match couldn't resolve that
    # got a fuzzy-matched (quantity + unit price) candidate item number from
    # a PO-lines lookup, for dashboard review - built once as a complete
    # list per run, so no accumulator reducer needed here (contrast
    # dependency_timings below, which multiple nodes append to over time).
    item_suggestions: list[dict]

    # HTTP Request5
    create_po_response: dict

    # Terminal status, for logging / metrics
    status: str

    # Per-external-call timing/status, one entry per dependency call
    # (document_ai, gemini, jde, invoice_server). Annotated with operator.add
    # so each node's single-entry list gets concatenated onto the running
    # list instead of overwriting it - LangGraph's standard pattern for
    # append-only state fields.
    dependency_timings: Annotated[list[dict], operator.add]
