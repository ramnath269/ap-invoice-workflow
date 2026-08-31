"""LangGraph port of the "FTP based WF" n8n workflow.

Node names mirror the original n8n nodes where practical:

  load_schema             <- Schema Fields1 + Array fields1 + Code in JavaScript3
  read_and_encode_file    <- Code in JavaScript (base64 encode)
  run_ocr                 <- HTTP Request3 (Document AI) + Edit Fields1
  sanitize_text           <- Guardrails1
  call_gemini              <- Gemini Vertex AI1
  parse_gemini_response    <- Parse Gemini JSON1
  build_voucher_payload    <- Code in JavaScript2
  call_voucher_match       <- Voucher Match + Edit Fields2
  move_file_to_processed   <- Execute Command
  attach_account_numbers   <- Code in JavaScript1
  create_po                <- HTTP Request5
  report_metrics_success   <- HTTP Request6

Two deliberate simplifications versus the n8n graph:
1. Schema/array field lookups are loaded once from a static YAML config
   (config/schema_fields.yaml) rather than live Data Table nodes.
2. The "move file" and "look up account numbers" branches are sequenced
   rather than run in parallel and joined by a chooseBranch merge, since in
   the original workflow their only relationship was execution ordering
   (make sure the file move happens before PO creation), not a data
   dependency - a plain sequence achieves the same guarantee more simply.

Everything else - including the two real parallel branches (schema
building vs. OCR+sanitize, both joined at call_gemini) and the If2 gate on
PO receipt records - is preserved.
"""
import logging
import os
import shutil
import uuid

from langgraph.graph import END, START, StateGraph

from . import document_ai, gemini_extract, guardrails, invoice_server, jde_client, schema_builder
from .settings import settings
from .state import InvoiceState

logger = logging.getLogger("ap_invoice_workflow")


def load_schema(state: InvoiceState) -> dict:
    return {"extraction_schema": schema_builder.build_extraction_schema()}


def read_and_encode_file(state: InvoiceState) -> dict:
    base64_content, mime_type = document_ai.encode_file(state["file_path"])
    return {"base64_content": base64_content, "mime_type": mime_type}


def run_ocr(state: InvoiceState) -> dict:
    return {"ocr_text": document_ai.run_ocr(state["base64_content"], state["mime_type"])}


def sanitize_text(state: InvoiceState) -> dict:
    return {"sanitized_text": guardrails.sanitize(state["ocr_text"])}


def call_gemini(state: InvoiceState) -> dict:
    raw = gemini_extract.extract(state["sanitized_text"], state["extraction_schema"])
    return {"gemini_raw_response": raw}


def parse_gemini_response(state: InvoiceState) -> dict:
    result = gemini_extract.parse_response(state["gemini_raw_response"])
    if result.get("parse_error"):
        return {"parse_error": True, "parse_error_message": result["error"]}
    return {"extracted": result["output"], "parse_error": False}


def route_after_parse(state: InvoiceState) -> str:
    return "handle_parse_error" if state.get("parse_error") else "build_voucher_payload"


def build_voucher_payload(state: InvoiceState) -> dict:
    return {"voucher_match_payload": jde_client.build_payload(state["extracted"])}


def call_voucher_match(state: InvoiceState) -> dict:
    response = jde_client.voucher_match(state["voucher_match_payload"])
    return {
        "voucher_match_response": response,
        "erp_fields": response,
        "po_receipt_valid": jde_client.has_receipt_records(response),
    }


def route_after_voucher_match(state: InvoiceState) -> str:
    return "move_file_to_processed" if state.get("po_receipt_valid") else "skip_no_receipt"


def move_file_to_processed(state: InvoiceState) -> dict:
    # os.path.join discards WATCH_FOLDER if PROCESSED_SUBFOLDER is itself
    # absolute, which is fine - it just means "processed files go here"
    # regardless of where they were watched from.
    processed_dir = os.path.join(settings.WATCH_FOLDER, settings.PROCESSED_SUBFOLDER)
    os.makedirs(processed_dir, exist_ok=True)
    destination = os.path.join(processed_dir, state["file_name"])
    shutil.move(state["file_path"], destination)
    return {"processed_file_path": destination}


def attach_account_numbers(state: InvoiceState) -> dict:
    extracted = dict(state["extracted"])
    for key in ("total_freight", "handling_fee"):
        amount = extracted.get(key)
        if amount:
            extracted[key] = {"amount": amount, "AccountNo": schema_builder.account_number_for(key)}
    return {"extracted": extracted}


def create_po(state: InvoiceState) -> dict:
    response = invoice_server.create_po(
        pdf_fields=state["extracted"],
        erp_fields=state["erp_fields"],
        file_path=state["processed_file_path"],
    )
    return {"create_po_response": response}


def report_metrics_success(state: InvoiceState) -> dict:
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status="success",
        execution_start_ms=state["execution_start_ms"],
    )
    return {"status": "success"}


def skip_no_receipt(state: InvoiceState) -> dict:
    logger.warning("No PO receipt records found for %s; skipping PO creation.", state["file_name"])
    return {"status": "skipped_no_receipt"}


def handle_parse_error(state: InvoiceState) -> dict:
    logger.error(
        "Gemini output failed to parse for %s: %s", state["file_name"], state.get("parse_error_message")
    )
    return {"status": "parse_error"}


def build_graph():
    graph = StateGraph(InvoiceState)

    for name, fn in [
        ("load_schema", load_schema),
        ("read_and_encode_file", read_and_encode_file),
        ("run_ocr", run_ocr),
        ("sanitize_text", sanitize_text),
        ("call_gemini", call_gemini),
        ("parse_gemini_response", parse_gemini_response),
        ("build_voucher_payload", build_voucher_payload),
        ("call_voucher_match", call_voucher_match),
        ("move_file_to_processed", move_file_to_processed),
        ("attach_account_numbers", attach_account_numbers),
        ("create_po", create_po),
        ("report_metrics_success", report_metrics_success),
        ("skip_no_receipt", skip_no_receipt),
        ("handle_parse_error", handle_parse_error),
    ]:
        graph.add_node(name, fn)

    # Fan-out: schema build and OCR+sanitize run in parallel, join at call_gemini.
    # NOTE: a real AND-join needs the source nodes passed as a single list to
    # one add_edge call - two separate add_edge(...) calls into the same
    # target are an OR-trigger (the node runs after the first predecessor to
    # finish, not after all of them), which fired call_gemini prematurely.
    graph.add_edge(START, "load_schema")
    graph.add_edge(START, "read_and_encode_file")
    graph.add_edge("read_and_encode_file", "run_ocr")
    graph.add_edge("run_ocr", "sanitize_text")
    graph.add_edge(["load_schema", "sanitize_text"], "call_gemini")

    graph.add_edge("call_gemini", "parse_gemini_response")
    graph.add_conditional_edges(
        "parse_gemini_response",
        route_after_parse,
        {
            "build_voucher_payload": "build_voucher_payload",
            "handle_parse_error": "handle_parse_error",
        },
    )

    graph.add_edge("build_voucher_payload", "call_voucher_match")
    graph.add_conditional_edges(
        "call_voucher_match",
        route_after_voucher_match,
        {
            "move_file_to_processed": "move_file_to_processed",
            "skip_no_receipt": "skip_no_receipt",
        },
    )

    graph.add_edge("move_file_to_processed", "attach_account_numbers")
    graph.add_edge("attach_account_numbers", "create_po")
    graph.add_edge("create_po", "report_metrics_success")

    graph.add_edge("report_metrics_success", END)
    graph.add_edge("skip_no_receipt", END)
    graph.add_edge("handle_parse_error", END)

    return graph.compile()
