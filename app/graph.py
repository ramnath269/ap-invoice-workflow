"""LangGraph port of the "FTP based WF" n8n workflow.

Node names mirror the original n8n nodes where practical:

  read_and_encode_file    <- Code in JavaScript (base64 encode)
  extract_fields           <- HTTP Request3 (Document AI custom extractor -
                                OCR + schema field extraction in one call;
                                replaces the old OCR -> Guardrails1 ->
                                Gemini Vertex AI1 -> Parse Gemini JSON1 chain)
  build_voucher_payload    <- Code in JavaScript2
  call_voucher_match       <- Voucher Match + Edit Fields2
  move_file_to_processed   <- Execute Command
  attach_account_numbers   <- Code in JavaScript1
  resolve_item_numbers     <- (new) item_crossref cache + jde_mcp_client PO-lines fuzzy match
  create_po                <- HTTP Request5
  report_metrics_success   <- HTTP Request6

One deliberate simplification versus the n8n graph: the "move file" and
"look up account numbers" branches are sequenced rather than run in
parallel and joined by a chooseBranch merge, since in the original workflow
their only relationship was execution ordering (make sure the file move
happens before PO creation), not a data dependency - a plain sequence
achieves the same guarantee more simply.

Everything else, including the If2 gate on PO receipt records, is
preserved. config/schema_fields.json is no longer read at graph-run time -
Document AI's custom extractor has its own fixed schema (configured on the
processor itself, not per-request) - but schema_builder.account_number_for()
is still used by attach_account_numbers.
"""
import json
import logging
import os
import shutil
import time
import uuid

from langgraph.graph import END, START, StateGraph

from . import document_ai, invoice_server, jde_client, jde_mcp_client, schema_builder
from .settings import settings
from .state import InvoiceState

logger = logging.getLogger("ap_invoice_workflow")

# Fields too large or too sensitive to dump whole into a log line. Node
# output logging below truncates the former and redacts the latter -
# base64_content alone is ~300KB for a typical invoice PDF.
_TRUNCATE_KEYS = {"base64_content"}
_REDACT_KEYS = {"password"}
_MAX_LOG_LEN = 800

_RULE = "-" * 70


def _for_log(value, key=None):
    """Redacts secrets, truncates huge blobs, and - critically - flattens
    embedded newlines out of every string. Real PDF-extracted text (ocr_text,
    sanitized_text) contains actual \\n characters, and systemd's journal
    capture treats stdout as line-oriented: a log line with a literal
    newline in it gets split into multiple journal entries that lose their
    logger prefix on every line after the first."""
    if isinstance(value, dict):
        return {k: _for_log(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_for_log(v) for v in value]
    if key in _REDACT_KEYS:
        return "<redacted>"
    if isinstance(value, str):
        flat = value.replace("\n", " ").replace("\r", " ")
        if key in _TRUNCATE_KEYS and len(flat) > _MAX_LOG_LEN:
            return f"<{len(flat)} chars> {flat[:_MAX_LOG_LEN]}..."
        return flat
    return value


def _format_output_lines(result: dict) -> list[str]:
    """Renders a node's output as one 'key: value' line per field instead of
    a single giant JSON blob, so it can actually be scanned by eye. Each
    string returned here is logged as its own logger.info() call - a single
    call with embedded newlines gets shredded by systemd's line-oriented
    journal capture, losing the log prefix on every line but the first."""
    if not result:
        return ["    (no new data)"]
    lines = []
    for key, raw_value in _for_log(result).items():
        value_str = json.dumps(raw_value, default=str) if isinstance(raw_value, (dict, list)) else str(raw_value)
        lines.append(f"    {key}: {value_str}")
    return lines


def _time_dependency(dependency: str, op: str, started: float, status: str, error: str | None = None) -> dict:
    """Times and logs one external call (Document AI / Gemini / JDE /
    invoice-server) as a consistent, grep-able METRIC line, and returns the
    same data as a dict for the calling node to fold into
    state["dependency_timings"]."""
    duration_ms = int((time.monotonic() - started) * 1000)
    if status == "ok":
        logger.info("METRIC dependency=%s op=%s status=ok duration_ms=%d", dependency, op, duration_ms)
    else:
        logger.warning(
            "METRIC dependency=%s op=%s status=error duration_ms=%d error=%s",
            dependency, op, duration_ms, error,
        )
    timing = {"dependency": dependency, "op": op, "status": status, "duration_ms": duration_ms}
    if error:
        timing["error"] = error
    return timing


def _report_exception_metrics(state: InvoiceState, node_name: str, exc: Exception) -> None:
    """Reports the 'exception' outcome for #1 (outcome classification) when
    any node raises. Only prior nodes' successful dependency_timings are
    available here - the failing call's own timing was already logged by
    _time_dependency at its point of failure, but LangGraph only accumulates
    state from nodes that actually return, so it can't also land in this
    payload."""
    execution_start_ms = state.get("execution_start_ms")
    if not execution_start_ms:
        return
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status="exception",
        execution_start_ms=execution_start_ms,
        failed_node=node_name,
        error=f"{type(exc).__name__}: {exc}",
        dependency_timings=state.get("dependency_timings"),
    )


def _log_node(name, fn):
    """Wraps a node so its start/end and the data it produced are always
    logged, regardless of whether the graph is run via `langgraph dev` (which
    auto-tags stdlib log records with the active node) or via the folder
    watcher's plain graph.invoke() (which doesn't)."""

    def wrapper(state):
        logger.info(">> START %s", name)
        started = time.monotonic()
        try:
            result = fn(state)
        except Exception as exc:
            elapsed = time.monotonic() - started
            logger.exception("!! FAILED %s (%.2fs)", name, elapsed)
            logger.info(_RULE)
            _report_exception_metrics(state, name, exc)
            raise
        elapsed = time.monotonic() - started
        logger.info("OK DONE  %-28s (%.2fs)", name, elapsed)
        for line in _format_output_lines(result):
            logger.info(line)
        logger.info(_RULE)
        return result

    return wrapper


def read_and_encode_file(state: InvoiceState) -> dict:
    base64_content, mime_type = document_ai.encode_file(state["file_path"])
    return {"base64_content": base64_content, "mime_type": mime_type}


def extract_fields(state: InvoiceState) -> dict:
    """Document AI's custom extractor does OCR and schema field extraction
    in one call - see document_ai.parse_entities() for how its structured
    document.entities are converted into the same canonical field shape the
    old OCR -> guardrails sanitize -> Gemini extract -> parse JSON chain
    used to produce, so every downstream node is unaffected.

    parse_error stays in the state contract (route_after_parse still checks
    it) but this path can't really produce a soft parse failure the way
    free-text LLM JSON parsing could - a bad call raises and is handled by
    _log_node's exception path instead, so parse_error is always False here."""
    started = time.monotonic()
    try:
        raw = document_ai.process_document(state["base64_content"], state["mime_type"])
    except Exception as exc:
        _time_dependency("document_ai", "process_document", started, "error", f"{type(exc).__name__}: {exc}")
        raise
    timing = _time_dependency("document_ai", "process_document", started, "ok")
    extracted = document_ai.parse_entities(raw)["output"]
    return {"extracted": extracted, "parse_error": False, "dependency_timings": [timing]}


def route_after_parse(state: InvoiceState) -> str:
    route = "handle_parse_error" if state.get("parse_error") else "build_voucher_payload"
    logger.info(">> ROUTE   route_after_parse: parse_error=%s => %s", state.get("parse_error"), route)
    return route


def build_voucher_payload(state: InvoiceState) -> dict:
    return {"voucher_match_payload": jde_client.build_payload(state["extracted"])}


def call_voucher_match(state: InvoiceState) -> dict:
    started = time.monotonic()
    try:
        response = jde_client.voucher_match(state["voucher_match_payload"])
    except Exception as exc:
        _time_dependency("jde", "voucher_match", started, "error", f"{type(exc).__name__}: {exc}")
        raise
    timing = _time_dependency("jde", "voucher_match", started, "ok")
    return {
        "voucher_match_response": response,
        "erp_fields": response,
        "po_receipt_valid": jde_client.has_receipt_records(response),
        "dependency_timings": [timing],
    }


def route_after_voucher_match(state: InvoiceState) -> str:
    route = "move_file_to_processed" if state.get("po_receipt_valid") else "skip_no_receipt"
    logger.info(
        ">> ROUTE   route_after_voucher_match: po_receipt_valid=%s => %s",
        state.get("po_receipt_valid"),
        route,
    )
    return route


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


def resolve_item_numbers(state: InvoiceState) -> dict:
    """Fixes up lines where JDE's voucher-match couldn't resolve the
    supplier's item number (a rowset entry with a non-blank Message, e.g.
    "Unable to fetch Order/Item information"). Tries the invoice-server
    item_crossref cache first - a previously user-confirmed mapping applies
    immediately, no review needed - and on a cache miss falls back to
    fetching the PO's lines via jde_mcp_client and fuzzy-matching by
    quantity + unit price, attaching any single confident match as a
    suggestion for dashboard review rather than applying it outright.

    Deliberately never blocks create_po, even on its own bugs: an unresolved
    or ambiguous line just proceeds with whatever item number it already
    had, plus whatever suggestions were found along the way.
    """
    try:
        return _resolve_item_numbers(state)
    except Exception:
        logger.exception(
            "resolve_item_numbers: unexpected failure - proceeding without resolving or suggesting anything"
        )
        return {"item_suggestions": []}


def _resolve_item_numbers(state: InvoiceState) -> dict:
    rowset = state["voucher_match_response"].get(jde_client.RECEIPT_INQUIRY_KEY, {}).get("rowset", [])
    extracted = dict(state["extracted"])
    products = [dict(p) for p in extracted.get("products", [])]
    supplier_number = state["erp_fields"].get("VendorNumber")
    order_number = extracted.get("purchase_order")
    order_type = state["erp_fields"].get("OrderType") or "OP"

    suggestions = []
    timings = []
    po_lines = None  # fetched at most once per run, shared across every unresolved line on this PO

    for row in rowset:
        message = (row.get("Message") or "").strip()
        if not message:
            continue  # this line resolved fine, nothing to do

        seq = row.get("SequenceNumber")
        idx = seq - 1 if seq else None
        if idx is None or not (0 <= idx < len(products)):
            logger.warning(
                "resolve_item_numbers: unresolved row has no matching product (seq=%s): %s", seq, message
            )
            continue

        product = products[idx]
        # Same priority order jde_client.build_payload uses to pick ItemNo -
        # this needs to be the identifier that was actually sent to (and
        # rejected by) JDE, not a fallback description field, since it's
        # both the crossref cache lookup key and the identifier suffix-
        # matched against PO lines below.
        pdf_item_number = product.get("item_number") or product.get("supplier_item_number") or product.get("item")
        if not pdf_item_number:
            continue

        logger.warning(
            "resolve_item_numbers: line %s unresolved (%s), supplier_item_number=%s", seq, message, pdf_item_number
        )

        started = time.monotonic()
        assigned = invoice_server.lookup_item_crossref(supplier_number, pdf_item_number)
        timings.append(_time_dependency("invoice_server", "lookup_item_crossref", started, "ok"))
        if assigned:
            logger.info(
                "resolve_item_numbers: cache hit, supplier=%s item=%s -> %s", supplier_number, pdf_item_number, assigned
            )
            product["item_number"] = assigned
            continue

        try:
            quantity = float(product.get("quantity") or 0)
            unit_price = float(product.get("unit_price") or 0)
        except (TypeError, ValueError):
            quantity = unit_price = 0.0

        if not quantity and not unit_price:
            logger.warning(
                "resolve_item_numbers: no quantity or unit_price to match on for line %s, skipping fuzzy match", seq
            )
            continue

        if po_lines is None:
            started = time.monotonic()
            try:
                po_lines = jde_mcp_client.get_po_lines(order_number, order_type)
            except Exception as exc:
                timings.append(
                    _time_dependency("jde_mcp", "get_po_lines", started, "error", f"{type(exc).__name__}: {exc}")
                )
                logger.exception(
                    "resolve_item_numbers: get_po_lines failed for PO %s - skipping fuzzy match for remaining lines",
                    order_number,
                )
                po_lines = []
            else:
                timings.append(_time_dependency("jde_mcp", "get_po_lines", started, "ok"))

        # Identifier match first (the PDF's own item number as a suffix of
        # JDE's SecondItemNumber, corroborated by quantity or cost) - only
        # falls back to a bare quantity+unit-price coincidence if that finds
        # nothing, since the identifier is the stronger, OCR-noise-resistant
        # signal when it's available.
        match = jde_mcp_client.find_po_line_by_item_number(po_lines, pdf_item_number, quantity, unit_price)
        match_basis = "item_number"
        if not match:
            match = jde_mcp_client.find_matching_po_line(po_lines, quantity, unit_price)
            match_basis = "quantity_and_unit_price"

        if match:
            suggested = str(match.get("SecondItemNumber"))
            logger.info("resolve_item_numbers: %s match for line %s -> %s", match_basis, seq, suggested)
            suggestions.append(
                {
                    "line_index": idx,
                    "pdf_item_number": pdf_item_number,
                    "suggested_jde_item_number": suggested,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "match_basis": match_basis,
                }
            )
        else:
            logger.warning(
                "resolve_item_numbers: no confident match for line %s (supplier_item_number=%s)", seq, pdf_item_number
            )

    extracted["products"] = products
    return {"extracted": extracted, "item_suggestions": suggestions, "dependency_timings": timings}


def create_po(state: InvoiceState) -> dict:
    started = time.monotonic()
    try:
        response = invoice_server.create_po(
            pdf_fields=state["extracted"],
            erp_fields=state["erp_fields"],
            file_path=state["processed_file_path"],
            item_suggestions=state.get("item_suggestions"),
        )
    except Exception as exc:
        _time_dependency("invoice_server", "create_po", started, "error", f"{type(exc).__name__}: {exc}")
        raise
    timing = _time_dependency("invoice_server", "create_po", started, "ok")
    return {"create_po_response": response, "dependency_timings": [timing]}


def report_metrics_success(state: InvoiceState) -> dict:
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status="success",
        execution_start_ms=state["execution_start_ms"],
        dependency_timings=state.get("dependency_timings"),
    )
    return {"status": "success"}


def skip_no_receipt(state: InvoiceState) -> dict:
    logger.warning("No PO receipt records found for %s; skipping PO creation.", state["file_name"])
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status="skipped_no_receipt",
        execution_start_ms=state["execution_start_ms"],
        dependency_timings=state.get("dependency_timings"),
    )
    return {"status": "skipped_no_receipt"}


def handle_parse_error(state: InvoiceState) -> dict:
    logger.error(
        "Field extraction failed to parse for %s: %s", state["file_name"], state.get("parse_error_message")
    )
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status="parse_error",
        execution_start_ms=state["execution_start_ms"],
        error=state.get("parse_error_message"),
        dependency_timings=state.get("dependency_timings"),
    )
    return {"status": "parse_error"}


def build_graph():
    graph = StateGraph(InvoiceState)

    for name, fn in [
        ("read_and_encode_file", read_and_encode_file),
        ("extract_fields", extract_fields),
        ("build_voucher_payload", build_voucher_payload),
        ("call_voucher_match", call_voucher_match),
        ("move_file_to_processed", move_file_to_processed),
        ("attach_account_numbers", attach_account_numbers),
        ("resolve_item_numbers", resolve_item_numbers),
        ("create_po", create_po),
        ("report_metrics_success", report_metrics_success),
        ("skip_no_receipt", skip_no_receipt),
        ("handle_parse_error", handle_parse_error),
    ]:
        graph.add_node(name, _log_node(name, fn))

    graph.add_edge(START, "read_and_encode_file")
    graph.add_edge("read_and_encode_file", "extract_fields")
    graph.add_conditional_edges(
        "extract_fields",
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
    graph.add_edge("attach_account_numbers", "resolve_item_numbers")
    graph.add_edge("resolve_item_numbers", "create_po")
    graph.add_edge("create_po", "report_metrics_success")

    graph.add_edge("report_metrics_success", END)
    graph.add_edge("skip_no_receipt", END)
    graph.add_edge("handle_parse_error", END)

    return graph.compile()
