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
extract_fields' first call always uses the custom extractor's own stored
schema; only a targeted retry (see extract_fields, document_ai.
build_schema_override) sends a per-request processOptions.schemaOverride
for the specific fields that came back missing/low-confidence - but
schema_builder.account_number_for() is still used by attach_account_numbers.
"""
import base64
import json
import logging
import os
import shutil
import time
import uuid

from langgraph.graph import END, START, StateGraph

from . import (
    document_ai,
    invoice_server,
    jde_client,
    jde_mcp_client,
    pdf_text_layer,
    schema_builder,
    vendor_schema,
)
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
    _log_node's exception path instead, so parse_error is always False here.

    The processor's foundation-model backend isn't fully deterministic call
    to call (see document_ai.find_extraction_issues' docstring for the
    evidence), so a response missing a required field or holding low
    confidence on one gets exactly one targeted retry: instead of blindly
    re-running the whole 26-field schema, it identifies the vendor from the
    first call's own "vendor" extraction, looks up that vendor's
    field-specific extraction rules (vendor_schema.get_vendor_field_overrides,
    backed by invoice-server's vendor-extract-schema CouchDB store), and asks
    Document AI for ONLY the missing/low-confidence fields via
    processOptions.schemaOverride - cheaper than a full second pass, and more
    likely to succeed than retrying with the same generic, multi-vendor field
    description that may have contributed to the miss. A vendor that can't be
    classified (blank/unrecognized "vendor" text) or has no seeded overrides
    still gets a same-fields retry using the processor's own generic
    descriptions, so plain model flakiness still benefits. The override
    retry never raises on failure - a lookup or override-call problem falls
    back to the first call's result rather than losing an otherwise-usable
    extraction.

    Vendors in vendor_schema.LINE_MATH_VENDORS additionally get an
    arithmetic check (quantity x unit_price == amount per line, lines sum to
    the total): a failure triggers a full product re-extraction that replaces
    the first call's product list if it scores better, and
    document_ai.repair_line_math then fixes any single value per line the
    arithmetic uniquely supports."""
    started = time.monotonic()
    try:
        raw = document_ai.process_document(state["base64_content"], state["mime_type"])
    except Exception as exc:
        _time_dependency("document_ai", "process_document", started, "error", f"{type(exc).__name__}: {exc}")
        raise
    timings = [_time_dependency("document_ai", "process_document", started, "ok")]

    extracted = document_ai.parse_entities(raw)["output"]

    issues = document_ai.find_extraction_issues(raw)
    missing = document_ai.find_missing_fields(raw)
    vendor_name = vendor_schema.classify_vendor(extracted.get("vendor"))

    # For a vendor whose lines obey quantity x unit_price == amount, a
    # present-but-misread value (dropped digit/decimal point, Q read as O)
    # never shows up as missing or low-confidence. Arithmetic catches it, and
    # the retry then re-extracts every product field and replaces the first
    # call's product list outright - an item-anchored fill-only merge can't
    # correct a value that's present, nor realign rows by an item number
    # that is itself one of the misreads being corrected.
    line_math = vendor_name in vendor_schema.LINE_MATH_VENDORS
    replace_products = False
    if line_math:
        line_math_issues = document_ai.find_line_math_issues(extracted)
        if line_math_issues["bad_rows"] or not line_math_issues["sum_ok"]:
            logger.warning(
                "extract_fields: %d line(s) fail quantity x unit_price == amount, line sum %s vs "
                "expected %s - retrying with a full product re-extraction",
                len(line_math_issues["bad_rows"]), line_math_issues["line_sum"], line_math_issues["expected"],
            )
            missing["product"] |= document_ai.FULL_PRODUCT_RETRY_FIELDS
            replace_products = True

    if missing["top_level"] or missing["product"]:
        logger.warning(
            "extract_fields: %d issue(s) found, retrying with a targeted schemaOverride for %s: %s",
            len(issues), sorted(missing["top_level"] | missing["product"]), "; ".join(issues),
        )
        vendor_fields = vendor_schema.get_vendor_field_overrides(vendor_name) if vendor_name else {}
        base_schema = document_ai.fetch_live_schema()
        override = document_ai.build_schema_override(missing, vendor_fields, base_schema)

        override_started = time.monotonic()
        try:
            override_raw = document_ai.process_document(
                state["base64_content"], state["mime_type"], schema_override=override
            )
            timings.append(_time_dependency("document_ai", "process_document_override", override_started, "ok"))
            extracted_override = document_ai.parse_entities(override_raw)["output"]
            first_score = document_ai.line_math_score(extracted) if replace_products else None
            override_score = document_ai.line_math_score(extracted_override) if replace_products else None
            if replace_products and override_score > first_score:
                logger.info(
                    "extract_fields: full product re-extraction scored better (consistent lines, sum ok) "
                    "%s vs first call %s - replacing product list", override_score, first_score,
                )
                products = extracted_override["products"]
                filled = document_ai.backfill_products(products, extracted.get("products", []))
                logger.info("extract_fields: filled %d empty product field(s) from the first call's lines", filled)
                extracted = document_ai.merge_extracted(
                    extracted, extracted_override, {"top_level": missing["top_level"], "product": set()}
                )
                extracted["products"] = products
            else:
                if replace_products:
                    logger.info(
                        "extract_fields: full product re-extraction scored %s, not better than first call %s "
                        "- only filling empty fields", override_score, first_score,
                    )
                extracted = document_ai.merge_extracted(extracted, extracted_override, missing)
        except Exception as exc:
            timings.append(_time_dependency(
                "document_ai", "process_document_override", override_started, "error",
                f"{type(exc).__name__}: {exc}",
            ))
            logger.warning(
                "extract_fields: schemaOverride retry failed (%s), proceeding with the first call's result",
                exc,
            )

    moved_charges = document_ai.move_charge_lines(extracted)
    if moved_charges:
        logger.info("extract_fields: moved %d freight/handling line(s) out of products: %s", len(moved_charges), moved_charges)

    # Document AI reads the page image even when the PDF embeds exact text,
    # so item numbers it misreads (Q as O on dot-matrix prints) can be
    # corrected from the text layer. A no-op for scans.
    if state["mime_type"] == "application/pdf":
        text_lines = pdf_text_layer.read_text_lines(base64.b64decode(state["base64_content"]))
        if text_lines:
            corrections = pdf_text_layer.correct_items(extracted.get("products", []), text_lines)
            if corrections:
                logger.info(
                    "extract_fields: corrected %d item number(s) from the PDF text layer: %s",
                    len(corrections), "; ".join(corrections),
                )
            fixed = pdf_text_layer.fix_descriptions(extracted.get("products", []), text_lines)
            if fixed:
                logger.info("extract_fields: set %d empty/garbled description(s) from the PDF text layer", fixed)

    if line_math:
        repairs = document_ai.repair_line_math(extracted)
        remaining = document_ai.find_line_math_issues(extracted)
        logger.info(
            "extract_fields: line-math repaired %d value(s)%s; %d line(s) still inconsistent, line sum %s vs expected %s",
            len(repairs), (": " + "; ".join(repairs)) if repairs else "",
            len(remaining["bad_rows"]), remaining["line_sum"], remaining["expected"],
        )

    not_an_invoice = not document_ai.looks_like_invoice(raw)
    return {
        "extracted": extracted,
        "parse_error": False,
        "not_an_invoice": not_an_invoice,
        "dependency_timings": timings,
    }


def has_valid_order_number(order_number) -> bool:
    """JDE order numbers are purely numeric - anything else (letters, dashes,
    spaces, OCR noise like "35760Z") can't be a real PO and would only come
    back from JDE as a misleading "order not found"."""
    return str(order_number or "").strip().isascii() and str(order_number or "").strip().isdigit()


def route_after_parse(state: InvoiceState) -> str:
    order_number = (state.get("extracted") or {}).get("purchase_order")
    missing_order_number = not order_number
    incorrect_order_number = not missing_order_number and not has_valid_order_number(order_number)
    if state.get("parse_error"):
        route = "handle_parse_error"
    elif state.get("not_an_invoice"):
        route = "handle_not_an_invoice"
    elif missing_order_number:
        route = "handle_missing_order_number"
    elif incorrect_order_number:
        route = "handle_incorrect_order_number"
    else:
        route = "check_already_processed"
    logger.info(
        ">> ROUTE   route_after_parse: parse_error=%s not_an_invoice=%s missing_order_number=%s => %s",
        state.get("parse_error"), state.get("not_an_invoice"), missing_order_number, route,
    )
    return route


# Not a JDE ErrorMessage classification (contrast jde_client.
# ERROR_DUPLICATE_INVOICE/ERROR_ORDER_NOT_FOUND) - this is our own local
# duplicate check, so it lives here rather than in jde_client.py.
ALREADY_PROCESSED = "already_processed"


def check_already_processed(state: InvoiceState) -> dict:
    """Looks up invoice-server for an existing SUCCESSFUL purchase_order
    record on the same (OrderNumber, VendorInvoiceNo) before ever calling
    JDE again. Catches the case JDE's own duplicate-invoice check doesn't:
    the same invoice resubmitted (e.g. the folder watcher/email watcher
    seeing the same file twice, or a person re-uploading it) at a point
    where JDE's voucher-match doesn't happen to flag it as a duplicate
    itself, which previously sailed straight through to create_po and
    produced a second row in the dashboard for an invoice already showing
    there as processed."""
    extracted = state["extracted"]
    order_number = str(extracted.get("purchase_order") or "")
    vendor_invoice_no = extracted.get("invoice_number") or ""
    existing = invoice_server.find_existing_po(order_number, vendor_invoice_no)
    return {"already_processed_doc": existing}


def route_after_already_processed_check(state: InvoiceState) -> str:
    route = "handle_already_processed" if state.get("already_processed_doc") else "build_voucher_payload"
    logger.info(
        ">> ROUTE   route_after_already_processed_check: already_processed=%s => %s",
        bool(state.get("already_processed_doc")),
        route,
    )
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
        "voucher_match_error": jde_client.classify_voucher_match_error(response),
        "dependency_timings": [timing],
    }


def route_after_voucher_match(state: InvoiceState) -> str:
    error = state.get("voucher_match_error")
    if error == jde_client.ERROR_DUPLICATE_INVOICE:
        route = "handle_duplicate_invoice"
    elif error == jde_client.ERROR_ORDER_NOT_FOUND:
        route = "handle_order_not_found"
    elif state.get("po_receipt_valid"):
        route = "move_file_to_processed"
    else:
        route = "skip_no_receipt"
    logger.info(
        ">> ROUTE   route_after_voucher_match: voucher_match_error=%s po_receipt_valid=%s => %s",
        error,
        state.get("po_receipt_valid"),
        route,
    )
    return route


def _move_to_processed(file_path: str, file_name: str) -> str:
    # os.path.join discards WATCH_FOLDER if PROCESSED_SUBFOLDER is itself
    # absolute, which is fine - it just means "processed files go here"
    # regardless of where they were watched from.
    processed_dir = os.path.join(settings.WATCH_FOLDER, settings.PROCESSED_SUBFOLDER)
    os.makedirs(processed_dir, exist_ok=True)
    destination = os.path.join(processed_dir, file_name)
    shutil.move(file_path, destination)
    return destination


def move_file_to_processed(state: InvoiceState) -> dict:
    destination = _move_to_processed(state["file_path"], state["file_name"])
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
    item_crossref cache first - a previously user-confirmed mapping becomes a
    pending suggestion (match_basis "crossref") without touching the extracted
    item number - and on a cache miss falls back to
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
        pdf_item_number = product.get("item") or product.get("supplier_item_number")
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
            # Surfaced as a pending suggestion like any other match (pdf_fields'
            # item number is left as extracted) - a person still confirms it in the
            # dashboard.
            try:
                cache_quantity = float(product.get("quantity") or 0)
                cache_unit_price = float(product.get("unit_price") or 0)
            except (TypeError, ValueError):
                cache_quantity = cache_unit_price = 0.0
            suggestions.append(
                {
                    "line_index": idx,
                    "pdf_item_number": pdf_item_number,
                    "suggested_jde_item_number": str(assigned),
                    "quantity": cache_quantity,
                    "unit_price": cache_unit_price,
                    "match_basis": "crossref",
                }
            )
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


def _record_exception_outcome(
    state: InvoiceState, exception_reason: str, erp_fields: dict, error_message: str
) -> dict:
    """Shared tail for every exception outcome (duplicate_invoice,
    order_not_found, not_an_invoice): moves the PDF to the processed folder
    itself - routing to any of these bypasses the move_file_to_processed node
    entirely, so without this, invoice_server.record_exception's file_path
    would point at a file that was never moved out of the watch folder, and
    the dashboard's /file/:id lookup (which only looks in the processed
    folder) would 404/500 on it - then persists the exception (dedupes
    server-side so a resubmitted invoice doesn't pile up repeat records) and
    reports metrics, so it shows up for review instead of only leaving a
    trace in workflow metrics."""
    processed_file_path = _move_to_processed(state["file_path"], state["file_name"])
    invoice_server.record_exception(
        pdf_fields=state["extracted"],
        erp_fields=erp_fields,
        file_path=processed_file_path,
        exception_reason=exception_reason,
        error_message=error_message,
    )
    invoice_server.report_metrics(
        execution_id=str(uuid.uuid4()),
        status=exception_reason,
        execution_start_ms=state["execution_start_ms"],
        error=error_message,
        dependency_timings=state.get("dependency_timings"),
    )
    return {"status": exception_reason}


def _handle_voucher_match_exception(state: InvoiceState, exception_reason: str) -> dict:
    """Common body for the two recognized JDE voucher-match errors: unlike
    skip_no_receipt (a legitimate "nothing's been received yet" outcome),
    these are exceptions that get persisted for review."""
    response = state.get("voucher_match_response") or {}
    error_message = response.get("ErrorMessage") or ""
    logger.error(
        "%s for %s: OrderNumber=%s VendorInvoiceNo=%s ErrorMessage=%s",
        exception_reason,
        state["file_name"],
        response.get("OrderNumber"),
        response.get("VendorInvoiceNo"),
        error_message,
    )
    return _record_exception_outcome(state, exception_reason, response, error_message)


def handle_duplicate_invoice(state: InvoiceState) -> dict:
    return _handle_voucher_match_exception(state, jde_client.ERROR_DUPLICATE_INVOICE)


def handle_already_processed(state: InvoiceState) -> dict:
    """Routed here by route_after_already_processed_check when
    check_already_processed found an existing SUCCESSFUL purchase_order
    record for the same OrderNumber/VendorInvoiceNo. Recorded as its own
    exception_reason rather than reusing ERROR_DUPLICATE_INVOICE: unlike
    that case, this isn't a JDE-side data problem a corrected invoice number
    can fix, so the dashboard's "reprocess with corrected invoice number"
    action (built for the JDE case) doesn't apply here - it just needs a
    person to notice the resubmission and discard it.

    erp_fields comes from the EXISTING record (not this run, which never
    called JDE) so the dashboard row still shows vendor name/amount instead
    of just the bare OrderNumber/VendorInvoiceNo used to find it."""
    existing = state.get("already_processed_doc") or {}
    extracted = state.get("extracted") or {}
    order_number = extracted.get("purchase_order")
    vendor_invoice_no = extracted.get("invoice_number")
    erp_fields = existing.get("erp_fields") or {
        "OrderNumber": order_number,
        "VendorInvoiceNo": vendor_invoice_no,
    }
    error_message = (
        f"This invoice (PO {order_number}, invoice #{vendor_invoice_no}) was already "
        f"processed - see existing record {existing.get('_id')}."
    )
    logger.error("already_processed for %s: %s", state["file_name"], error_message)
    return _record_exception_outcome(state, ALREADY_PROCESSED, erp_fields, error_message)


def handle_order_not_found(state: InvoiceState) -> dict:
    return _handle_voucher_match_exception(state, jde_client.ERROR_ORDER_NOT_FOUND)


def handle_not_an_invoice(state: InvoiceState) -> dict:
    """Routed here by route_after_parse when extract_fields's Document AI
    call found none of the required top-level fields (document_ai.
    looks_like_invoice) even after its one retry - e.g. a signature/logo or
    T&Cs/W9 document rendered as its own PDF that slipped past
    email_watcher's PDF-only attachment filter. Never got far enough to call
    JDE, so there's no OrderNumber/VendorInvoiceNo to key the exception
    dedupe on (see invoice_server.record_exception) - file_name stands in for
    VendorInvoiceNo instead, which is enough to keep two different bad files
    from colliding into the same dashboard row without pretending either one
    is a real ERP invoice number."""
    error_message = (
        "Document AI could not find an invoice number, invoice date, or "
        "total amount on this document - it may not be an invoice."
    )
    logger.error("not_an_invoice for %s: %s", state["file_name"], error_message)
    return _record_exception_outcome(
        state, "not_an_invoice", {"VendorInvoiceNo": state["file_name"]}, error_message
    )


def handle_missing_order_number(state: InvoiceState) -> dict:
    """Routed here by route_after_parse when Document AI found an invoice
    (route_after_parse's not_an_invoice check already passed) but no
    purchase_order field on it. Caught before build_voucher_payload/
    call_voucher_match rather than letting it through: jde_client.
    build_payload would otherwise send JDE an empty OrderNumber, and JDE
    reports that back as "order information not found" - the same error
    message a real, populated-but-unknown PO number produces (see
    jde_client.classify_voucher_match_error) - which would make a plainly
    missing PO number indistinguishable from a mistyped one on the
    dashboard."""
    extracted = state.get("extracted") or {}
    vendor_invoice_no = extracted.get("invoice_number") or state["file_name"]
    error_message = "No purchase order / order number was found on this invoice."
    logger.error("missing_order_number for %s: %s", state["file_name"], error_message)
    return _record_exception_outcome(
        state, "missing_order_number", {"VendorInvoiceNo": vendor_invoice_no}, error_message
    )


def handle_incorrect_order_number(state: InvoiceState) -> dict:
    """Routed here by route_after_parse when the extracted purchase_order
    contains any non-numeric character (has_valid_order_number). Caught before
    JDE for the same reason as handle_missing_order_number."""
    extracted = state.get("extracted") or {}
    order_number = extracted.get("purchase_order")
    vendor_invoice_no = extracted.get("invoice_number") or state["file_name"]
    error_message = f"The order number '{order_number}' is not valid - order numbers must contain digits only."
    logger.error("incorrect_order_number for %s: %s", state["file_name"], error_message)
    return _record_exception_outcome(
        state,
        "incorrect_order_number",
        {"OrderNumber": order_number, "VendorInvoiceNo": vendor_invoice_no},
        error_message,
    )


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
        ("check_already_processed", check_already_processed),
        ("build_voucher_payload", build_voucher_payload),
        ("call_voucher_match", call_voucher_match),
        ("move_file_to_processed", move_file_to_processed),
        ("attach_account_numbers", attach_account_numbers),
        ("resolve_item_numbers", resolve_item_numbers),
        ("create_po", create_po),
        ("report_metrics_success", report_metrics_success),
        ("skip_no_receipt", skip_no_receipt),
        ("handle_duplicate_invoice", handle_duplicate_invoice),
        ("handle_already_processed", handle_already_processed),
        ("handle_order_not_found", handle_order_not_found),
        ("handle_not_an_invoice", handle_not_an_invoice),
        ("handle_missing_order_number", handle_missing_order_number),
        ("handle_incorrect_order_number", handle_incorrect_order_number),
        ("handle_parse_error", handle_parse_error),
    ]:
        graph.add_node(name, _log_node(name, fn))

    graph.add_edge(START, "read_and_encode_file")
    graph.add_edge("read_and_encode_file", "extract_fields")
    graph.add_conditional_edges(
        "extract_fields",
        route_after_parse,
        {
            "check_already_processed": "check_already_processed",
            "handle_not_an_invoice": "handle_not_an_invoice",
            "handle_missing_order_number": "handle_missing_order_number",
            "handle_incorrect_order_number": "handle_incorrect_order_number",
            "handle_parse_error": "handle_parse_error",
        },
    )

    graph.add_conditional_edges(
        "check_already_processed",
        route_after_already_processed_check,
        {
            "build_voucher_payload": "build_voucher_payload",
            "handle_already_processed": "handle_already_processed",
        },
    )

    graph.add_edge("build_voucher_payload", "call_voucher_match")
    graph.add_conditional_edges(
        "call_voucher_match",
        route_after_voucher_match,
        {
            "move_file_to_processed": "move_file_to_processed",
            "skip_no_receipt": "skip_no_receipt",
            "handle_duplicate_invoice": "handle_duplicate_invoice",
            "handle_order_not_found": "handle_order_not_found",
        },
    )

    graph.add_edge("move_file_to_processed", "attach_account_numbers")
    graph.add_edge("attach_account_numbers", "resolve_item_numbers")
    graph.add_edge("resolve_item_numbers", "create_po")
    graph.add_edge("create_po", "report_metrics_success")

    graph.add_edge("report_metrics_success", END)
    graph.add_edge("skip_no_receipt", END)
    graph.add_edge("handle_duplicate_invoice", END)
    graph.add_edge("handle_already_processed", END)
    graph.add_edge("handle_incorrect_order_number", END)
    graph.add_edge("handle_order_not_found", END)
    graph.add_edge("handle_not_an_invoice", END)
    graph.add_edge("handle_missing_order_number", END)
    graph.add_edge("handle_parse_error", END)

    return graph.compile()
