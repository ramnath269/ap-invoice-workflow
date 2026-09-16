"""Internal HTTP endpoint for retrying voucher-match on a corrected invoice
number, for the "duplicate_invoice" exception case.

Run as a background thread inside app/watcher.py's process (see watch()) -
not the full LangGraph (that starts at read_and_encode_file/extract_fields,
re-running OCR pointlessly on a file that's already been parsed and moved to
processed_files/) but the same jde_client calls and post-voucher-match node
functions (attach_account_numbers/resolve_item_numbers) the graph itself
uses, so a reprocessed invoice gets the same treatment a first-time success
would.

Binds 0.0.0.0, not 127.0.0.1 - see settings.REPROCESS_API_HOST's comment for
why (the caller, server/index.js, runs in a Docker container reached via
host.docker.internal, not this host's own loopback).
"""
import logging
import os

from flask import Flask, jsonify, request

from . import graph, invoice_server, jde_client
from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

app = Flask(__name__)


@app.post("/reprocess-duplicate-invoice")
def reprocess_duplicate_invoice():
    body = request.get_json(silent=True) or {}
    record_id = body.get("record_id")
    new_invoice_number = (body.get("invoice_number") or "").strip()
    if not record_id or not new_invoice_number:
        return jsonify({"error": "record_id and invoice_number are required."}), 400

    try:
        record = invoice_server.get_po(record_id)
    except Exception as exc:
        logger.exception("reprocess_duplicate_invoice: fetch failed for %s", record_id)
        return jsonify({"error": f"Could not load invoice record: {exc}"}), 502

    if record.get("exception_reason") != jde_client.ERROR_DUPLICATE_INVOICE:
        return jsonify({"error": "Not a duplicate_invoice exception - reprocess not applicable."}), 404

    extracted = dict(record.get("pdf_fields") or {})
    extracted["invoice_number"] = new_invoice_number
    payload = jde_client.build_payload(extracted)

    logger.info(
        "reprocess_duplicate_invoice: retrying voucher-match for %s with invoice_number=%s",
        record_id, new_invoice_number,
    )
    try:
        response = jde_client.voucher_match(payload)
    except Exception as exc:
        # JDE call itself failed (timeout/network) - not a classified
        # voucher-match outcome, so nothing gets written to CouchDB. The
        # record is left exactly as it was for a retry.
        logger.exception("reprocess_duplicate_invoice: JDE call failed for %s", record_id)
        return jsonify({"error": f"JDE voucher-match call failed: {exc}"}), 502

    error = jde_client.classify_voucher_match_error(response)
    has_records = jde_client.has_receipt_records(response)

    if error:
        # Still an exception - possibly a different reason now (e.g. the
        # corrected number doesn't exist at all -> order_not_found, or it
        # collides with a *different* existing voucher). Reflect whatever
        # JDE actually said now rather than leaving the old reason stale.
        error_message = response.get("ErrorMessage") or ""
        logger.warning(
            "reprocess_duplicate_invoice: %s still an exception (%s) for invoice_number=%s: %s",
            record_id, error, new_invoice_number, error_message,
        )
        try:
            invoice_server.update_po(
                record_id,
                pdf_fields={**extracted, "status": "EXCEPTION"},
                erp_fields=response,
                exception_reason=error,
                error_message=error_message,
            )
        except Exception as exc:
            logger.exception("reprocess_duplicate_invoice: save failed (re-exception) for %s", record_id)
            return jsonify({"error": f"JDE returned {error!r} but saving the update failed: {exc}"}), 502
        return jsonify({"outcome": "exception", "exception_reason": error, "error_message": error_message}), 200

    # No recognized error - reuse the same node functions the main graph
    # calls after a successful voucher-match, without re-running OCR.
    state = {"extracted": extracted, "erp_fields": response, "voucher_match_response": response}
    state.update(graph.attach_account_numbers(state))
    state.update(graph.resolve_item_numbers(state))

    # Clear pdf_fields.status entirely rather than setting it to something -
    # that's what a normal first-time /create-po success looks like (no
    # status field at all), and the dashboard's mapStatus('') already
    # defaults that to 'pending', putting this back in the normal review
    # queue instead of leaving it looking like an exception.
    pdf_fields = dict(state["extracted"])
    pdf_fields.pop("status", None)
    item_suggestions = state.get("item_suggestions", [])
    try:
        invoice_server.update_po(
            record_id,
            pdf_fields=pdf_fields,
            erp_fields=response,
            exception_reason=None,
            error_message=None,
            item_suggestions=item_suggestions,
        )
    except Exception as exc:
        logger.exception("reprocess_duplicate_invoice: save failed (success) for %s", record_id)
        return jsonify({"error": f"JDE voucher-match succeeded but saving the result failed: {exc}"}), 502

    outcome = "success" if has_records else "no_receipt_yet"
    logger.info("reprocess_duplicate_invoice: %s -> %s for invoice_number=%s", record_id, outcome, new_invoice_number)
    return jsonify({"outcome": outcome, "item_suggestions": item_suggestions}), 200


def run() -> None:
    """Runs in its own daemon thread (see watcher.watch()) - a thread dying
    on its own (e.g. the port is already in use) wouldn't fail the systemd
    unit or even show up as unhealthy, since the watchdog Observer on the
    main thread would carry on regardless. os._exit(1) turns that into a
    real process exit so systemd's Restart=on-failure actually engages and
    the failure is visible in `systemctl status`/the journal instead of
    silently leaving this endpoint dead."""
    logger.info(
        "reprocess_api: starting Flask on %s:%d", settings.REPROCESS_API_HOST, settings.REPROCESS_API_PORT
    )
    try:
        app.run(
            host=settings.REPROCESS_API_HOST,
            port=settings.REPROCESS_API_PORT,
            threaded=True,
            use_reloader=False,
            debug=False,
        )
    except Exception:
        logger.exception("reprocess_api: Flask server failed - exiting process so systemd can restart it")
        os._exit(1)
