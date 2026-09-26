"""Final PO write-back + workflow metrics.

Ports n8n's "HTTP Request5" (create-po) and "HTTP Request6"
(workflow-metrics) nodes.
"""
import logging
import os
import time

import requests

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")


def create_po(
    pdf_fields: dict, erp_fields: dict, file_path: str, item_suggestions: list[dict] | None = None
) -> dict:
    legacy_file_path = f"/processed_files/{os.path.basename(file_path)}"
    logger.info(
        "invoice_server.create_po: POST %s/create-po file_path=%s vendor=%s po=%s item_suggestions=%d",
        settings.INVOICE_SERVER_BASE_URL,
        legacy_file_path,
        pdf_fields.get("vendor"),
        pdf_fields.get("purchase_order"),
        len(item_suggestions or []),
    )
    resp = requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/create-po",
        # n8n's bodyParameters node sent form-encoded data, but the server's
        # Express handler destructures req.body assuming JSON (confirmed by
        # its "Cannot destructure ... req.body is undefined" 500 response to
        # a form-encoded request) - so this sends JSON instead.
        json={
            "pdf_fields": pdf_fields,
            "erp_fields": erp_fields,
            # The server's handler does file_path.replace('/processed_files/', '')
            # then path.join('/home/node/.n8n-files', ...) - a leftover from n8n
            # where file_path was always this short relative fragment. Sending our
            # real absolute host path breaks that string replace, so match the
            # format it expects instead.
            "file_path": legacy_file_path,
            # Lines JDE's voucher-match couldn't resolve but item_resolution
            # found a fuzzy-matched candidate for - the PO is created either
            # way, these just surface as "needs review" in the dashboard.
            "item_suggestions": item_suggestions or [],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    logger.info("invoice_server.create_po: response=%s", data)
    return data


def record_exception(
    pdf_fields: dict, erp_fields: dict, file_path: str, exception_reason: str, error_message: str
) -> dict:
    """Persists an invoice that failed voucher-match with a recognized JDE
    error (duplicate invoice / order not found) instead of silently dropping
    it - only report_metrics used to be called for these, so the invoice
    itself never showed up anywhere for a person to review.

    invoice-server dedupes on (OrderNumber, VendorInvoiceNo, exception_reason)
    before inserting, so resubmitting the same failing invoice repeatedly
    (e.g. a folder watcher rescan) doesn't pile up duplicate exception
    records - see POST /create-po-exception.
    """
    legacy_file_path = f"/processed_files/{os.path.basename(file_path)}"
    logger.info(
        "invoice_server.record_exception: POST %s/create-po-exception reason=%s order=%s vendor_invoice=%s",
        settings.INVOICE_SERVER_BASE_URL,
        exception_reason,
        erp_fields.get("OrderNumber"),
        erp_fields.get("VendorInvoiceNo"),
    )
    resp = requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/create-po-exception",
        json={
            "pdf_fields": pdf_fields,
            "erp_fields": erp_fields,
            "file_path": legacy_file_path,
            "exception_reason": exception_reason,
            "error_message": error_message,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    logger.info("invoice_server.record_exception: response=%s", data)
    return data


def get_po(record_id: str) -> dict:
    """Fetches a purchase_order/exception doc by CouchDB _id - used by
    reprocess_api to load the pdf_fields/erp_fields/exception_reason of a
    duplicate-invoice exception before retrying voucher-match."""
    logger.info("invoice_server.get_po: GET %s/po/%s", settings.INVOICE_SERVER_BASE_URL, record_id)
    resp = requests.get(f"{settings.INVOICE_SERVER_BASE_URL}/po/{record_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def update_po(
    record_id: str,
    pdf_fields: dict,
    erp_fields: dict,
    exception_reason: str | None,
    error_message: str | None,
    item_suggestions: list[dict] | None = None,
) -> dict:
    """Overwrites an existing doc's pdf_fields/erp_fields/exception fields in
    place (POST /update-po/:id), used by reprocess_api to write back a
    voucher-match retry's outcome onto the SAME record instead of inserting
    a new one via create_po/record_exception.

    exception_reason must always be passed explicitly (None to clear it on a
    clean success, or a string to record a new/different exception) - the
    server only touches exception_reason/error_message when the key is
    present in the request body at all, so it can tell "clear it" apart from
    "the caller doesn't know about this field" (existing callers of
    /update-po/:id, e.g. the dashboard's plain status updates, never send it
    and must leave it untouched).

    item_suggestions is the opposite: omit it (leave the default None) unless
    a fresh resolve_item_numbers pass actually ran and should replace
    whatever suggestions were on the doc - passing None here means the
    "item_suggestions" key never lands in the request body at all, versus
    exception_reason's explicit None which the server reads as "clear it."
    Sending a bare `None`/`null` for item_suggestions instead of omitting the
    key would make the server wipe out suggestions a person may have already
    confirmed, on a request that never meant to touch them.
    """
    logger.info(
        "invoice_server.update_po: POST %s/update-po/%s exception_reason=%s item_suggestions=%s",
        settings.INVOICE_SERVER_BASE_URL,
        record_id,
        exception_reason,
        "omitted" if item_suggestions is None else len(item_suggestions),
    )
    payload = {
        "pdf_fields": pdf_fields,
        "erp_fields": erp_fields,
        "exception_reason": exception_reason,
        "error_message": error_message,
    }
    if item_suggestions is not None:
        payload["item_suggestions"] = item_suggestions
    resp = requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/update-po/{record_id}",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    logger.info("invoice_server.update_po: response=%s", data)
    return data


def lookup_item_crossref(supplier_number, item_number: str) -> str | None:
    """Checks invoice-server's item_crossref cache for a previously
    user-confirmed (supplier_number, item_number) -> assigned_item_number
    mapping. Returns None on a cache miss OR on any error reaching
    invoice-server - a lookup failure here should fall through to the
    fuzzy-match fallback, not abort the run."""
    try:
        resp = requests.get(
            f"{settings.INVOICE_SERVER_BASE_URL}/item-crossref",
            params={"supplier_number": supplier_number, "item_number": item_number},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception(
            "invoice_server.lookup_item_crossref: lookup failed for supplier=%s item=%s - treating as miss",
            supplier_number,
            item_number,
        )
        return None

    if not data.get("found"):
        return None
    return data.get("crossref", {}).get("assigned_item_number")


def find_existing_po(order_number: str, vendor_invoice_no: str) -> dict | None:
    """Looks up an existing SUCCESSFUL purchase_order doc (invoice-server's
    GET /po/lookup - only matches docs with no exception_reason, i.e. ones
    that made it all the way through create_po) for the same
    (OrderNumber, VendorInvoiceNo). Called by graph.py's
    check_already_processed before ever calling JDE, so reprocessing an
    invoice that's already been fully processed gets caught locally instead
    of only being caught on the cases JDE itself happens to flag as a
    duplicate (see jde_client.ERROR_DUPLICATE_INVOICE). Returns None on a
    miss OR on any error reaching invoice-server - a lookup failure here
    should fall through to normal processing, not abort the run."""
    try:
        resp = requests.get(
            f"{settings.INVOICE_SERVER_BASE_URL}/po/lookup",
            params={"order_number": order_number, "vendor_invoice_no": vendor_invoice_no},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception(
            "invoice_server.find_existing_po: lookup failed for order=%s vendor_invoice=%s - treating as miss",
            order_number,
            vendor_invoice_no,
        )
        return None

    if not data.get("found"):
        return None
    existing = data.get("po")
    logger.info(
        "invoice_server.find_existing_po: found existing record id=%s for order=%s vendor_invoice=%s",
        (existing or {}).get("_id"),
        order_number,
        vendor_invoice_no,
    )
    return existing


def report_metrics(
    execution_id: str,
    status: str,
    execution_start_ms: float,
    failed_node: str | None = None,
    error: str | None = None,
    dependency_timings: list[dict] | None = None,
) -> None:
    """Reports one workflow_metric doc to invoice-server's CouchDB-backed
    /workflow-metrics endpoint (its handler is a plain `{...req.body}` spread
    into the doc, so any extra fields here are stored, not rejected).

    Called for every terminal outcome now - success, skipped_no_receipt,
    parse_error, and exception - not just success, so failures actually show
    up in GET /metrics instead of being invisible there. Deliberately never
    raises: metrics reporting must not be able to crash the run it's
    measuring, especially on the exception path where invoice-server being
    unreachable may be related to the original failure.
    """
    execution_time_ms = int(time.time() * 1000) - int(execution_start_ms)
    payload = {
        "workflow_name": settings.WORKFLOW_NAME,
        "execution_id": execution_id,
        "status": status,
        "execution_time_ms": execution_time_ms,
    }
    if failed_node:
        payload["failed_node"] = failed_node
    if error:
        payload["error"] = error
    if dependency_timings:
        payload["dependency_timings"] = dependency_timings

    logger.info(
        "invoice_server.report_metrics: status=%s execution_time_ms=%d failed_node=%s dependency_calls=%d",
        status,
        execution_time_ms,
        failed_node,
        len(dependency_timings or []),
    )
    try:
        resp = requests.post(
            f"{settings.INVOICE_SERVER_BASE_URL}/workflow-metrics", json=payload, timeout=30
        )
        if not resp.ok:
            logger.warning(
                "invoice_server.report_metrics: server returned %s: %s",
                resp.status_code,
                resp.text[:300],
            )
    except Exception:
        logger.exception(
            "invoice_server.report_metrics: failed to report metrics (status=%s) - continuing", status
        )
