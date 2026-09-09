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
        "workflow_id": settings.WORKFLOW_ID,
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
