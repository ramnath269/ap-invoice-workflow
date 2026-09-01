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


def create_po(pdf_fields: dict, erp_fields: dict, file_path: str) -> dict:
    legacy_file_path = f"/processed_files/{os.path.basename(file_path)}"
    logger.info(
        "invoice_server.create_po: POST %s/create-po file_path=%s vendor=%s po=%s",
        settings.INVOICE_SERVER_BASE_URL,
        legacy_file_path,
        pdf_fields.get("vendor"),
        pdf_fields.get("purchase_order"),
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
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    logger.info("invoice_server.create_po: response=%s", data)
    return data


def report_metrics(execution_id: str, status: str, execution_start_ms: float) -> None:
    execution_time_ms = int(time.time() * 1000) - int(execution_start_ms)
    logger.info(
        "invoice_server.report_metrics: execution_id=%s status=%s execution_time_ms=%d",
        execution_id,
        status,
        execution_time_ms,
    )
    requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/workflow-metrics",
        json={
            "workflow_name": settings.WORKFLOW_NAME,
            "workflow_id": settings.WORKFLOW_ID,
            "execution_id": execution_id,
            "status": status,
            "execution_time_ms": execution_time_ms,
        },
        timeout=30,
    )
