"""Final PO write-back + workflow metrics.

Ports n8n's "HTTP Request5" (create-po) and "HTTP Request6"
(workflow-metrics) nodes.
"""
import time

import requests

from .settings import settings


def create_po(pdf_fields: dict, erp_fields: dict, file_path: str) -> dict:
    resp = requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/create-po",
        # n8n's bodyParameters node sent form-encoded data, but the server's
        # Express handler destructures req.body assuming JSON (confirmed by
        # its "Cannot destructure ... req.body is undefined" 500 response to
        # a form-encoded request) - so this sends JSON instead.
        json={
            "pdf_fields": pdf_fields,
            "erp_fields": erp_fields,
            "file_path": file_path,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def report_metrics(execution_id: str, status: str, execution_start_ms: float) -> None:
    requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/workflow-metrics",
        json={
            "workflow_name": settings.WORKFLOW_NAME,
            "workflow_id": settings.WORKFLOW_ID,
            "execution_id": execution_id,
            "status": status,
            "execution_time_ms": int(time.time() * 1000) - int(execution_start_ms),
        },
        timeout=30,
    )
