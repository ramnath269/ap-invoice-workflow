"""Final PO write-back + workflow metrics.

Ports n8n's "HTTP Request5" (create-po, form-encoded body) and
"HTTP Request6" (workflow-metrics, JSON body) nodes.
"""
import json
import time

import requests

from .settings import settings


def create_po(pdf_fields: dict, erp_fields: dict, file_name: str) -> dict:
    resp = requests.post(
        f"{settings.INVOICE_SERVER_BASE_URL}/create-po",
        # n8n's bodyParameters form-encodes object values as JSON strings -
        # mirrored here since the receiving endpoint expects that shape.
        data={
            "pdf_fields": json.dumps(pdf_fields),
            "erp_fields": json.dumps(erp_fields),
            "file_path": f"/{settings.PROCESSED_SUBFOLDER}/{file_name}",
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
