"""JD Edwards PO receipt inquiry ("voucher match").

Ports n8n's "Code in JavaScript2" (payload builder) and "Voucher Match"
(HTTP POST to the JDE orchestrator) nodes.
"""
import logging

import requests

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

RECEIPT_INQUIRY_KEY = "55_DREQ_PO_ReceiptFile_Inquiry_V2"


def build_payload(extracted: dict) -> dict:
    products = extracted.get("products") or []
    detail = [
        {
            "ItemNo": str(
                p.get("item_number") or p.get("supplier_item_number") or p.get("item") or ""
            ),
            "SequenceNumber": i + 1,
            "Quantity": p.get("quantity"),
            "UOM": p.get("unit_of_measure"),
        }
        for i, p in enumerate(products)
    ]
    payload = {
        "username": settings.JDE_USERNAME,
        "password": settings.JDE_PASSWORD,
        "OrderNumber": str(extracted.get("purchase_order") or ""),
        "OrderType": "",
        "OrderCompany": "",
        "VendorInvoiceNo": extracted.get("invoice_number") or "",
        "Detail": detail,
    }
    logger.info(
        "jde_client.build_payload: OrderNumber=%s VendorInvoiceNo=%s line_items=%d",
        payload["OrderNumber"],
        payload["VendorInvoiceNo"],
        len(detail),
    )
    return payload


def voucher_match(payload: dict) -> dict:
    if settings.JDE_MOCK_MODE:
        logger.warning(
            "JDE_MOCK_MODE is on - returning a canned response instead of calling %s",
            settings.JDE_ORCHESTRATOR_URL,
        )
        return {RECEIPT_INQUIRY_KEY: {"records": settings.JDE_MOCK_RECORDS}}

    logger.info(
        "jde_client.voucher_match: POST %s OrderNumber=%s VendorInvoiceNo=%s",
        settings.JDE_ORCHESTRATOR_URL,
        payload.get("OrderNumber"),
        payload.get("VendorInvoiceNo"),
    )
    resp = requests.post(settings.JDE_ORCHESTRATOR_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    logger.info(
        "jde_client.voucher_match: response ErrorCode=%s ErrorMessage=%s records=%s",
        data.get("ErrorCode"),
        data.get("ErrorMessage"),
        data.get(RECEIPT_INQUIRY_KEY, {}).get("records"),
    )
    return data


def has_receipt_records(voucher_match_response: dict) -> bool:
    records = voucher_match_response.get(RECEIPT_INQUIRY_KEY, {}).get("records", 0)
    valid = bool(records and records > 0)
    logger.info("jde_client.has_receipt_records: records=%s valid=%s", records, valid)
    return valid
