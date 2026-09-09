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
            "ItemNo": str(p.get("item") or p.get("supplier_item_number") or ""),
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
    if not resp.ok:
        logger.error(
            "jde_client.voucher_match: HTTP %d response body=%s",
            resp.status_code,
            resp.text,
        )
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


ERROR_DUPLICATE_INVOICE = "duplicate_invoice"
ERROR_ORDER_NOT_FOUND = "order_not_found"

# Substrings of JDE's ErrorMessage that identify each exception case. JDE
# reuses the same ErrorCode ("1") for both, so ErrorMessage text is the only
# reliable discriminator - confirmed against live responses:
#   ErrorCode=1 ErrorMessage="Order information not found"
#   ErrorCode=1 ErrorMessage="Invoice already exists with following details"
_ERROR_MESSAGE_PATTERNS = {
    ERROR_DUPLICATE_INVOICE: "invoice already exists",
    ERROR_ORDER_NOT_FOUND: "order information not found",
}


def classify_voucher_match_error(voucher_match_response: dict) -> str | None:
    """Maps JDE's ErrorMessage to one of the known exception cases, or None
    if the response carries no recognized error (e.g. a clean receipt-record
    lookup, or a genuinely empty receipt file with no ErrorMessage at all -
    that's handled separately by has_receipt_records, not as an exception)."""
    message = (voucher_match_response.get("ErrorMessage") or "").strip().lower()
    if not message:
        return None
    for reason, pattern in _ERROR_MESSAGE_PATTERNS.items():
        if pattern in message:
            return reason
    logger.warning("jde_client.classify_voucher_match_error: unrecognized ErrorMessage=%r", message)
    return None
