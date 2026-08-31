"""JD Edwards PO receipt inquiry ("voucher match").

Ports n8n's "Code in JavaScript2" (payload builder) and "Voucher Match"
(HTTP POST to the JDE orchestrator) nodes.
"""
import requests

from .settings import settings

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
    return {
        "username": settings.JDE_USERNAME,
        "password": settings.JDE_PASSWORD,
        "OrderNumber": str(extracted.get("purchase_order") or ""),
        "OrderType": "",
        "OrderCompany": "",
        "VendorInvoiceNo": extracted.get("invoice_number") or "",
        "Detail": detail,
    }


def voucher_match(payload: dict) -> dict:
    resp = requests.post(settings.JDE_ORCHESTRATOR_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def has_receipt_records(voucher_match_response: dict) -> bool:
    records = voucher_match_response.get(RECEIPT_INQUIRY_KEY, {}).get("records", 0)
    return bool(records and records > 0)
