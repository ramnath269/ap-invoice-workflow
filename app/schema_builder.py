"""GL account-number lookup from config/schema_fields.json.

Ports n8n's "Code in JavaScript3" node, which read the same per-field
account numbers from the "schema_fields" Data Table node.
"""
from __future__ import annotations

import json
import logging

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_cache: dict | None = None


def load_field_config() -> dict:
    global _cache
    if _cache is None:
        with open(settings.SCHEMA_FIELDS_CONFIG, "r") as f:
            _cache = json.load(f)
    return _cache


def account_number_for(field_name: str) -> str | None:
    for field in load_field_config().get("fields", []):
        if field["name"] == field_name:
            return field.get("account_number")
    return None
