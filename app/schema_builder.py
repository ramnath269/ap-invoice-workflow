"""Builds the Gemini response schema from config/schema_fields.json.

Ports n8n's "Code in JavaScript3" node, which built the same shape from the
"schema_fields" / "schema_array_fields" Data Table nodes.
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


def build_extraction_schema() -> dict:
    config = load_field_config()
    fields = config.get("fields", [])
    array_fields = config.get("array_fields", [])

    schema: dict = {"type": "object", "properties": {}}
    for field in fields:
        key = field["name"]
        if field.get("is_array"):
            item_properties = {
                af["field_name"]: {"type": af["type"]}
                for af in array_fields
                if af["array_name"] == key
            }
            schema["properties"][key] = {
                "type": "array",
                "items": {"type": "object", "properties": item_properties},
            }
        else:
            schema["properties"][key] = {"type": field["type"]}
    logger.info(
        "schema_builder.build_extraction_schema: %d fields (%d array fields)",
        len(fields),
        len(array_fields),
    )
    return schema


def account_number_for(field_name: str) -> str | None:
    for field in load_field_config().get("fields", []):
        if field["name"] == field_name:
            return field.get("account_number")
    return None
