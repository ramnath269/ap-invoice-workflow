"""OCR + field extraction via Google Document AI.

Ports n8n's "Code in JavaScript" (base64 encode) and "HTTP Request3"
(Document AI :process call) nodes. DOCAI_PROCESSOR_ID now points at a custom
extractor processor that returns structured `document.entities` alongside
the raw OCR `document.text` in one call - see parse_entities() below for how
those entities are converted into the same canonical field shape
gemini_extract.parse_response() used to produce, so this replaces the old
OCR-only processor + separate Gemini extraction step without changing what
downstream graph nodes consume.
"""
import base64
import logging
import mimetypes
import re

import requests

from .google_auth import get_access_token
from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_MONEY_STRIP_RE = re.compile(r"[^\d.\-]")
_LEADING_SYMBOL_RE = re.compile(r"^([^\d\s.\-]+)")

# Entity types this processor's schema doesn't carry at all - defaulted the
# same way Gemini did when a field was absent from the source document.
_MISSING_TOP_LEVEL_DEFAULTS = {"purchase_order": "", "currency": "", "special_level_tax": "0.00"}
_MISSING_PRODUCT_DEFAULTS = {"unit_of_measure": "", "line_number": ""}

_TOP_LEVEL_MONEY_FIELDS = {
    "county_level_tax", "handling_fee", "sales_tax", "state_level_tax",
    "sub_total", "total_amount_due", "total_freight",
}
_TOP_LEVEL_TEXT_FIELDS = {"due_date", "invoice_date", "invoice_number", "payment_terms", "vendor"}

# DocAI's "item" entity carries the actual part/item number (JDE's
# first-choice identifier per jde_client.build_payload) -> canonical
# "item". Its "description" entity carries a short name (e.g. "SENSOR")
# -> canonical "description". "supplier_item_number" keeps its own name -
# this processor populates it reliably (Gemini often left it blank).
_PRODUCT_MONEY_FIELDS = {"amount", "unit_price"}
_PRODUCT_ENTITY_TO_FIELD = {
    "item": "item",
    "description": "description",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "amount": "amount",
    "supplier_item_number": "supplier_item_number",
    "purchase_order_number": "purchase_order_number",
}

# This processor's foundation-model backend isn't perfectly deterministic:
# the same bytes can come back on one call with every entity populated and
# on the next with a property silently absent (see graph.py's extract_fields
# retry). CONFIDENCE_THRESHOLD and the two sets below are what
# find_extraction_issues() checks before we trust a response - this is a
# manually-kept-in-sync snapshot of the processor's schema
# (projects/.../schemas/2fd6b5078c2734d4), not a live fetch, so re-check it
# here after any schema edit in the Document AI console. As of
# schemaVersions/524ee19051f45a55: top level is REQUIRED_ONCE for
# invoice_date/total_amount_due/invoice_number only (vendor was
# REQUIRED_ONCE, now OPTIONAL_ONCE - it was coming back empty on every call
# for at least one real vendor's invoice layout, making it a useless retry
# trigger); every other top-level field and, previously, every product
# property was OPTIONAL_ONCE - "item" is now REQUIRED_MULTIPLE (the schema
# itself now enforces the field that was actually flaking, on Google's
# side). _REQUIRED_PRODUCT_PROPERTY_TYPES stays deliberately broader than
# schema-required alone: description/quantity/unit_price/amount are still
# schema-optional but make up the substance of an invoice line (a line
# always has all four on a real invoice) and are what we directly observed
# a foundation-model call drop on an otherwise-clean rerun, whereas fields
# like unit_of_measure/line_number/supplier_item_number/purchase_order_number
# are routinely absent even on correctly-extracted invoices and would retry
# every single document if included here.
CONFIDENCE_THRESHOLD = 0.80
_REQUIRED_TOP_LEVEL_TYPES = {"invoice_date", "total_amount_due", "invoice_number"}
_REQUIRED_PRODUCT_PROPERTY_TYPES = {"item", "description", "quantity", "unit_price", "amount"}

# Some vendors' line-item tables (observed on Ford: category subtotal rows
# like "MOTORCRAFT LT REPAIR  136  $8,487.93" summarizing a block of part
# rows) get picked up by the processor as their own "products" entity -
# they carry amount/quantity/description like a real line but, structurally,
# can never have an item number, unit price, or supplier item number (a
# subtotal isn't a single part). Treated as non-products: excluded from
# find_missing_fields/find_extraction_issues (no point retrying a field that
# was never going to be there) and dropped entirely from parse_entities'
# output (an item-less "product" is useless to JDE voucher-match/PO
# creation downstream anyway).
_PRODUCT_IDENTITY_FIELDS = {"item", "unit_price", "supplier_item_number"}
_PRODUCT_CONTENT_FIELDS = {"amount", "quantity", "description"}


def _is_subtotal_artifact(present_fields: set) -> bool:
    return not (present_fields & _PRODUCT_IDENTITY_FIELDS) and bool(present_fields & _PRODUCT_CONTENT_FIELDS)


def _clean_text(value) -> str:
    if not value:
        return ""
    return value.replace("\n", " ").replace("\r", " ").strip()


def _clean_money(value) -> str:
    return _MONEY_STRIP_RE.sub("", _clean_text(value))


def _leading_symbol(value) -> str:
    match = _LEADING_SYMBOL_RE.match(_clean_text(value))
    return match.group(1) if match else ""


def encode_file(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        content = f.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "application/pdf"
    logger.info(
        "document_ai.encode_file: path=%s bytes=%d mime=%s", file_path, len(content), mime_type
    )
    return base64.b64encode(content).decode("utf-8"), mime_type


def process_document(base64_content: str, mime_type: str, schema_override: dict | None = None) -> dict:
    """Calls the Document AI processor and returns its full raw response
    (document.text plus, for a custom extractor processor, document.entities).

    schema_override, when given, is sent as processOptions.schemaOverride - a
    call-scoped full replacement of the processor's stored schema (confirmed
    empirically: a call with a one-field override returns ONLY that field,
    none of the stored schema's other fields, and never touches the stored
    schema itself). Used by extract_fields' targeted second call to
    re-extract just the fields a first call got wrong, with vendor-specific
    descriptions substituted in - see build_schema_override.

    Requires v1beta3, not v1 - v1's schema message doesn't support the
    description/entityTypeMetadata fields schemaOverride payloads carry and
    rejects them with INVALID_ARGUMENT. v1beta3 handles a plain (no-override)
    call identically to v1, so there's no reason to keep both versions
    around for the two call shapes."""
    url = (
        f"https://us-documentai.googleapis.com/v1beta3/projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{settings.DOCAI_LOCATION}/processors/{settings.DOCAI_PROCESSOR_ID}:process"
    )
    body = {"rawDocument": {"content": base64_content, "mimeType": mime_type}}
    if schema_override:
        body["processOptions"] = {"schemaOverride": schema_override}
    logger.info(
        "document_ai.process_document: POST %s mime=%s encoded_len=%d schema_override=%s",
        url, mime_type, len(base64_content), bool(schema_override),
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {get_access_token()}"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    document = data.get("document", {})
    logger.info(
        "document_ai.process_document: received %d chars of OCR text, %d entities",
        len(document.get("text", "")),
        len(document.get("entities", [])),
    )
    return data


_live_schema_cache: dict | None = None


def fetch_live_schema() -> dict:
    """Fetches the processor's stored dataset schema (GET .../dataset/
    datasetSchema) and returns {"document": {field_name: prop}, "product":
    {field_name: prop}} - the generic (multi-vendor, as currently authored on
    the processor) field definitions used as the fallback source for
    build_schema_override when a field has no vendor-specific row, or the
    extracted vendor couldn't be classified at all.

    Cached for the process lifetime: this changes only when someone edits
    the schema in the Document AI console, which is rare and already
    requires re-syncing find_extraction_issues' hardcoded field sets above
    by hand - a process restart to pick up a schema edit is consistent with
    that existing precedent, and avoids an extra API round trip on every
    retry."""
    global _live_schema_cache
    if _live_schema_cache is not None:
        return _live_schema_cache

    url = (
        f"https://us-documentai.googleapis.com/v1beta3/projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{settings.DOCAI_LOCATION}/processors/{settings.DOCAI_PROCESSOR_ID}"
        f"/dataset/datasetSchema"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {get_access_token()}"}, timeout=30)
    resp.raise_for_status()
    entity_types = resp.json().get("documentSchema", {}).get("entityTypes", [])

    document_props, product_props = {}, {}
    for entity_type in entity_types:
        target = product_props if entity_type.get("name") == "products" else document_props
        for prop in entity_type.get("properties", []):
            target[prop["name"]] = {
                "name": prop["name"],
                "valueType": prop["valueType"],
                "occurrenceType": prop["occurrenceType"],
                "method": prop.get("method", "EXTRACT"),
                "description": prop.get("description", ""),
            }

    _live_schema_cache = {"document": document_props, "product": product_props}
    logger.info(
        "document_ai.fetch_live_schema: cached %d document field(s), %d product field(s)",
        len(document_props), len(product_props),
    )
    return _live_schema_cache


def run_ocr(base64_content: str, mime_type: str) -> str:
    return process_document(base64_content, mime_type)["document"]["text"]


def _split_entities(raw_response: dict) -> tuple[dict, list[dict]]:
    entities = raw_response.get("document", {}).get("entities", [])
    top_level = {}
    product_entities = []
    for entity in entities:
        if entity.get("type") == "products":
            product_entities.append(entity)
        else:
            top_level[entity.get("type")] = entity
    return top_level, product_entities


def find_extraction_issues(raw_response: dict) -> list[str]:
    """Checks a raw process_document() response for required fields
    (_REQUIRED_TOP_LEVEL_TYPES / _REQUIRED_PRODUCT_PROPERTY_TYPES) that are
    either absent or below CONFIDENCE_THRESHOLD. Returns a list of
    human-readable reasons (empty if the response looks trustworthy) - the
    caller (graph.py's extract_fields) logs these before retrying with a
    targeted schemaOverride (see find_missing_fields, which flags the same
    issues as a field-name set instead of messages)."""
    top_level, product_entities = _split_entities(raw_response)

    issues = []
    for field_type in _REQUIRED_TOP_LEVEL_TYPES:
        entity = top_level.get(field_type)
        if entity is None:
            issues.append(f"required field '{field_type}' missing")
        elif entity.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            issues.append(f"field '{field_type}' confidence {entity.get('confidence', 0):.2f} < {CONFIDENCE_THRESHOLD}")

    for line_num, entity in enumerate(product_entities, start=1):
        properties = {prop.get("type"): prop for prop in entity.get("properties", [])}
        if _is_subtotal_artifact(set(properties)):
            continue
        for field_type in _REQUIRED_PRODUCT_PROPERTY_TYPES:
            prop = properties.get(field_type)
            if prop is None:
                issues.append(f"product line {line_num}: '{field_type}' missing")
            elif prop.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                issues.append(
                    f"product line {line_num}: '{field_type}' confidence {prop.get('confidence', 0):.2f} < {CONFIDENCE_THRESHOLD}"
                )

    return issues


def find_missing_fields(raw_response: dict) -> dict:
    """Same check as find_extraction_issues (required field missing, or
    present below CONFIDENCE_THRESHOLD), but returns the affected field
    names instead of messages: {"top_level": set[str], "product": set[str]}.
    "product" is the union across every line - a schemaOverride's "products"
    entityType applies to the whole document-level re-extraction, there's no
    way to target "only line 3's unit_price". Empty sets mean the response
    looks trustworthy and no retry is needed."""
    top_level, product_entities = _split_entities(raw_response)

    missing_top_level = {
        field_type for field_type in _REQUIRED_TOP_LEVEL_TYPES
        if top_level.get(field_type) is None
        or top_level[field_type].get("confidence", 0) < CONFIDENCE_THRESHOLD
    }

    missing_product = set()
    for entity in product_entities:
        properties = {prop.get("type"): prop for prop in entity.get("properties", [])}
        if _is_subtotal_artifact(set(properties)):
            continue
        for field_type in _REQUIRED_PRODUCT_PROPERTY_TYPES:
            prop = properties.get(field_type)
            if prop is None or prop.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                missing_product.add(field_type)

    return {"top_level": missing_top_level, "product": missing_product}


def build_schema_override(missing: dict, vendor_fields: dict, base_schema: dict) -> dict:
    """Builds a processOptions.schemaOverride payload that asks for ONLY the
    fields in `missing` (as produced by find_missing_fields), so the second
    call re-extracts the minimum needed instead of the whole 26-field schema.

    For each missing field, a vendor-specific definition in `vendor_fields`
    (from vendor_schema.get_vendor_field_overrides) wins; otherwise falls
    back to the processor's own generic definition in `base_schema` (from
    fetch_live_schema) - covering both an unclassified/unseeded vendor and
    plain model flakiness on a field with no vendor-specific rule at all."""

    def field_entry(name: str, base_props: dict) -> dict:
        field = vendor_fields.get(name) or base_props[name]
        return {
            "name": name,
            "valueType": field["valueType"],
            "occurrenceType": field["occurrenceType"],
            "method": field.get("method", "EXTRACT"),
            "description": field["description"],
        }

    document_properties = [
        field_entry(name, base_schema["document"]) for name in sorted(missing["top_level"])
    ]

    entity_types = []
    if missing["product"]:
        # The top-level "products" property itself must be declared too -
        # it's what tells Document AI to look for line items at all using
        # the nested "products" entityType below.
        document_properties.append(dict(base_schema["document"]["products"]))
        # "item" is always included, even when it's not itself missing: a
        # narrower product schema can shift the model's own line
        # segmentation (observed directly - a document with 125 first-call
        # product lines came back with 124 on a quantity-only override), so
        # the Nth entry in each call's product list isn't reliably the same
        # invoice line. merge_extracted realigns by matching "item" values
        # instead of position - it's the field with the most consistently
        # high confidence across every real invoice tested so far, making it
        # the best available anchor.
        product_field_names = set(missing["product"]) | {"item"}
        entity_types.append({
            "name": "products",
            "baseTypes": ["object"],
            "properties": [
                field_entry(name, base_schema["product"]) for name in sorted(product_field_names)
            ],
        })

    entity_types.insert(0, {
        "name": "custom_extraction_document_type",
        "baseTypes": ["document"],
        "properties": document_properties,
    })
    return {"entityTypes": entity_types}


def merge_extracted(first: dict, second: dict, missing: dict) -> dict:
    """Merges a targeted second call's results into the first call's parsed
    output (both already run through parse_entities), overwriting only the
    fields that were actually part of the override AND came back non-empty -
    a field the second call still couldn't find keeps the first call's
    (already empty/absent) value rather than being blanked out again."""
    merged = dict(first)
    merged["confidence"] = dict(first.get("confidence", {}))

    for name in missing["top_level"]:
        value = second.get(name)
        if value:
            merged[name] = value
            merged["confidence"][name] = second.get("confidence", {}).get(name)

    if missing["product"] and second.get("products"):
        first_products = first.get("products", [])
        second_products = second["products"]

        # Realign by "item" (always requested in a product override, even
        # when it's not itself missing - see build_schema_override) instead
        # of raw list position: a narrower schema can change how many
        # product lines the model reports, so position alone isn't a
        # reliable match once the two calls' counts diverge. Duplicate item
        # numbers on the same invoice (seen in practice - a part ordered on
        # two separate lines) are handled by matching in first-seen order:
        # the Nth line with item "X" in the first call pairs with the Nth
        # line with item "X" in the second.
        second_by_item: dict = {}
        for i, product in enumerate(second_products):
            second_by_item.setdefault(product.get("item"), []).append(i)

        merged_products = []
        unmatched = 0
        for product in first_products:
            product = dict(product)
            product["confidence"] = dict(product.get("confidence", {}))
            candidates = second_by_item.get(product.get("item")) if product.get("item") else None
            if candidates:
                second_product = second_products[candidates.pop(0)]
                for name in missing["product"]:
                    value = second_product.get(name)
                    if value:
                        product[name] = value
                        product["confidence"][name] = second_product.get("confidence", {}).get(name)
            else:
                unmatched += 1
            merged_products.append(product)

        if unmatched:
            logger.warning(
                "document_ai.merge_extracted: could not match %d of %d product line(s) between "
                "calls by item number (override returned %d line(s), first call had %d) - left "
                "those lines' %s as-is",
                unmatched, len(first_products), len(second_products), len(first_products),
                sorted(missing["product"]),
            )
        merged["products"] = merged_products

    return merged


# Product fields a full product re-extraction asks for (graph.py's
# extract_fields, for a LINE_MATH_VENDORS invoice whose values are present
# but arithmetically wrong). Wider than _REQUIRED_PRODUCT_PROPERTY_TYPES
# because the retry's product list REPLACES the first call's wholesale
# rather than being merged into it - any product field left out here would
# be lost from every line.
FULL_PRODUCT_RETRY_FIELDS = {
    "item", "description", "quantity", "unit_price", "amount",
    "purchase_order_number", "supplier_item_number", "unit_of_measure",
}


def backfill_products(replacement: list[dict], original: list[dict]) -> int:
    """Fills, in place, fields a full product re-extraction left empty from
    the first call's matching line - observed on a 527-line Federated
    invoice, where the re-extraction got item/quantity/price/amount far more
    right but came back with "description" empty on ~410 lines the first
    call had filled. Never overwrites a non-empty value.

    Lines are paired by item (first-seen order, as merge_extracted does),
    falling back to a unique quantity/unit_price/amount match for the lines
    whose item the first call misread. Returns the number of fields filled."""
    by_item: dict = {}
    for i, product in enumerate(original):
        by_item.setdefault(product.get("item"), []).append(i)
    used = set()

    def numbers(p: dict) -> tuple:
        return tuple(_to_float(p.get(k)) for k in ("quantity", "unit_price", "amount"))

    filled = 0
    for product in replacement:
        match = None
        for i in by_item.get(product.get("item"), []) if product.get("item") else []:
            if i not in used:
                match = i
                break
        if match is None and None not in numbers(product):
            candidates = [i for i, p in enumerate(original) if i not in used and numbers(p) == numbers(product)]
            if len(candidates) == 1:
                match = candidates[0]
        if match is None:
            continue
        used.add(match)
        for name, value in original[match].items():
            if name in FULL_PRODUCT_RETRY_FIELDS and value and not product.get(name):
                product[name] = value
                filled += 1
    return filled


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _money_str(value: float) -> str:
    return f"{value:.2f}"


def _line_math_ok(product: dict) -> bool | None:
    """True/False whether quantity x unit_price == amount (to the cent), or
    None when any of the three is missing/unparseable and it can't be
    checked."""
    qty, price, amount = (_to_float(product.get(k)) for k in ("quantity", "unit_price", "amount"))
    if qty is None or price is None or amount is None:
        return None
    return round(qty * price, 2) == round(amount, 2)


def find_line_math_issues(extracted: dict) -> dict:
    """For a vendor in vendor_schema.LINE_MATH_VENDORS: flags product lines
    whose quantity, unit_price and amount are all present but don't satisfy
    quantity x unit_price == amount, and whether the line amounts sum to
    sub_total (or total_amount_due when no subtotal is printed). Catches the
    values find_missing_fields can't: present and high-confidence, but
    misread (a dropped digit or decimal point on a dot-matrix print).

    An unparseable expected total (e.g. a credit memo's trailing-minus
    "47.60-") skips the sum check rather than failing it."""
    products = extracted.get("products", [])
    bad_rows = [i for i, p in enumerate(products) if _line_math_ok(p) is False]
    line_sum = round(sum(_to_float(p.get("amount")) or 0.0 for p in products), 2)
    expected = _to_float(extracted.get("sub_total")) or _to_float(extracted.get("total_amount_due"))
    sum_ok = expected is None or line_sum == round(expected, 2)
    return {"bad_rows": bad_rows, "line_sum": line_sum, "expected": expected, "sum_ok": sum_ok}


def line_math_score(extracted: dict) -> tuple[int, bool]:
    """(number of arithmetically consistent product lines, whether the line
    amounts sum to the expected total) - compared as a tuple to decide
    whether a full product re-extraction is better than the first call."""
    consistent = sum(1 for p in extracted.get("products", []) if _line_math_ok(p))
    return consistent, find_line_math_issues(extracted)["sum_ok"]


def _is_dropped_character_misread(raw: str, correct: float) -> bool:
    """True when `raw` (the value exactly as extracted) can be produced by
    deleting characters - digits or the decimal point - from `correct` as
    it would be printed ('0.65' or '.65'): e.g. '65' / '2' from .65,
    '.68' from 22.68, '19.3' from 19.37, '8666' from 86.66. Rejects a raw
    value that has anything the correct one doesn't (e.g. '1.30' can't come
    from 130.00), so a correct value is never "repaired" into a wrong one."""
    raw = str(raw or "").strip()
    if not raw or raw == _money_str(correct):
        return False
    printed_forms = {_money_str(correct), _money_str(correct).lstrip("0")}
    for printed in printed_forms:
        chars = iter(printed)
        if all(ch in chars for ch in raw):
            return True
    return False


def repair_line_math(extracted: dict) -> list[str]:
    """For a vendor in vendor_schema.LINE_MATH_VENDORS: fills or corrects one
    of quantity/unit_price/amount per line from the other two, in place.
    Returns a description of each repair made (for logging).

    Only makes a correction the arithmetic uniquely supports:
    - a single missing value is computed from the other two (quantity only
      when it comes out whole, unit_price only when it comes out to the
      cent);
    - when all three are present but inconsistent, quantity is trusted (it's
      DERIVE-extracted and was correct on every tested line), and unit_price
      or amount is replaced only when the extracted value
      can be produced by deleting characters from the arithmetically correct
      value - the signature of a dropped digit or decimal point ('65' for
      .65, '.68' for 22.68, '19.3' for 19.37). Anything else is left for
      review rather than guessed at.

    Repaired fields are listed on the line under "repaired_fields" - like
    "confidence", it rides through unchanged since nothing downstream reads
    more than the specific field keys it expects."""
    repairs = []
    for line_num, product in enumerate(extracted.get("products", []), start=1):
        qty, price, amount = (_to_float(product.get(k)) for k in ("quantity", "unit_price", "amount"))
        fix = None
        if qty is None and price and amount is not None:
            candidate = amount / price
            if abs(candidate - round(candidate)) < 1e-6 and round(candidate) > 0:
                fix = ("quantity", str(int(round(candidate))))
        elif qty and price is None and amount is not None:
            candidate = round(amount / qty, 2)
            if round(candidate * qty, 2) == round(amount, 2):
                fix = ("unit_price", _money_str(candidate))
        elif qty is not None and price is not None and amount is None:
            fix = ("amount", _money_str(qty * price))
        elif _line_math_ok(product) is False and qty:
            price_candidate = round(amount / qty, 2)
            amount_candidate = round(qty * price, 2)
            if (round(price_candidate * qty, 2) == round(amount, 2)
                    and _is_dropped_character_misread(product.get("unit_price"), price_candidate)):
                fix = ("unit_price", _money_str(price_candidate))
            elif _is_dropped_character_misread(product.get("amount"), amount_candidate):
                fix = ("amount", _money_str(amount_candidate))

        if fix:
            field, value = fix
            repairs.append(f"line {line_num} ({product.get('item')}): {field} {product.get(field)!r} -> {value!r}")
            product[field] = value
            product.setdefault("repaired_fields", []).append(field)
    return repairs


# Product lines that are really additional charges: some invoices print freight/
# handling as a row in the parts table (Part Number "FREIGHT", Description
# "SPECIAL ITEM") instead of a separate label under the subtotal, so the
# extractor returns it as a product and total_freight/handling_fee stay empty.
_FREIGHT_LINE_WORDS = {"freight", "frt", "freight charge", "freight charges", "shipping", "shipping charge", "delivery charge"}
_HANDLING_LINE_WORDS = {"handling", "handling fee", "handling charge", "shipping and handling", "shipping handling"}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _charge_field_for_line(product: dict) -> str | None:
    for key in ("item", "supplier_item_number", "description"):
        text = _NON_ALNUM_RE.sub(" ", str(product.get(key) or "").lower()).strip()
        if text in _FREIGHT_LINE_WORDS:
            return "total_freight"
        if text in _HANDLING_LINE_WORDS:
            return "handling_fee"
    return None


def move_charge_lines(extracted: dict) -> list[dict]:
    """Moves freight/handling rows out of extracted["products"] into
    total_freight/handling_fee (added onto any amount already there), in
    place. Returns the moved lines, for logging."""
    kept, moved = [], []
    for product in extracted.get("products", []):
        field = _charge_field_for_line(product)
        amount = _to_float(product.get("amount"))
        if field is None or amount is None:
            kept.append(product)
            continue
        existing = _to_float(extracted.get(field)) or 0.0
        extracted[field] = f"{existing + amount:.2f}"
        extracted.setdefault("confidence", {})[field] = (product.get("confidence") or {}).get("amount")
        moved.append({"field": field, "item": product.get("item"), "amount": amount})
    extracted["products"] = kept
    return moved


def looks_like_invoice(raw_response: dict) -> bool:
    """True unless Document AI found none of the three required top-level
    fields (_REQUIRED_TOP_LEVEL_TYPES) at all - not merely low-confidence on
    one, since a present-but-shaky field is still much stronger evidence this
    is a real invoice than a signature block, logo, or T&Cs/W9 document
    rendered as its own PDF, where Document AI typically detects none of
    them. Deliberately conservative (any one present is enough) so a genuine
    invoice with one troublesome field is never misrouted - see graph.py's
    extract_fields, which uses this (not find_extraction_issues' full,
    stricter list) to decide whether to stop before ever calling JDE."""
    entities = raw_response.get("document", {}).get("entities", [])
    present = {e.get("type") for e in entities if e.get("type") != "products"}
    return bool(present & _REQUIRED_TOP_LEVEL_TYPES)


def parse_entities(raw_response: dict) -> dict:
    """Converts this processor's structured document.entities into the same
    canonical field shape gemini_extract.parse_response()'s {"output": ...}
    used to produce - see the module docstring and the field-mapping notes
    above _PRODUCT_ENTITY_TO_FIELD for the specific renames/derivations.

    Also attaches a "confidence" dict (field name -> DocAI's 0-1 confidence)
    at the top level and on every product line, alongside the field values
    themselves - rides through state["extracted"] / pdf_fields unchanged
    since nothing downstream reads more than the specific field keys it
    already expects."""
    entities = raw_response.get("document", {}).get("entities", [])

    output: dict = dict(_MISSING_TOP_LEVEL_DEFAULTS)
    # Field name -> DocAI confidence (0-1) for whatever field it actually
    # extracted. A field left at its _MISSING_*_DEFAULTS value (DocAI never
    # returned that entity at all) has no key here - keep that distinct from
    # "extracted at 0.0 confidence" for the dashboard to render differently.
    output["confidence"] = {}
    products = []

    dropped_subtotal_rows = 0
    for entity in entities:
        etype = entity.get("type")
        if etype == "products":
            properties = entity.get("properties", [])
            if _is_subtotal_artifact({prop.get("type") for prop in properties}):
                dropped_subtotal_rows += 1
                continue
            product = dict(_MISSING_PRODUCT_DEFAULTS)
            product["confidence"] = {}
            for prop in properties:
                field = _PRODUCT_ENTITY_TO_FIELD.get(prop.get("type"))
                if not field:
                    continue
                text = _clean_text(prop.get("mentionText"))
                product[field] = _clean_money(text) if field in _PRODUCT_MONEY_FIELDS else text
                product["confidence"][field] = prop.get("confidence")
            product.setdefault("description", "")
            products.append(product)
            continue

        text = _clean_text(entity.get("mentionText"))
        if etype in _TOP_LEVEL_MONEY_FIELDS:
            output[etype] = _clean_money(text)
            output["confidence"][etype] = entity.get("confidence")
            if etype == "total_amount_due":
                output["currency"] = _leading_symbol(text)
        elif etype in _TOP_LEVEL_TEXT_FIELDS:
            output[etype] = text
            output["confidence"][etype] = entity.get("confidence")

    output["products"] = products
    if products:
        output["purchase_order"] = products[0].get("purchase_order_number") or ""

    logger.info(
        "document_ai.parse_entities: parsed OK | fields=%s products=%d dropped_subtotal_rows=%d",
        list(output.keys()), len(products), dropped_subtotal_rows,
    )
    return {"output": output}
