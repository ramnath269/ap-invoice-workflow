"""Corrects misread item numbers, and empty or garbled descriptions,
against a PDF's own embedded text layer.

Document AI's custom extractor always reads the page image, even when the
PDF carries exact embedded text (processOptions.ocrConfig /
enableNativePdfParsing is rejected for CUSTOM_EXTRACTION_PROCESSOR with
OCR_CONFIG_UNSUPPORTED). On some print styles that misreads characters the
text layer has right - e.g. Federated's dot-matrix font, where 'QX1563' is
OCR'd as 'OX1563' and 'MO81' as 'M081'.

The text layer is used only as a correction source, never as the primary
extraction: an extracted item that isn't found in the text layer is
replaced by a near-identical text-layer token, and only when that token is
the unique closest match on a text-layer row that also carries the line's
own quantity, price and amount; a description is set from the words after
the item on that line's row (see fix_descriptions). Scanned PDFs (no text
layer) are left untouched.
"""
from __future__ import annotations

import io
import logging
import re

from pypdf import PdfReader

logger = logging.getLogger("ap_invoice_workflow")

# Below this many characters of embedded text per page on average, the PDF
# is treated as a scan (an image with at most a stray label or two).
_MIN_TEXT_CHARS_PER_PAGE = 100
# Max edit distance between an extracted item and its text-layer correction
# - covers 1-2 misread characters (Q->O, O->0, 8->3) and a dropped trailing
# character or two, without matching a genuinely different part number.
_MAX_EDIT_DISTANCE = 2
_MONEY_RE = re.compile(r"^\d*\.\d\d$")


def read_text_lines(pdf_bytes: bytes) -> list[str] | None:
    """Returns the PDF's embedded text as layout-preserving lines, or None
    when it has no usable text layer (a scan) or can't be parsed."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    except Exception:
        logger.exception("pdf_text_layer.read_text_lines: could not read the PDF's text layer")
        return None
    if not pages or sum(len(p.strip()) for p in pages) / len(pages) < _MIN_TEXT_CHARS_PER_PAGE:
        return None
    return [line for page in pages for line in page.splitlines() if line.strip()]


def _norm(value) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def _norm_number(value) -> str | None:
    """'19.37' / '.53' / '0.53' / '4' -> a canonical string, or None."""
    text = str(value or "").strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return f"{number:.2f}" if "." in text else str(int(number)) if number.is_integer() else f"{number:.2f}"


def _squash(words: list[str]) -> str:
    return re.sub(r"[\W_]", "", "".join(words))


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _row_candidates(line: str) -> list[str]:
    """Every token on a text-layer row, plus adjacent token pairs for part
    numbers printed with an internal space (e.g. '0955  02')."""
    tokens = line.split()
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def correct_items(products: list[dict], text_lines: list[str]) -> list[str]:
    """Replaces, in place, each product "item" that doesn't appear in the
    text layer with its unique closest text-layer match, found only on rows
    that carry the same amount (and unit_price/quantity, when extracted).
    Returns a description of each correction (for logging); corrected lines
    get "item" added to "repaired_fields"."""
    rows = []
    known = set()
    for line in text_lines:
        candidates = _row_candidates(line)
        known.update(_norm(c) for c in candidates)
        numbers = {n for n in (_norm_number(t) for t in line.split()) if n is not None}
        rows.append((candidates, numbers))

    corrections = []
    for line_num, product in enumerate(products, start=1):
        item = product.get("item") or ""
        if not item or _norm(item) in known:
            continue
        needed = [_norm_number(product.get(k)) for k in ("amount", "unit_price", "quantity")]
        if needed[0] is None:
            continue  # without an amount there's no reliable row anchor
        needed = [n for n in needed if n is not None]

        best: dict[str, int] = {}
        for candidates, numbers in rows:
            if not all(n in numbers for n in needed):
                continue
            for candidate in candidates:
                if _MONEY_RE.match(candidate) or candidate.isdigit() and len(candidate) < 3:
                    continue  # prices/amounts/quantities, not part numbers
                distance = _edit_distance(_norm(item), _norm(candidate))
                if distance <= _MAX_EDIT_DISTANCE:
                    best[candidate] = min(distance, best.get(candidate, distance))

        if not best:
            continue
        closest = min(best.values())
        winners = {_norm(c): c for c, d in best.items() if d == closest}
        if len(winners) != 1:
            logger.info(
                "pdf_text_layer.correct_items: line %d item %r has %d equally close text-layer matches %s - left as-is",
                line_num, item, len(winners), sorted(winners.values()),
            )
            continue
        corrected = next(iter(winners.values()))
        corrections.append(f"line {line_num}: item {item!r} -> {corrected!r}")
        product["item"] = corrected
        product.setdefault("repaired_fields", []).append("item")
    return corrections


def fix_descriptions(products: list[dict], text_lines: list[str]) -> int:
    """Sets, in place, "description" from the line's text-layer row: the
    words following the item number, up to the first number (the
    quantity/price/amount columns). Only when exactly one row carries both
    the line's item and its amount, and then only if the extracted
    description is empty or is a garbled form of the text-layer one - every
    word of it, other than the item number and quantity fused onto it,
    appears in the text-layer description (scrambled 'ROTORS STOP SILENT',
    truncated 'DURAGLOSS', 'CONNECTOR 2'), or it's the same text spaced or
    punctuated differently ('PUMP, WAT'). Words found nowhere in the whole
    text layer are ignored in that check, since they can only have come from
    an image on the page (watermark, stamp). A description with any other
    word the row doesn't have is left alone. Returns the number changed."""
    document_words = {w.upper() for line in text_lines for w in line.split()}
    changed = 0
    for product in products:
        item = _norm(product.get("item"))
        amount = _norm_number(product.get("amount"))
        if not item or amount is None:
            continue
        found = []
        for line in text_lines:
            tokens = line.split()
            if amount not in {_norm_number(t) for t in tokens}:
                continue
            # the item may span two tokens ('0955  02'); take the last match on
            # the row, since some layouts print it twice (ordered/shipped)
            ends = [i + width for width in (1, 2) for i in range(len(tokens) - width + 1)
                    if _norm("".join(tokens[i:i + width])) == item]
            if not ends:
                continue
            words = []
            for token in tokens[max(ends):]:
                # numbers end the description, as do malformed numeric runs and
                # labels from text printed on the same row ('Core-:  28.029.00')
                if _norm_number(token) is not None or re.fullmatch(r"[\d.,\-]+", token) or token.endswith(":"):
                    break
                words.append(token)
            if words:
                found.append(" ".join(words))
        if len(found) != 1 or product.get("description") == found[0]:
            continue
        fused = {item, str(product.get("quantity") or "").upper()}
        current = [w for w in str(product.get("description") or "").upper().split() if w not in fused]
        target = found[0].upper().split()
        # the same text split or punctuated differently ('PUMP, WAT' / 'PUMP,WAT')
        respaced = _squash(current) == _squash(target)
        # a word found nowhere in the document's text layer isn't printed text -
        # it came from an image (watermark, stamp), e.g. 'PROFESSION' from the
        # 'PROFESSIONALS' watermark behind Federated's line items
        printed = {w for w in current if w in document_words}
        if current and not respaced and not printed <= set(target):
            continue
        product["description"] = found[0]
        product.setdefault("repaired_fields", []).append("description")
        changed += 1
    return changed
