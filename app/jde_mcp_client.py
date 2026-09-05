"""Minimal MCP client for the jde-mcp-aoc-test server's jde_po_status tool.

Used as a fallback when JDE's voucher-match orchestrator can't resolve a
supplier's item number: fetches the PO's full line list so item_resolution
can fuzzy-match by quantity + unit price instead.

Authenticates via the server's manual POST /auth/login route (see
jde-mcp-aoc-test/auth/auth_routes.py), not the OAuth authorization_code
flow the server also exposes - that flow requires a human completing a
PKCE-protected browser consent step and only supports authorization_code /
refresh_token grants (no client_credentials), which a headless service like
this can't complete on its own. The manual route takes the same
JDE_USERNAME/JDE_PASSWORD already used for voucher-match and hands back a
bearer token directly - confirmed live against the real server before
building this. Tradeoff: that token is short-lived (~15 min) and carries no
refresh token, so this module re-logs-in whenever its cached token goes
stale rather than refreshing one.
"""
import itertools
import json
import logging
import time

import requests

from .settings import settings

logger = logging.getLogger("ap_invoice_workflow")

_MATCH_TOLERANCE = 0.01
_TOKEN_LIFETIME_SECONDS = 10 * 60  # real TTL is ~15 min; refresh before that

_token: str | None = None
_token_expiry: float = 0.0
_session_id: str | None = None
_next_request_id = itertools.count(2)  # id=1 is used by initialize


def _login() -> str:
    global _token, _token_expiry, _session_id
    resp = requests.post(
        f"{settings.JDE_MCP_BASE_URL}/auth/login",
        json={"username": settings.JDE_USERNAME, "password": settings.JDE_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token = data["token"]
    _token_expiry = time.monotonic() + _TOKEN_LIFETIME_SECONDS
    _session_id = None  # a new token needs a fresh MCP session too
    logger.info("jde_mcp_client: logged in as %s", data.get("username"))
    return _token


def _ensure_token() -> str:
    if _token is None or time.monotonic() >= _token_expiry:
        return _login()
    return _token


def _parse_mcp_response(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    return json.loads(text)


def _post_mcp(token: str, body: dict, session_id: str | None = None) -> tuple[dict, str | None]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {token}",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    resp = requests.post(f"{settings.JDE_MCP_BASE_URL}/mcp", headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return _parse_mcp_response(resp.text), resp.headers.get("Mcp-Session-Id", session_id)


def _ensure_session(token: str) -> str:
    global _session_id
    if _session_id:
        return _session_id

    result, session_id = _post_mcp(
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ap-invoice-workflow", "version": "0.1"},
            },
        },
    )
    if "error" in result:
        raise RuntimeError(f"MCP initialize failed: {result['error']}")
    if not session_id:
        raise RuntimeError("MCP server did not return an Mcp-Session-Id on initialize")

    _post_mcp(token, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=session_id)
    _session_id = session_id
    logger.info("jde_mcp_client: MCP session initialized")
    return session_id


def call_tool(name: str, arguments: dict) -> dict:
    """Calls an MCP tool, re-authenticating once on a 401/403 before giving up."""
    token = _ensure_token()
    session_id = _ensure_session(token)

    def _call(tok: str, sess: str) -> dict:
        result, _ = _post_mcp(
            tok,
            {
                "jsonrpc": "2.0",
                "id": next(_next_request_id),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            session_id=sess,
        )
        return result

    try:
        result = _call(token, session_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            logger.warning("jde_mcp_client: token/session rejected, re-authenticating")
            token = _login()
            session_id = _ensure_session(token)
            result = _call(token, session_id)
        else:
            raise

    if "error" in result:
        raise RuntimeError(f"MCP tool '{name}' failed: {result['error']}")

    tool_result = result.get("result", {})
    if tool_result.get("isError"):
        raise RuntimeError(f"MCP tool '{name}' returned an error: {tool_result}")
    return tool_result


def get_po_lines(order_number, order_type: str) -> list[dict]:
    """Returns every F4311 line for a PO via jde_po_status, or [] if none."""
    tool_result = call_tool(
        "jde_po_status", {"order_number": int(order_number), "order_type": order_type}
    )
    content = tool_result.get("content") or []
    if not content:
        return []

    text = content[0].get("text", "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.info("jde_mcp_client.get_po_lines: non-JSON response (no lines?): %s", text[:200])
        return []

    lines = parsed.get("data") if isinstance(parsed, dict) else None
    return lines or []


def find_matching_po_line(po_lines: list[dict], quantity: float, unit_price: float) -> dict | None:
    """Fuzzy-matches a PDF line item to exactly one PO line by quantity and
    unit price (derived as ExtendedCost / OrderedQuantity, since jde_po_status
    doesn't return a unit-price column directly). Returns None on zero or
    ambiguous (multiple) matches - only a confident, single match counts.

    jde_po_status's query selects PDECST (Extended Cost) raw from F4311,
    which JDE stores scaled x100 - confirmed empirically against a real
    invoice (PO 352890 line 200: ExtendedCost=887400, OrderedQuantity=1800
    -> 493.0 unscaled vs the PDF's real unit_price of 4.93; /100 gives
    exactly 4.93). Comparing the raw scaled value against a real-dollar
    unit_price would silently fail to match almost everything."""
    _EXTENDED_COST_SCALE = 100
    matches = []
    for line in po_lines:
        ordered_qty = line.get("OrderedQuantity") or 0
        extended_cost = (line.get("ExtendedCost") or 0) / _EXTENDED_COST_SCALE
        if not ordered_qty:
            continue
        line_unit_price = extended_cost / ordered_qty
        if abs(ordered_qty - quantity) <= _MATCH_TOLERANCE and abs(line_unit_price - unit_price) <= _MATCH_TOLERANCE:
            matches.append(line)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "jde_mcp_client.find_matching_po_line: %d ambiguous matches for quantity=%s unit_price=%s",
            len(matches), quantity, unit_price,
        )
    return None
