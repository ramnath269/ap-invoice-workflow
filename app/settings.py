"""Central configuration, loaded from environment variables (.env supported).

Replaces the credentials and hardcoded URLs baked directly into the n8n
workflow JSON (in particular the JDE username/password that were hardcoded
in plaintext in the "Code in JavaScript2" node).
"""
import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["JDE_USERNAME", "JDE_PASSWORD"]
IMAP_REQUIRED_VARS = ["IMAP_HOST", "IMAP_USERNAME", "IMAP_PASSWORD"]


class Settings:
    WORKFLOW_ID = "ftp-based-wf"
    WORKFLOW_NAME = "FTP based WF"

    # Local File Trigger1
    WATCH_FOLDER = os.environ.get("WATCH_FOLDER", "/home/node/.n8n-files")
    PROCESSED_SUBFOLDER = os.environ.get("PROCESSED_SUBFOLDER", "processed_files")

    # HTTP Request3 (Google Document AI)
    GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "678280534943")
    DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "us")
    DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "da343bda4810bc17")

    # Gemini Vertex AI1
    VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "invoice-ai-platform-496910")
    VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # Guardrails1
    PII_ENTITIES = [e.strip() for e in os.environ.get("PII_ENTITIES", "CREDIT_CARD,EMAIL_ADDRESS").split(",") if e.strip()]

    # Voucher Match
    JDE_ORCHESTRATOR_URL = os.environ.get(
        "JDE_ORCHESTRATOR_URL",
        "http://71.42.117.252:8130/jderest/orchestrator/55_ORCH_PO_ReceiptFile_Inquiry_V2",
    )
    JDE_USERNAME = os.environ.get("JDE_USERNAME")
    JDE_PASSWORD = os.environ.get("JDE_PASSWORD")
    # For local/offline testing when the JDE orchestrator isn't reachable
    # from this machine (e.g. it needs a VPN or an allowlisted network).
    # Never leave this on against a real deployment - it skips the PO
    # receipt-validity check entirely.
    JDE_MOCK_MODE = os.environ.get("JDE_MOCK_MODE", "false").lower() == "true"
    JDE_MOCK_RECORDS = int(os.environ.get("JDE_MOCK_RECORDS", "1"))

    # HTTP Request5 / HTTP Request6
    INVOICE_SERVER_BASE_URL = os.environ.get(
        "INVOICE_SERVER_BASE_URL", "https://aoctest.webine3.com/invoice-server"
    )

    # Schema Fields1 / Array fields1 (now a static config file, per project decision)
    SCHEMA_FIELDS_CONFIG = os.environ.get(
        "SCHEMA_FIELDS_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "config", "schema_fields.json"),
    )

    # Email trigger (IMAP) - polls a mailbox for new invoice emails and drops
    # their PDF attachments into WATCH_FOLDER, where the existing folder
    # watcher picks them up and runs them through the graph. This keeps graph
    # invocation in one place instead of duplicating it per trigger.
    IMAP_HOST = os.environ.get("IMAP_HOST")
    IMAP_PORT = int(os.environ.get("IMAP_PORT", "993"))
    IMAP_USE_SSL = os.environ.get("IMAP_USE_SSL", "true").lower() == "true"
    IMAP_USERNAME = os.environ.get("IMAP_USERNAME")
    IMAP_PASSWORD = os.environ.get("IMAP_PASSWORD")
    IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "INBOX")
    IMAP_POLL_SECONDS = int(os.environ.get("IMAP_POLL_SECONDS", "60"))

    # Item cross-reference fallback: when JDE's voucher-match can't resolve a
    # supplier's item number, jde_mcp_client calls this JDE MCP server's
    # jde_po_status tool (via its manual /auth/login route, not the OAuth
    # flow - that route needs a human in a browser, this one doesn't) to
    # fetch the PO's lines for a quantity/unit-price fuzzy match. Reuses
    # JDE_USERNAME/JDE_PASSWORD above - no separate credential needed.
    #
    # Talks to the backend directly on localhost, not through nginx's public
    # https://aoctest.webine3.com/aoc-test-mcp path - that reverse-proxy
    # rewrites /aoc-test-mcp/mcp incorrectly (nginx's proxy_pass replaces the
    # whole /aoc-test-mcp prefix with /mcp, so requesting .../aoc-test-mcp/mcp
    # 404s upstream at /mcp/mcp - the correct public path has no /mcp suffix
    # at all), and separately gates /aoc-test-mcp/auth/ behind an
    # X-Client-Secret header this service doesn't have. Both projects run on
    # the same host, so localhost sidesteps both issues entirely.
    JDE_MCP_BASE_URL = os.environ.get("JDE_MCP_BASE_URL", "http://127.0.0.1:8006")


settings = Settings()


def validate() -> None:
    missing = [name for name in REQUIRED_VARS if not getattr(settings, name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )


def validate_imap() -> None:
    missing = [name for name in IMAP_REQUIRED_VARS if not getattr(settings, name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set IMAP_HOST / IMAP_USERNAME / IMAP_PASSWORD in .env to enable the email trigger."
        )
