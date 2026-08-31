"""Central configuration, loaded from environment variables (.env supported).

Replaces the credentials and hardcoded URLs baked directly into the n8n
workflow JSON (in particular the JDE username/password that were hardcoded
in plaintext in the "Code in JavaScript2" node).
"""
import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["JDE_USERNAME", "JDE_PASSWORD"]


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

    # HTTP Request5 / HTTP Request6
    INVOICE_SERVER_BASE_URL = os.environ.get(
        "INVOICE_SERVER_BASE_URL", "https://aoctest.webine3.com/invoice-server"
    )

    # Schema Fields1 / Array fields1 (now a static config file, per project decision)
    SCHEMA_FIELDS_CONFIG = os.environ.get(
        "SCHEMA_FIELDS_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "config", "schema_fields.json"),
    )


settings = Settings()


def validate() -> None:
    missing = [name for name in REQUIRED_VARS if not getattr(settings, name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
