"""OCR via Google Document AI.

Ports n8n's "Code in JavaScript" (base64 encode) and "HTTP Request3"
(Document AI :process call) nodes.
"""
import base64
import mimetypes

import requests

from .google_auth import get_access_token
from .settings import settings


def encode_file(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        content = f.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "application/pdf"
    return base64.b64encode(content).decode("utf-8"), mime_type


def run_ocr(base64_content: str, mime_type: str) -> str:
    url = (
        f"https://us-documentai.googleapis.com/v1/projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{settings.DOCAI_LOCATION}/processors/{settings.DOCAI_PROCESSOR_ID}:process"
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {get_access_token()}"},
        json={"rawDocument": {"content": base64_content, "mimeType": mime_type}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["document"]["text"]
