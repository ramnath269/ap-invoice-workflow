"""Shared Google Cloud auth for the raw REST calls to Document AI and Vertex
AI, mirroring the "googleOAuth2Api" predefined credential used by n8n's
HTTP Request nodes. Uses Application Default Credentials - see .env.example.
"""
import google.auth
import google.auth.transport.requests

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_access_token() -> str:
    credentials, _ = google.auth.default(scopes=_SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token
