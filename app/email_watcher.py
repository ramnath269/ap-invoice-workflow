"""Polls an IMAP mailbox for new invoice emails and drops PDF attachments
into WATCH_FOLDER, where the existing folder watcher (app/watcher.py) picks
them up and runs them through the graph - this file's only job is
"unseen email with a PDF" -> "file in WATCH_FOLDER", so graph invocation
stays in one place regardless of which trigger fired.

Dedup strategy: search for UNSEEN messages and mark each one \\Seen once its
attachment is saved. Simple and needs no extra mailbox folders, but note
this repurposes the mailbox's read/unread state for automation - if a human
reads an invoice email in a normal mail client before this polls, it will
never be picked up.
"""
import email
import imaplib
import logging
import os
import time
from email.header import decode_header

from .settings import settings, validate_imap

logger = logging.getLogger("ap_invoice_workflow")


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for text, encoding in decode_header(value):
        if isinstance(text, bytes):
            parts.append(text.decode(encoding or "utf-8", errors="ignore"))
        else:
            parts.append(text)
    return "".join(parts)


def _unique_destination(directory: str, filename: str) -> str:
    destination = os.path.join(directory, filename)
    base, ext = os.path.splitext(destination)
    counter = 1
    while os.path.exists(destination):
        destination = f"{base}_{counter}{ext}"
        counter += 1
    return destination


def _save_attachment(payload: bytes, filename: str) -> str:
    os.makedirs(settings.WATCH_FOLDER, exist_ok=True)
    destination = _unique_destination(settings.WATCH_FOLDER, filename)
    with open(destination, "wb") as f:
        f.write(payload)
    logger.info("email_watcher: saved attachment (%d bytes) to %s", len(payload), destination)
    return destination


def _connect() -> imaplib.IMAP4:
    if settings.IMAP_USE_SSL:
        return imaplib.IMAP4_SSL(settings.IMAP_HOST, settings.IMAP_PORT)
    return imaplib.IMAP4(settings.IMAP_HOST, settings.IMAP_PORT)


def poll_once() -> int:
    """Runs one IMAP poll cycle. Returns the number of PDF attachments saved."""
    conn = _connect()
    saved_count = 0
    try:
        conn.login(settings.IMAP_USERNAME, settings.IMAP_PASSWORD)
        status, _ = conn.select(settings.IMAP_FOLDER)
        if status != "OK":
            raise RuntimeError(f"Could not select IMAP folder {settings.IMAP_FOLDER!r}")

        status, data = conn.search(None, "UNSEEN")
        if status != "OK":
            raise RuntimeError("IMAP UNSEEN search failed")
        uids = data[0].split()
        logger.info("email_watcher: %d unseen message(s) in %s", len(uids), settings.IMAP_FOLDER)

        for uid in uids:
            status, msg_data = conn.fetch(uid, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                logger.warning("email_watcher: failed to fetch uid=%s", uid.decode())
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject = _decode_header_value(msg.get("Subject"))
            sender = msg.get("From", "")

            found_pdf = False
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                filename = _decode_header_value(part.get_filename())
                is_pdf = part.get_content_type() == "application/pdf" or filename.lower().endswith(
                    ".pdf"
                )
                if not filename or not is_pdf:
                    continue
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                found_pdf = True
                logger.info(
                    "email_watcher: uid=%s from=%s subject=%r attachment=%s (%d bytes)",
                    uid.decode(),
                    sender,
                    subject,
                    filename,
                    len(payload),
                )
                _save_attachment(payload, filename)
                saved_count += 1

            if not found_pdf:
                logger.info(
                    "email_watcher: uid=%s from=%s subject=%r has no PDF attachment, skipping",
                    uid.decode(),
                    sender,
                    subject,
                )

            conn.store(uid, "+FLAGS", "\\Seen")
    finally:
        try:
            conn.logout()
        except Exception:
            pass
    return saved_count


def watch() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_imap()
    logger.info(
        "email_watcher: polling %s@%s folder=%s every %ds",
        settings.IMAP_USERNAME,
        settings.IMAP_HOST,
        settings.IMAP_FOLDER,
        settings.IMAP_POLL_SECONDS,
    )
    while True:
        try:
            poll_once()
        except Exception:
            logger.exception("email_watcher: poll cycle failed")
        time.sleep(settings.IMAP_POLL_SECONDS)


if __name__ == "__main__":
    watch()
