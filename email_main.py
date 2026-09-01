"""Entry point: polls an IMAP mailbox for new invoice emails and drops PDF
attachments into WATCH_FOLDER for the existing folder watcher to process.
"""
from app.email_watcher import watch

if __name__ == "__main__":
    watch()
