"""Shared logging setup for the two long-running entry points (main.py's
folder watcher, email_main.py's IMAP watcher).

Previously each called logging.basicConfig(level=logging.INFO) with no
filename, which only attaches a StreamHandler - every logger.info/warning
call in the app went to that process's stdout/stderr and nowhere else, so
closing the terminal that started it lost the history. This adds a
RotatingFileHandler alongside the console output so logs survive on disk.

A production deployment running these under systemd already gets logs via
journald (journalctl -u <service> - see .gitignore's *.log comment) and
doesn't need this; it's for grepping logs directly and for runs outside
systemd (local dev, a plain nohup, etc). Each service gets its own log file
rather than sharing one, since the two entry points are separate OS
processes and Python's RotatingFileHandler doesn't coordinate rotation
safely across processes writing to the same file.
"""
import logging
import logging.handlers
import os

from .settings import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(service_name: str) -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_path = os.path.join(settings.LOG_DIR, f"{service_name}.log")
    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=settings.LOG_MAX_BYTES, backupCount=settings.LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.getLogger("ap_invoice_workflow").info(
        "logging_config: writing %s logs to %s (max_bytes=%d backups=%d)",
        service_name, log_path, settings.LOG_MAX_BYTES, settings.LOG_BACKUP_COUNT,
    )
