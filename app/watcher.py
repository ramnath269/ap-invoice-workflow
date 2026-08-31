"""Watches WATCH_FOLDER for new files and runs the graph once per file.

Ports n8n's "Local File Trigger1" (folder watch, ignoring processed_files,
awaitWriteFinish) and "Set Start Time1" (execution_start_ms + file_name)
nodes.
"""
import logging
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .graph import build_graph
from .settings import settings, validate

logger = logging.getLogger("ap_invoice_workflow")


class _NewFileHandler(FileSystemEventHandler):
    def __init__(self, graph):
        self._graph = graph
        self._processed_dir = os.path.join(settings.WATCH_FOLDER, settings.PROCESSED_SUBFOLDER)

    def on_created(self, event):
        if event.is_directory or event.src_path.startswith(self._processed_dir):
            return
        self._await_write_finish(event.src_path)
        self._run(event.src_path)

    @staticmethod
    def _await_write_finish(path: str, poll_interval: float = 1.0, stable_checks_required: int = 2) -> None:
        last_size = -1
        stable_checks = 0
        while stable_checks < stable_checks_required:
            if not os.path.exists(path):
                return
            size = os.path.getsize(path)
            stable_checks = stable_checks + 1 if size == last_size else 0
            last_size = size
            time.sleep(poll_interval)

    def _run(self, file_path: str) -> None:
        state = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "execution_start_ms": time.time() * 1000,
        }
        logger.info("Processing %s", file_path)
        try:
            result = self._graph.invoke(state)
            logger.info("Finished %s with status=%s", file_path, result.get("status"))
        except Exception:
            logger.exception("Workflow failed for %s", file_path)


def watch() -> None:
    logging.basicConfig(level=logging.INFO)
    validate()

    os.makedirs(os.path.join(settings.WATCH_FOLDER, settings.PROCESSED_SUBFOLDER), exist_ok=True)

    graph = build_graph()
    handler = _NewFileHandler(graph)
    observer = Observer()
    observer.schedule(handler, settings.WATCH_FOLDER, recursive=False)
    observer.start()
    logger.info("Watching %s for new invoices...", settings.WATCH_FOLDER)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    watch()
