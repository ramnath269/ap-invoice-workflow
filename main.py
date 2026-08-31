"""Entry point: starts the folder watcher that runs the LangGraph AP-invoice
workflow on every new file. Port of the n8n workflow in
"FTP based WF (1).json".
"""
from app.watcher import watch

if __name__ == "__main__":
    watch()
