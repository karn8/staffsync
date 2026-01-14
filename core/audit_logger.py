# core/audit_logger.py
import logging
from datetime import datetime

#Function to maintain audit logs
class AuditLogger:
    def __init__(self, logfile="logs/audit.log"):
        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            format="%(asctime)s | %(message)s"
        )

    def log(self, action: str):
        logging.info(action)
