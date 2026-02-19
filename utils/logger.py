"""
Structured Logging Utility
Provides centralized logging for the NIDS system.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    compression: str = "zip"
) -> None:
    """Configure structured logging."""
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
               "- <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    logger.info(f"Logging initialized at {log_level} level")


def log_security_event(
    event_type: str,
    severity: str,
    source_ip: str,
    destination_ip: Optional[str] = None,
    details: Optional[dict] = None
) -> None:
    """Log security-related events."""
    logger.bind(
        event_type=event_type,
        severity=severity,
        source_ip=source_ip,
        destination_ip=destination_ip,
        details=details or {}
    ).warning(f"Security Event: {event_type}")


def log_performance(
    operation: str,
    duration_ms: float,
    records_processed: int = 0
) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms "
        f"({records_processed} records)"
    )