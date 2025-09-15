"""
Logging utility following SOLID principles.
Single Responsibility: Handle all logging configuration and setup.
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


class LoggerSetup:
    """Logger configuration and setup class."""
    
    @staticmethod
    def configure_logger() -> None:
        """Configure loguru logger with appropriate settings."""
        # Remove default logger
        logger.remove()
        
        # Add console logging
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Add file logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "pdf_qa_agent.log",
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )


def get_logger(name: str):
    """Get a logger instance for the given module name."""
    return logger.bind(name=name)


# Initialize logger on import
LoggerSetup.configure_logger()
