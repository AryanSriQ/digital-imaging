import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logging(log_level="INFO"):
    """
    Sets up structured JSON logging for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get root logger to configure all logging
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid conflicts
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)
    
    # Create file handler with JSON formatting
    file_handler = logging.FileHandler('processing_log.json')
    file_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(file_handler)
    
    # Return a specific logger for this application
    logger = logging.getLogger('dicom_processor')
    
    return logger