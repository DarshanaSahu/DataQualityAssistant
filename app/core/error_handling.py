"""
Error handling utilities for Data Quality Assistant API.
Provides standardized error handling with user-friendly messages.
"""

import logging
from fastapi import HTTPException
from typing import Dict, Any, Optional, Type, Union
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

logger = logging.getLogger(__name__)

# Error message templates for common error scenarios
ERROR_MESSAGES = {
    # Database errors
    "table_not_found": "Table '{table_name}' does not exist in the database. Please check the table name and try again.",
    "database_connection": "Could not connect to the database. Please verify database connection settings and try again later.",
    "database_query": "An error occurred while querying the database. Please check your query and try again.",
    
    # Resource errors
    "rule_not_found": "Rule with ID {rule_id} was not found. It may have been deleted or moved.",
    "resource_not_found": "The requested {resource_type} could not be found.",
    "invalid_request": "Invalid request: {detail}",
    
    # API errors
    "internal_error": "An internal server error occurred. Our team has been notified and is working to resolve the issue.",
    "ai_service_error": "The AI rule generation service is temporarily unavailable. Please try again later.",
    "validation_error": "Validation failed: {detail}",
    
    # Authentication errors
    "authentication_error": "Authentication failed. Please verify your credentials and try again.",
    "permission_error": "You do not have permission to {action}.",
    
    # Data format errors
    "invalid_json": "The provided data is not a valid JSON format. Please check your request body.",
    "parsing_error": "Could not parse {data_type} data: {detail}",
    
    # Rule generation errors
    "rule_generation_error": "Could not generate rules for table {table_name}: {detail}",
    "rule_execution_error": "Error executing rule(s): {detail}",
    "rule_update_error": "Could not update rule: {detail}",
    
    # Query errors
    "query_timeout": "The database query timed out. Try simplifying your query or filtering for fewer records.",
    "sampling_error": "Could not analyze data sample: {detail}"
}

def get_error_message(error_key: str, **kwargs) -> str:
    """Get a formatted error message for a specific error type."""
    if error_key not in ERROR_MESSAGES:
        return f"An unexpected error occurred: {kwargs.get('detail', '')}"
    
    try:
        return ERROR_MESSAGES[error_key].format(**kwargs)
    except KeyError:
        logger.error(f"Missing required parameter for error message template '{error_key}'")
        return ERROR_MESSAGES.get("internal_error", "An internal server error occurred.")
    except Exception as e:
        logger.error(f"Error formatting error message for '{error_key}': {str(e)}")
        return ERROR_MESSAGES.get("internal_error", "An internal server error occurred.")

def handle_api_error(
    error: Exception, 
    status_code: int = 500, 
    error_key: str = "internal_error", 
    log_error: bool = True,
    **kwargs
) -> HTTPException:
    """
    Handle API errors with appropriate status codes and user-friendly messages.
    
    Args:
        error: The exception that was raised
        status_code: HTTP status code to return
        error_key: Key for error message template in ERROR_MESSAGES
        log_error: Whether to log the error
        **kwargs: Additional parameters for error message formatting
    
    Returns:
        HTTPException with appropriate status code and error message
    """
    if "detail" not in kwargs:
        kwargs["detail"] = str(error)
    
    error_message = get_error_message(error_key, **kwargs)
    
    if log_error:
        logger.error(f"{error_key}: {kwargs.get('detail', str(error))}")
    
    return HTTPException(status_code=status_code, detail=error_message)

def handle_database_error(error: SQLAlchemyError, **kwargs) -> HTTPException:
    """Handle database-related errors with appropriate messages."""
    if isinstance(error, IntegrityError):
        return handle_api_error(
            error, 
            status_code=400, 
            error_key="validation_error", 
            log_error=True,
            **kwargs
        )
    elif isinstance(error, OperationalError):
        return handle_api_error(
            error, 
            status_code=503, 
            error_key="database_connection", 
            log_error=True,
            **kwargs
        )
    else:
        return handle_api_error(
            error, 
            status_code=500, 
            error_key="database_query", 
            log_error=True,
            **kwargs
        )

def format_error_response(error: Union[str, Exception], error_key: str = None, **kwargs) -> Dict[str, Any]:
    """
    Format an error response for returning as JSON directly.
    
    Args:
        error: The error message or exception
        error_key: Optional key for error message template
        **kwargs: Additional parameters for error message formatting
    
    Returns:
        Dict with error details for JSON response
    """
    error_str = str(error)
    if "detail" not in kwargs:
        kwargs["detail"] = error_str
    
    if error_key:
        message = get_error_message(error_key, **kwargs)
    else:
        message = error_str
    
    return {
        "success": False,
        "error": message,
        "error_details": error_str
    } 