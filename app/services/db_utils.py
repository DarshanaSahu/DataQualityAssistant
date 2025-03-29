"""
Database utility functions for working with database tables and schemas.
"""

import logging
from sqlalchemy import text
from app.db.session import engine

logger = logging.getLogger(__name__)

def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        table_name: The name of the table to check
        
    Returns:
        bool: True if the table exists, False otherwise
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                )
            """), {"table_name": table_name})
            
            return result.scalar()
    except Exception as e:
        logger.exception(f"Error checking if table {table_name} exists: {str(e)}")
        # Default to False if an error occurs
        return False

def get_column_names(table_name: str) -> list:
    """
    Get a list of column names for a table.
    
    Args:
        table_name: The name of the table
        
    Returns:
        list: A list of column names
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})
            
            return [row[0] for row in result]
    except Exception as e:
        logger.exception(f"Error getting column names for table {table_name}: {str(e)}")
        return []

def get_column_data_types(table_name: str) -> dict:
    """
    Get a dictionary of column names to data types for a table.
    
    Args:
        table_name: The name of the table
        
    Returns:
        dict: A dictionary mapping column names to their data types
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})
            
            return {row[0]: row[1] for row in result}
    except Exception as e:
        logger.exception(f"Error getting column data types for table {table_name}: {str(e)}")
        return {} 