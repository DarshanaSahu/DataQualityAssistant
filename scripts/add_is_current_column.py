from sqlalchemy import create_engine, text
from app.core.config import settings

def add_is_current_column():
    """Add is_current column to the rule_versions table if it doesn't exist."""
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as connection:
        # Check if column already exists
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'rule_versions' 
            AND column_name = 'is_current';
        """)
        result = connection.execute(check_query)
        existing_columns = [row[0] for row in result]
        
        # Add is_current column if it doesn't exist
        if 'is_current' not in existing_columns:
            print("Adding is_current column to rule_versions table...")
            connection.execute(text("""
                ALTER TABLE rule_versions 
                ADD COLUMN is_current BOOLEAN NOT NULL DEFAULT TRUE;
            """))
            print("is_current column added successfully!")
        else:
            print("is_current column already exists.")

if __name__ == "__main__":
    add_is_current_column() 