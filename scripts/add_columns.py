from sqlalchemy import create_engine, text
from app.core.config import settings

def run_migration():
    """Add is_draft and confidence columns to the rules table if they don't exist."""
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as connection:
        # Check if columns already exist
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'rules' 
            AND column_name IN ('is_draft', 'confidence');
        """)
        result = connection.execute(check_query)
        existing_columns = [row[0] for row in result]
        
        # Add is_draft column if it doesn't exist
        if 'is_draft' not in existing_columns:
            print("Adding is_draft column to rules table...")
            connection.execute(text("""
                ALTER TABLE rules 
                ADD COLUMN is_draft BOOLEAN NOT NULL DEFAULT FALSE;
            """))
            print("is_draft column added successfully!")
        else:
            print("is_draft column already exists.")
        
        # Add confidence column if it doesn't exist
        if 'confidence' not in existing_columns:
            print("Adding confidence column to rules table...")
            connection.execute(text("""
                ALTER TABLE rules 
                ADD COLUMN confidence INTEGER NULL;
            """))
            print("confidence column added successfully!")
        else:
            print("confidence column already exists.")

if __name__ == "__main__":
    run_migration() 