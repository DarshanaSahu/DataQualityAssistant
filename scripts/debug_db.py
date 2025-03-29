from sqlalchemy import create_engine, text
from app.core.config import settings

def debug_database():
    """Debug database connection and print schema information."""
    print(f"Connecting to database URL: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as connection:
        # Get database name
        result = connection.execute(text("SELECT current_database();"))
        db_name = result.scalar()
        print(f"Connected to database: {db_name}")
        
        # Get all tables
        result = connection.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """))
        tables = [row[0] for row in result]
        print(f"Tables in database: {tables}")
        
        # Get rules table columns
        if 'rules' in tables:
            result = connection.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = 'rules'
                ORDER BY ordinal_position;
            """))
            columns = [(row[0], row[1], row[2]) for row in result]
            print(f"\nColumns in rules table:")
            for col in columns:
                print(f"  - {col[0]} ({col[1]}, nullable: {col[2]})")
        else:
            print("Rules table not found!")
            
        # Try to add columns with explicit transaction
        trans = connection.begin()
        try:
            print("\nAttempting to add is_draft column with explicit transaction...")
            connection.execute(text("""
                ALTER TABLE rules 
                ADD COLUMN is_draft BOOLEAN NOT NULL DEFAULT FALSE;
            """))
            
            print("\nAttempting to add confidence column...")
            connection.execute(text("""
                ALTER TABLE rules 
                ADD COLUMN confidence INTEGER NULL;
            """))
            
            # Commit the transaction
            trans.commit()
            print("Transaction committed successfully!")
            
        except Exception as e:
            trans.rollback()
            print(f"Error executing ALTER TABLE: {str(e)}")
            
        # Verify columns were added after transaction
        result = connection.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = 'rules'
            ORDER BY ordinal_position;
        """))
        columns = [(row[0], row[1], row[2]) for row in result]
        print(f"\nColumns in rules table after migration:")
        for col in columns:
            print(f"  - {col[0]} ({col[1]}, nullable: {col[2]})")

if __name__ == "__main__":
    debug_database() 