import psycopg2
from app.core.config import settings

def execute_direct_sql():
    """Execute SQL directly using psycopg2 without SQLAlchemy ORM."""
    # Extract connection parameters from the DATABASE_URL
    db_url = settings.DATABASE_URL
    db_parts = db_url.replace("postgresql://", "").split("@")
    user_pass, host_db = db_parts
    username, password = user_pass.split(":")
    host_port, dbname = host_db.split("/")
    host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")
    
    print(f"Connecting to database {dbname} on {host}:{port} as {username}")
    
    # Connect directly with psycopg2
    conn = psycopg2.connect(
        dbname=dbname,
        user=username,
        password=password,
        host=host,
        port=port
    )
    
    # Set autocommit to True to ensure changes are committed
    conn.autocommit = True
    
    try:
        # Create a cursor
        cur = conn.cursor()
        
        # Check for existing columns
        print("Checking for existing columns...")
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'rules'
            AND column_name IN ('is_draft', 'confidence');
        """)
        existing_cols = [row[0] for row in cur.fetchall()]
        print(f"Found existing columns: {existing_cols}")
        
        # Add is_draft column if it doesn't exist
        if 'is_draft' not in existing_cols:
            print("Adding is_draft column...")
            cur.execute("""
                ALTER TABLE rules
                ADD COLUMN is_draft BOOLEAN NOT NULL DEFAULT FALSE;
            """)
            print("is_draft column added successfully!")
        
        # Add confidence column if it doesn't exist
        if 'confidence' not in existing_cols:
            print("Adding confidence column...")
            cur.execute("""
                ALTER TABLE rules
                ADD COLUMN confidence INTEGER NULL;
            """)
            print("confidence column added successfully!")
        
        # Verify the columns were added
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'rules'
            ORDER BY ordinal_position;
        """)
        all_columns = [row[0] for row in cur.fetchall()]
        print(f"\nAll columns in rules table: {all_columns}")
        
        # Close cursor
        cur.close()
        
    finally:
        # Close connection
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    execute_direct_sql() 