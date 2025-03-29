import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
print(f"Database URL: {DATABASE_URL}")

def list_tables():
    """List all tables in the database."""
    try:
        # Parse the connection string
        conn_parts = DATABASE_URL.replace("postgresql://", "").split("/")
        db_name = conn_parts[1]
        auth_host = conn_parts[0].split("@")
        
        host_port = auth_host[1] if len(auth_host) > 1 else auth_host[0]
        
        # Handle host:port format
        if ":" in host_port:
            host_parts = host_port.split(":")
            host = host_parts[0]
            port = host_parts[1]
        else:
            host = host_port
            port = "5432"  # Default PostgreSQL port
        
        if len(auth_host) > 1:
            auth = auth_host[0].split(":")
            user = auth[0]
            password = auth[1] if len(auth) > 1 else ""
        else:
            user = ""
            password = ""
        
        print(f"Connecting to database: {db_name} on {host}:{port} with user {user}")
        
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        cursor = conn.cursor()
        
        # Query to get a list of all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        
        print("Tables in the database:")
        for table in tables:
            print(f"- {table[0]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    list_tables() 