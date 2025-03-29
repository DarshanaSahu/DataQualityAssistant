import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

def check_table_schema(table_name):
    """Check the schema of a specified table."""
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
        
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        cursor = conn.cursor()
        
        # Query to get the columns of the table
        cursor.execute(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        print(f"Schema for table '{table_name}':")
        for column in columns:
            print(f"- {column[0]}: {column[1]} (Nullable: {column[2]})")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_table_schema("rule_versions") 