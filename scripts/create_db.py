import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.core.config import settings

def create_database():
    # Connect to PostgreSQL server
    engine = create_engine(settings.DATABASE_URL.rsplit('/', 1)[0])
    
    # Get database name from URL
    db_name = settings.DATABASE_URL.split('/')[-1]
    
    try:
        # Check if database exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            exists = result.scalar() is not None
            
            if not exists:
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"Database {db_name} created successfully!")
            else:
                print(f"Database {db_name} already exists.")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    create_database() 