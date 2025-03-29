import os
import psycopg2
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

def create_failing_rule():
    """Create rules that will fail validation."""
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
        
        # First, check if authors table has a country column
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'authors'
            ORDER BY ordinal_position
        """)
        
        columns = [row[0] for row in cursor.fetchall()]
        print(f"Columns in authors table: {columns}")
        
        # Create rules that will fail
        rules = [
            {
                "name": "Test Failing Rule - Names",
                "description": "This rule should fail to test sample rows with name",
                "rule_config": {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "name",
                        "value_set": ["John Smith", "Jane Doe"],
                        "mostly": 1.0
                    }
                }
            },
            {
                "name": "Test Failing Rule - Email Format",
                "description": "This rule checks email has @ symbol",
                "rule_config": {
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {
                        "column": "email",
                        "regex": "^.+@.+\\..+$",
                        "mostly": 1.0
                    }
                }
            }
        ]
        
        rule_ids = []
        
        for rule in rules:
            # Insert the rule
            cursor.execute("""
                INSERT INTO rules (name, description, table_name, rule_config, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                rule["name"],
                rule["description"],
                "authors",
                json.dumps(rule["rule_config"]),
                True,
                datetime.now(),
                datetime.now()
            ))
            
            rule_id = cursor.fetchone()[0]
            conn.commit()
            
            print(f"Created rule with ID: {rule_id}")
            print(f"Rule config: {rule['rule_config']}")
            
            # Also create a table version/snapshot of this rule
            cursor.execute("""
                INSERT INTO rule_versions (rule_id, version_number, rule_config, created_at)
                VALUES (%s, %s, %s, %s)
            """, (
                rule_id,
                1,
                json.dumps(rule["rule_config"]),
                datetime.now()
            ))
            
            conn.commit()
            rule_ids.append(rule_id)
        
        cursor.close()
        conn.close()
        
        print(f"Use these rule IDs for testing: {rule_ids}")
        return rule_ids
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    create_failing_rule() 