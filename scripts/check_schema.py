from sqlalchemy import create_engine, inspect, text
from app.core.config import settings
from app.models.rule import Rule, RuleVersion

def check_schema():
    """Check if the database schema matches the SQLAlchemy models."""
    engine = create_engine(settings.DATABASE_URL)
    inspector = inspect(engine)
    
    # Print database and connection info
    with engine.connect() as connection:
        result = connection.execute(text("SELECT current_database();"))
        db_name = result.scalar()
        print(f"Connected to database: {db_name}")
    
    # Check Rule table columns
    rule_columns = inspector.get_columns('rules')
    rule_column_names = [col['name'] for col in rule_columns]
    print("\n--- Rules Table ---")
    print(f"Found columns: {rule_column_names}")
    
    expected_rule_columns = ['id', 'name', 'description', 'table_name', 'rule_config', 
                           'is_active', 'is_draft', 'confidence', 'created_at', 'updated_at']
    
    for col_name in expected_rule_columns:
        if col_name in rule_column_names:
            print(f"✅ Column '{col_name}' exists in rules table")
        else:
            print(f"❌ Column '{col_name}' is MISSING from rules table")
    
    # Check RuleVersion table columns
    rule_version_columns = inspector.get_columns('rule_versions')
    rule_version_column_names = [col['name'] for col in rule_version_columns]
    print("\n--- Rule Versions Table ---")
    print(f"Found columns: {rule_version_column_names}")
    
    expected_version_columns = ['id', 'rule_id', 'version_number', 'rule_config', 
                              'is_current', 'created_at']
    
    for col_name in expected_version_columns:
        if col_name in rule_version_column_names:
            print(f"✅ Column '{col_name}' exists in rule_versions table")
        else:
            print(f"❌ Column '{col_name}' is MISSING from rule_versions table")
    
    # Additional direct check using SQL
    print("\n--- Direct SQL Query Check ---")
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = 'rules'
            AND column_name IN ('is_draft', 'confidence');
        """))
        sql_columns = [row[0] for row in result]
        print(f"SQL query found columns: {sql_columns}")

if __name__ == "__main__":
    check_schema() 