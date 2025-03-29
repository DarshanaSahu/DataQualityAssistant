#!/usr/bin/env python
"""
Migration script to convert existing rule_config from single expectation format to 
multi-expectation format (array of expectations).

Usage:
    python migrate_rules.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.db.session import engine
    from app.models.rule import Rule, RuleVersion
except ImportError:
    # If direct import fails, set up connection using environment variable
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    engine = create_engine(database_url)

def migrate_rules():
    """Convert rule_config from single expectation format to multi-expectation format."""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all rules
        rules = session.query(Rule).all()
        
        print(f"Found {len(rules)} rules to check...")
        migrated_count = 0
        error_count = 0
        skipped_count = 0
        
        # Process each rule
        for rule in rules:
            try:
                # Skip rules that already have a list in rule_config
                if isinstance(rule.rule_config, list):
                    skipped_count += 1
                    continue
                
                print(f"Migrating rule ID {rule.id}: {rule.name}")
                
                # Convert to list format
                if rule.rule_config is None:
                    # Empty rule_config (unusual case)
                    rule.rule_config = []
                else:
                    # Normal case - convert dict to list with single item
                    rule.rule_config = [{
                        "expectation_type": rule.rule_config.get("expectation_type", ""),
                        "kwargs": rule.rule_config.get("kwargs", {})
                    }]
                
                # Also update the versions if any
                for version in rule.versions:
                    if not isinstance(version.rule_config, list):
                        version.rule_config = [{
                            "expectation_type": version.rule_config.get("expectation_type", ""),
                            "kwargs": version.rule_config.get("kwargs", {})
                        }]
                
                migrated_count += 1
                
            except Exception as e:
                print(f"Error migrating rule ID {rule.id}: {str(e)}")
                error_count += 1
        
        # Commit the changes
        session.commit()
        
        print("\nMigration complete!")
        print(f"- Total rules checked: {len(rules)}")
        print(f"- Rules migrated: {migrated_count}")
        print(f"- Rules already in new format: {skipped_count}")
        print(f"- Rules with errors: {error_count}")
        
    except Exception as e:
        session.rollback()
        print(f"ERROR: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    migrate_rules() 