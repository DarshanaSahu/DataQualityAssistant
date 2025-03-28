import pandas as pd
from sqlalchemy import create_engine, text
import anthropic
from app.core.config import settings
from typing import List, Dict, Any

class AIRuleGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.engine = create_engine(settings.DATABASE_URL)

    def analyze_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Analyze table schema and return column information."""
        with self.engine.connect() as connection:
            # Get column information
            query = text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position;
            """)
            result = connection.execute(query, {"table_name": table_name})
            columns = [dict(row) for row in result]
            
            # Get sample data
            sample_query = text(f"SELECT * FROM {table_name} LIMIT 100")
            sample_data = pd.read_sql(sample_query, connection)
            
            return {
                "columns": columns,
                "sample_data": sample_data.to_dict(orient="records")
            }

    def generate_rules(self, table_name: str) -> List[Dict[str, Any]]:
        """Generate Great Expectations rules using Claude."""
        schema_info = self.analyze_table_schema(table_name)
        
        # Prepare prompt for Claude
        prompt = f"""You are a data quality expert specializing in Great Expectations rules generation.
        Analyze the following PostgreSQL table schema and sample data to generate appropriate Great Expectations rules.
        Focus on data quality aspects like:
        - Completeness (null checks)
        - Data types and formats
        - Value ranges and distributions
        - Uniqueness
        - Referential integrity
        
        Table: {table_name}
        Columns: {schema_info['columns']}
        Sample Data: {schema_info['sample_data'][:5]}  # First 5 rows
        
        Generate rules in Great Expectations format. Each rule should be a dictionary with the following structure:
        {{
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {{
                "column": "column_name",
                "mostly": 0.95
            }}
        }}
        
        Return as a list of such rule configurations. Make sure the response is valid JSON that can be parsed by Python's json.loads() function.
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse and validate the generated rules
        try:
            import json
            rules = json.loads(response.content[0].text)
            if not isinstance(rules, list):
                rules = [rules]
            
            # Validate and format each rule
            formatted_rules = []
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                    
                # Ensure the rule has the required structure
                formatted_rule = {
                    "expectation_type": rule.get("expectation_type", ""),
                    "kwargs": rule.get("kwargs", {})
                }
                
                # Only add valid rules
                if formatted_rule["expectation_type"] and formatted_rule["kwargs"]:
                    formatted_rules.append(formatted_rule)
            
            return formatted_rules if formatted_rules else [{
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": schema_info['columns'][0]['column_name'],
                    "mostly": 0.95
                }
            }]
            
        except Exception as e:
            print(f"Error parsing rules: {str(e)}")
            # Return a default rule if parsing fails
            return [{
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": schema_info['columns'][0]['column_name'],
                    "mostly": 0.95
                }
            }]

    def check_rule_outdated(self, rule_id: int, table_name: str) -> Dict[str, Any]:
        """Check if a rule is outdated based on current data patterns."""
        # Get current rule configuration
        with self.engine.connect() as connection:
            query = text("""
                SELECT rule_config FROM rules WHERE id = :rule_id
            """)
            result = connection.execute(query, {"rule_id": rule_id})
            current_rule = result.fetchone()
            
            if not current_rule:
                raise ValueError("Rule not found")
            
            # Analyze current data patterns
            schema_info = self.analyze_table_schema(table_name)
            
            # Generate new rules for comparison
            new_rules = self.generate_rules(table_name)
            
            # Compare and identify outdated aspects
            prompt = f"""You are a data quality expert analyzing rule configurations.
            Compare the following rule configurations and identify if the current rule is outdated:
            
            Current Rule: {current_rule[0]}
            New Generated Rules: {new_rules}
            
            Analyze if the current rule needs updates based on:
            1. Data type changes
            2. Value range changes
            3. New patterns in the data
            4. Missing validations
            
            Return a JSON with:
            - is_outdated: boolean
            - outdated_aspects: list of strings
            - recommended_updates: list of strings
            
            Make sure the response is valid JSON that can be parsed by Python's eval() function.
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return eval(response.content[0].text) 