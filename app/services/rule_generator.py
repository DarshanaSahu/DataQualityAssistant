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
            # Convert Row objects to dictionaries properly
            columns = [{"column_name": row[0], "data_type": row[1], "is_nullable": row[2], "column_default": row[3]} for row in result]
            
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

    def generate_rule_from_description(self, table_name: str, rule_description: str) -> List[Dict[str, Any]]:
        """Generate a rule from natural language description."""
        schema_info = self.analyze_table_schema(table_name)
        
        # Even simpler format - direct instructional prompt with fixed structure
        prompt = """You are a data quality expert. Generate a single Great Expectations rule based on this description.

Table: {table}
Columns: {columns}
Sample Data: {sample_data}
Rule Description: "{description}"

Your task is to:
1. Identify what column and validation the rule applies to
2. Choose the most appropriate expectation type
3. Return ONLY a JSON object, nothing else

ONLY return this JSON structure:
{{
  "expectation_type": "expect_column_values_to_not_be_null",
  "kwargs": {{ 
    "column": "column_name" 
  }},
  "confidence": 90
}}

Set confidence (0-100) based on how confident you are in this rule. Respond only when the rule is above 90% confident.
Include ONLY required parameters in kwargs.
DO NOT add any text before or after the JSON.
"""
        
        # Format the prompt safely with actual values
        formatted_prompt = prompt.format(
            table=table_name,
            columns=schema_info['columns'],
            sample_data=schema_info['sample_data'][:5],
            description=rule_description
        )
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.5, # Reduced temperature for more consistent results
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]
        )
        
        # Parse and validate the generated rule
        try:
            import json
            import re
            
            # Extract JSON from the response
            content = response.content[0].text.strip()
            print(f"Claude response: {content}")  # Debug output
            
            try:
                # First try direct JSON parsing
                rule_data = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, use regex to extract the JSON object
                pattern = r'(\{[\s\S]*\})'
                matches = re.findall(pattern, content)
                
                if not matches:
                    raise ValueError("No JSON found in Claude's response")
                    
                # Try to parse each match until we find a valid one
                rule_data = None
                
                for match in matches:
                    try:
                        rule_data = json.loads(match)
                        # Verify it's a valid rule structure
                        if "expectation_type" in rule_data and "kwargs" in rule_data:
                            break
                    except json.JSONDecodeError:
                        continue
                        
                if not rule_data:
                    raise ValueError("No valid rule configuration found")
            
            # Create a clean rule structure
            rule = {
                "expectation_type": rule_data.get("expectation_type", ""),
                "kwargs": rule_data.get("kwargs", {}),
                "confidence": rule_data.get("confidence", 0)
            }
                
            return [rule]
            
        except Exception as e:
            print(f"Error parsing rule: {str(e)}")
            print(f"Original response: {response.content[0].text}")
            # Create a fallback rule
            fallback_rule = self._create_fallback_rule(rule_description, schema_info)
            return [fallback_rule]
            
    def _create_fallback_rule(self, rule_description: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback rule based on simple analysis of the rule description."""
        rule_description_lower = rule_description.lower()
        columns = schema_info['columns']
        
        # Try to identify the most mentioned column
        mentioned_column = None
        for col in columns:
            if col["column_name"].lower() in rule_description_lower:
                mentioned_column = col["column_name"]
                break
        
        # If no column mentioned, use the first column
        if not mentioned_column and columns:
            mentioned_column = columns[0]["column_name"]
        
        # Use heuristics to guess the appropriate rule type
        if any(term in rule_description_lower for term in ['unique', 'duplicate']):
            return {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": mentioned_column},
                "confidence": 50
            }
        elif any(term in rule_description_lower for term in ['null', 'missing', 'empty', 'required']):
            return {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": mentioned_column, "mostly": 0.95},
                "confidence": 60
            }
        elif any(term in rule_description_lower for term in ['between', 'range', 'minimum', 'maximum']):
            # Try to extract numbers from the description
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', rule_description)
            if len(numbers) >= 2:
                return {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": mentioned_column,
                        "min_value": float(numbers[0]),
                        "max_value": float(numbers[1])
                    },
                    "confidence": 55
                }
            else:
                return {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": mentioned_column},
                    "confidence": 40
                }
        elif any(term in rule_description_lower for term in ['match', 'pattern', 'regex', 'expression']):
            return {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"column": mentioned_column, "regex": ".*"},
                "confidence": 45
            }
        elif any(term in rule_description_lower for term in ['in set', 'value', 'allowed', 'list', 'valid']):
            return {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": mentioned_column, "value_set": []},
                "confidence": 45
            }
        else:
            # Default fallback
            return {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": mentioned_column, "mostly": 0.95},
                "confidence": 30
            } 