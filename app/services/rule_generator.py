import json
import re
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

try:
    # Try importing from the latest Anthropic client version
    from anthropic import Anthropic
    USE_NEW_CLIENT = True
except ImportError:
    # Fall back to the older client if needed
    import anthropic
    USE_NEW_CLIENT = False

from app.core.config import settings
from typing import List, Dict, Any, Optional, Tuple, Union
from app.services import db_utils

logger = logging.getLogger(__name__)

class AIRuleGenerator:
    def __init__(self):
        if USE_NEW_CLIENT:
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
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
            
            # Get a random sample of 100 rows for better performance
            sample_query = text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 100")
            sample_data = pd.read_sql(sample_query, connection)
            
            return {
                "columns": columns,
                "sample_data": sample_data.to_dict(orient="records")
            }

    def generate_rules(self, table_name: str) -> List[Dict[str, Any]]:
        """Generate Great Expectations rules using Claude."""
        schema_info = self.analyze_table_schema(table_name)
        
        # First identify potential column relationships
        multi_column_insights = self._analyze_column_relationships(table_name, schema_info)
        
        # Prepare prompt for Claude with enhanced multi-column support
        prompt = f"""You are a data quality expert specializing in Great Expectations rules generation.
        Analyze the following PostgreSQL table schema and sample data to generate appropriate Great Expectations rules.
        Focus on data quality aspects like:
        - Completeness (null checks)
        - Data types and formats
        - Value ranges and distributions
        - Uniqueness
        - Referential integrity
        - Multi-column relationships and constraints
        
        Table: {table_name}
        Columns: {schema_info['columns']}
        Sample Data: {schema_info['sample_data'][:5]}  # First 5 rows
        
        Multi-Column Relationship Analysis: {multi_column_insights}
        
        Generate rules in Great Expectations format. Include both single-column rules and multi-column rules.
        
        SINGLE-COLUMN RULE EXAMPLE:
        {{
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {{
                "column": "column_name",
                "mostly": 0.95
            }}
        }}
        
        MULTI-COLUMN RULE EXAMPLES:
        
        1. Column Pair Equality:
        {{
            "expectation_type": "expect_column_pair_values_to_be_equal",
            "kwargs": {{
                "column_A": "first_column",
                "column_B": "second_column"
            }}
        }}
        
        2. Column Pair Valid Combinations:
        {{
            "expectation_type": "expect_column_pair_values_to_be_in_set",
            "kwargs": {{
                "column_A": "country",
                "column_B": "currency",
                "value_pairs_set": [["USA", "USD"], ["Canada", "CAD"]]
            }}
        }}
        
        3. Date Column Ordering:
        {{
            "expectation_type": "expect_column_values_to_be_greater_than_other_column",
            "kwargs": {{
                "column": "end_date",
                "compare_to": "start_date"
            }}
        }}
        
        Return as a list of such rule configurations. Include a mix of single-column and multi-column rules as appropriate.
        Make sure the response is valid JSON that can be parsed by Python's json.loads() function.
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
            
            # If no valid rules were found, provide a default rule
            if not formatted_rules:
                # Try to include both a single-column and a multi-column rule if possible
                default_rules = []
                
                # Add a default single-column rule
                if schema_info['columns']:
                    default_rules.append({
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "kwargs": {
                            "column": schema_info['columns'][0]['column_name'],
                            "mostly": 0.95
                        }
                    })
                
                # Add a default multi-column rule if possible
                if len(schema_info['columns']) >= 2:
                    default_rules.append({
                        "expectation_type": "expect_column_pair_values_to_be_in_set",
                        "kwargs": {
                            "column_A": schema_info['columns'][0]['column_name'],
                            "column_B": schema_info['columns'][1]['column_name'],
                            "value_pairs_set": []
                        }
                    })
                
                return default_rules if default_rules else [{
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": schema_info['columns'][0]['column_name'],
                        "mostly": 0.95
                    }
                }]
                
            return formatted_rules
            
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
        
        # Updated prompt that supports multi-column rules
        prompt = """You are a data quality expert. Generate a Great Expectations rule based on this description.

Table: {table}
Columns: {columns}
Sample Data: {sample_data}
Rule Description: "{description}"

Your task is to:
1. Identify what column(s) and validation the rule applies to
2. Choose the most appropriate expectation type
3. Return ONLY a JSON object, nothing else

IMPORTANT: If the rule involves MULTIPLE COLUMNS, use appropriate multi-column expectations 
or return MULTIPLE single-column expectations as an array.

For multi-column rules, use any of these approaches as appropriate:
1. Use "expect_column_pair_values_to_be_in_set" for column pair validations
2. Use "expect_column_pair_values_to_be_equal" for equality checks between columns
3. Use "expect_column_values_to_be_in_set" with dynamic value sets based on another column
4. For complex relationships, return multiple related rules in an array

ONLY return a JSON structure like ONE of these:

SINGLE COLUMN EXPECTATION:
{{
  "expectation_type": "expect_column_values_to_not_be_null",
  "kwargs": {{ 
    "column": "column_name" 
  }},
  "confidence": 90
}}

MULTI-COLUMN EXPECTATION:
{{
  "expectation_type": "expect_column_pair_values_to_be_in_set",
  "kwargs": {{ 
    "column_A": "first_column",
    "column_B": "second_column",
    "value_pairs_set": [["value1_A", "value1_B"], ["value2_A", "value2_B"]]
  }},
  "confidence": 90
}}

MULTI-RULE RESPONSE (for complex validations):
[
  {{
    "expectation_type": "expect_column_values_to_not_be_null",
    "kwargs": {{ "column": "first_column" }},
    "confidence": 90
  }},
  {{
    "expectation_type": "expect_column_values_to_be_in_set",
    "kwargs": {{ 
      "column": "second_column",
      "value_set": ["value1", "value2"]
    }},
    "confidence": 85
  }}
]

Set confidence (0-100) based on how confident you are in this rule. Include ONLY required parameters in kwargs.
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
                pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'  # Match either object or array
                matches = re.findall(pattern, content)
                
                if not matches:
                    raise ValueError("No JSON found in Claude's response")
                    
                # Try to parse each match until we find a valid one
                rule_data = None
                
                for match in matches:
                    try:
                        rule_data = json.loads(match)
                        # Verify it's a valid rule structure
                        if isinstance(rule_data, dict) and "expectation_type" in rule_data and "kwargs" in rule_data:
                            break
                        elif isinstance(rule_data, list) and all("expectation_type" in r and "kwargs" in r for r in rule_data):
                            break
                    except json.JSONDecodeError:
                        continue
                        
                if not rule_data:
                    raise ValueError("No valid rule configuration found")
            
            # Handle both single rule and multiple rules
            if isinstance(rule_data, dict):
                # Single rule
                rule = {
                    "expectation_type": rule_data.get("expectation_type", ""),
                    "kwargs": rule_data.get("kwargs", {}),
                    "confidence": rule_data.get("confidence", 0)
                }
                return [rule]
            elif isinstance(rule_data, list):
                # Multiple rules
                rules = []
                for r in rule_data:
                    rule = {
                        "expectation_type": r.get("expectation_type", ""),
                        "kwargs": r.get("kwargs", {}),
                        "confidence": r.get("confidence", 0)
                    }
                    rules.append(rule)
                return rules
                
            return [rule_data]  # Fallback
            
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
        
        # Try to identify mentioned columns
        mentioned_columns = []
        for col in columns:
            col_name = col["column_name"].lower()
            if col_name in rule_description_lower:
                mentioned_columns.append(col["column_name"])
        
        # If no columns mentioned, use the first column
        if not mentioned_columns and columns:
            mentioned_columns.append(columns[0]["column_name"])
        
        # For multi-column rules
        if len(mentioned_columns) > 1 and any(term in rule_description_lower for term in ['equal', 'same', 'match between', 'comparison', 'equals', 'equivalent']):
            return {
                "expectation_type": "expect_column_pair_values_to_be_equal",
                "kwargs": {
                    "column_A": mentioned_columns[0],
                    "column_B": mentioned_columns[1]
                },
                "confidence": 40
            }
        
        # Use the first mentioned column for single-column rules
        mentioned_column = mentioned_columns[0] if mentioned_columns else columns[0]["column_name"]
        
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

    def scan_table_for_rule_suggestions(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze a table and suggest data quality rules based on patterns found.
        Returns suggested new rules and updates to existing rules.
        """
        logger.info(f"Scanning table {table_name} for rule suggestions")
        
        # Make sure table exists
        if not db_utils.table_exists(table_name):
            logger.error(f"Table {table_name} does not exist")
            return {
                "new_rule_suggestions": [],
                "rule_update_suggestions": [],
                "error": f"Table {table_name} does not exist"
            }
        
        # Analyze the table schema - use a shorter timeout for better performance
        try:
            # Limit schema analysis time for better performance
            logger.info(f"Starting schema analysis for {table_name} with random sampling")
            schema_info = self.analyze_table_schema(table_name)
            logger.info(f"Schema analysis complete for {table_name}")
        except Exception as e:
            logger.exception(f"Error analyzing schema for table {table_name}: {str(e)}")
            return {
                "new_rule_suggestions": [],
                "rule_update_suggestions": [],
                "error": f"Error analyzing schema: {str(e)}"
            }
        
        # Analyze column relationships - limit the analysis to key relationships only
        logger.info(f"Analyzing column relationships for {table_name}")
        multi_column_insights = self._analyze_column_relationships(table_name, schema_info)
        
        # Create the prompt for rule suggestion generation
        prompt = self._create_rule_suggestion_prompt(table_name, schema_info, multi_column_insights)
        
        try:
            # Send the request to Claude with reduced temperature for more consistent results
            logger.info(f"Sending request to Claude for {table_name}")
            
            # If Claude API call fails or returns invalid JSON, use simple rule generation instead
            try:
                if USE_NEW_CLIENT:
                    # Use new Anthropic client with optimized parameters
                    response = self.client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=3000,  # Reduced token count for faster response
                        temperature=0.1,  # Lower temperature for more consistent results
                        system="You are a data quality expert. Analyze the sample data and suggest rules that would likely apply to the full dataset.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                else:
                    # Use older Anthropic client
                    response = self.client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=3000,  # Reduced token count
                        temperature=0.1,  # Lower temperature
                        system="You are a data quality expert. Analyze the sample data and suggest rules that would likely apply to the full dataset.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                
                # Parse the response
                content = ""
                if USE_NEW_CLIENT:
                    content = response.content[0].text.strip()
                else:
                    content = response.content.strip()
                    
                logger.info(f"Received response from Claude (first 200 chars): {content[:200]}...")
            except Exception as api_error:
                logger.error(f"Error calling Claude API: {str(api_error)}")
                logger.info("Falling back to basic rule generation")
                # Generate basic rules instead
                return self._generate_basic_rule_suggestions(table_name, schema_info)
            
            # Attempt to extract JSON from response if not already JSON
            try:
                # Try parsing directly first
                suggestions = json.loads(content)
            except json.JSONDecodeError as decode_error:
                # If direct parsing fails, try to extract JSON using regex
                logger.error(f"JSON decode error at position {decode_error.pos}, line {decode_error.lineno}, column {decode_error.colno}")
                logger.info("Direct JSON parsing failed, trying to extract JSON...")
                
                # First try with code block format
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        suggestions = json.loads(json_str)
                        logger.info("Successfully extracted JSON from code block")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from code block")
                        # Fall back to basic rule generation
                        return self._generate_basic_rule_suggestions(table_name, schema_info)
                else:
                    # If regex extraction fails, look for the first { and last } for a complete JSON object
                    logger.info("No code block found, searching for JSON object boundaries")
                    start_idx = content.find('{')
                    if start_idx == -1:
                        logger.error("No opening brace found in response")
                        # Fall back to basic rule generation
                        return self._generate_basic_rule_suggestions(table_name, schema_info)
                        
                    end_idx = content.rfind('}')
                    if end_idx == -1 or end_idx <= start_idx:
                        logger.error("No closing brace found in response")
                        # Fall back to basic rule generation
                        return self._generate_basic_rule_suggestions(table_name, schema_info)
                        
                    json_str = content[start_idx:end_idx+1]
                    
                    # Try to clean up the JSON
                    # Remove any trailing commas before closing braces or brackets
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    # Fix any single quoted strings if needed
                    json_str = re.sub(r'\'([^\']+)\'', r'"\1"', json_str)
                    
                    logger.info(f"Extracted JSON string (first 200 chars): {json_str[:200]}...")
                    try:
                        suggestions = json.loads(json_str)
                        logger.info("Successfully extracted and parsed JSON object")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted JSON: {e}")
                        # Fall back to basic rule generation
                        return self._generate_basic_rule_suggestions(table_name, schema_info)
            
            # Ensure required keys exist
            if "new_rule_suggestions" not in suggestions:
                suggestions["new_rule_suggestions"] = []
            
            if "rule_update_suggestions" not in suggestions:
                suggestions["rule_update_suggestions"] = []
            
            # Ensure all rule configs are lists
            for rule in suggestions.get("new_rule_suggestions", []):
                if "rule_config" in rule and not isinstance(rule["rule_config"], list):
                    rule["rule_config"] = [rule["rule_config"]]
            
            # Ensure all suggested configs are lists
            for rule in suggestions.get("rule_update_suggestions", []):
                if "suggested_config" in rule and not isinstance(rule["suggested_config"], list):
                    rule["suggested_config"] = [rule["suggested_config"]]
                    
                # Make sure rule_id is an integer
                if "rule_id" in rule and not isinstance(rule["rule_id"], int):
                    try:
                        rule["rule_id"] = int(rule["rule_id"])
                    except (ValueError, TypeError):
                        # If it can't be converted to an integer, remove the suggestion
                        logger.error(f"Invalid rule_id: {rule.get('rule_id')}")
                        suggestions["rule_update_suggestions"].remove(rule)
                        continue
                
                # Handle special case for current_config
                if "current_config" in rule and isinstance(rule["current_config"], str):
                    try:
                        # Try to parse it as JSON
                        config = json.loads(rule["current_config"])
                        if isinstance(config, dict):
                            rule["current_config"] = [config]
                        elif isinstance(config, list):
                            rule["current_config"] = config
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat it as a string description
                        pass
            
            return suggestions
        
        except Exception as e:
            logger.exception(f"Error generating rule suggestions for {table_name}: {str(e)}")
            # Fall back to basic rule generation
            return self._generate_basic_rule_suggestions(table_name, schema_info)

    def _analyze_column_relationships(self, table_name: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between columns in the table.
        
        This helps identify potential multi-column rules like:
        - Foreign key relationships
        - Column pairs that should match or be related
        - Columns with conditional relationships
        """
        try:
            # Limit analysis to most important relationship types for better performance
            column_pairs = []
            columns = [col["column_name"] for col in schema_info["columns"]]
            sample_data = schema_info["sample_data"]
            
            # Look for potential naming pattern relationships 
            # Examples: first_name/last_name, start_date/end_date, etc.
            # These are quick to compute based just on column names
            related_by_name = []
            name_patterns = [
                ("start", "end"), ("begin", "end"), ("first", "last"),
                ("min", "max"), ("source", "target"), ("from", "to")
            ]
            
            for pattern_a, pattern_b in name_patterns:
                for col_a in columns:
                    if pattern_a in col_a.lower():
                        base_name = col_a.lower().replace(pattern_a, "")
                        for col_b in columns:
                            if pattern_b in col_b.lower() and base_name in col_b.lower():
                                related_by_name.append((col_a, col_b))
            
            # Look for potential foreign key relationships based on naming conventions
            # Examples: user_id in a table might reference id in users table
            # Also quick to compute based on column names
            potential_foreign_keys = []
            id_columns = [col for col in columns if col.endswith("_id")]
            
            for id_col in id_columns:
                # The referenced table name might be the prefix of the _id column
                referenced_table = id_col.replace("_id", "")
                potential_foreign_keys.append({
                    "column": id_col,
                    "potential_reference": f"{referenced_table}.id"
                })
            
            # Look for natural date comparisons (e.g., start_date should be before end_date)
            # Quick to compute based on data types and naming patterns
            date_comparisons = []
            date_columns = []
            
            for col in schema_info["columns"]:
                if col["data_type"] in ("date", "timestamp", "timestamp without time zone", "timestamp with time zone"):
                    date_columns.append(col["column_name"])
            
            # Find pairs of date columns that might have a chronological relationship
            for date_pattern_a, date_pattern_b in [("start", "end"), ("begin", "end"), ("from", "to")]:
                for col_a in date_columns:
                    if date_pattern_a in col_a.lower():
                        for col_b in date_columns:
                            if date_pattern_b in col_b.lower() and col_a != col_b:
                                date_comparisons.append({
                                    "column_pair": [col_a, col_b],
                                    "expected_relationship": f"{col_a} should be before {col_b}"
                                })
            
            # Limit the correlated value analysis to a smaller number of column combinations
            # for better performance
            value_correlations = []
            
            # Only process if we have sample data and the number of columns is reasonable
            if sample_data and len(sample_data) > 0 and len(columns) <= 20:
                # Limit to a smaller number of column pairs by prioritizing:
                # 1. Columns with similar names or prefixes
                # 2. Columns with compatible data types
                prioritized_pairs = []
                
                # Find column pairs with similar names
                for i, col_a in enumerate(columns):
                    for col_b in columns[i+1:]:
                        # Check for common prefixes or similar names
                        if (col_a.split('_')[0] == col_b.split('_')[0] or
                            col_a.replace('_', '') in col_b or col_b.replace('_', '') in col_a):
                            prioritized_pairs.append((col_a, col_b))
                
                # Analyze a smaller subset of column pairs (max 10)
                for col_a, col_b in prioritized_pairs[:10]:
                    matching_count = 0
                    total_count = 0
                    
                    for row in sample_data:
                        if col_a in row and col_b in row:
                            val_a = row[col_a]
                            val_b = row[col_b]
                            # Skip null values
                            if val_a is None or val_b is None:
                                continue
                                
                            total_count += 1
                            # Check exact matches or pattern matches
                            if val_a == val_b:
                                matching_count += 1
                            # Check if one value is contained in the other
                            elif isinstance(val_a, str) and isinstance(val_b, str):
                                if val_a in val_b or val_b in val_a:
                                    matching_count += 1
                    
                    if total_count > 0 and matching_count / total_count > 0.5:
                        value_correlations.append({
                            "column_pair": [col_a, col_b],
                            "match_percentage": round(matching_count / total_count * 100, 2)
                        })
            
            return {
                "related_by_name": related_by_name,
                "potential_foreign_keys": potential_foreign_keys,
                "value_correlations": value_correlations,
                "date_comparisons": date_comparisons
            }
        
        except Exception as e:
            print(f"Error analyzing column relationships: {str(e)}")
            return {
                "error": str(e),
                "related_by_name": [],
                "potential_foreign_keys": [],
                "value_correlations": [],
                "date_comparisons": []
            } 

    def _create_rule_suggestion_prompt(self, table_name: str, schema_info: Dict[str, Any], 
                                       multi_column_insights: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for rule suggestion generation.
        
        Args:
            table_name: Name of the table being analyzed
            schema_info: Dictionary with table schema information
            multi_column_insights: Dictionary with multi-column relationship insights
            
        Returns:
            String prompt for Claude to generate rule suggestions
        """
        # Get existing rules for this table
        existing_rules = []
        try:
            with self.engine.connect() as connection:
                query = text("""
                    SELECT id, name, rule_config::text
                    FROM rules
                    WHERE table_name = :table_name
                    AND is_active = true
                """)
                result = connection.execute(query, {"table_name": table_name})
                existing_rules = [
                    {"id": row[0], "name": row[1], "rule_config": row[2]} 
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Error fetching existing rules: {str(e)}")
            # Continue with empty existing rules
        
        # Generate potential new rules based on current data
        potential_new_rules = []
        try:
            potential_new_rules = self.generate_rules(table_name)
        except Exception as e:
            logger.error(f"Error generating potential rules: {str(e)}")
            # Continue with empty potential rules

        # Prepare a complete analysis prompt
        prompt = f"""You are a data quality expert analyzing table data and existing rules.

Table: {table_name}
Columns: {schema_info['columns']}
Sample Data (random 5 rows from a 100 row sample): {schema_info['sample_data'][:5]}

Note: You're analyzing a random sample of 100 rows, not the entire table. Focus on patterns that would likely apply to the full dataset.

Existing Rules: {existing_rules}

Potential New Rules: {potential_new_rules}

Multi-Column Relationship Analysis: {multi_column_insights}

Your task:
1. Analyze the current data patterns within this random sample
2. Compare existing rules with potential new rules
3. Identify gaps in rule coverage or outdated rules
4. Pay special attention to multi-column relationships and constraints

Return a detailed JSON with:

{{
    "new_rule_suggestions": [
        {{
            "rule_type": "expectation_type",
            "column": "column_name",  // For single-column rules
            "columns": ["column1", "column2"],  // For multi-column rules
            "description": "Human readable description of why this rule is suggested",
            "rule_config": {{
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {{ "column": "column_name" }}
            }},
            "confidence": 90
        }}
    ],
    "rule_update_suggestions": [
        {{
            "rule_id": 123,
            "current_config": "existing rule config",
            "suggested_config": {{
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {{ "column": "column_name", "mostly": 0.95 }}
            }},
            "reason": "Human readable explanation of why this update is suggested",
            "confidence": 85
        }}
    ]
}}"""
        return prompt 

    def _generate_basic_rule_suggestions(self, table_name: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate basic rule suggestions without using the Claude API.
        Used as a fallback when the API call fails or returns invalid JSON.
        """
        logger.info(f"Generating basic rule suggestions for {table_name}")
        
        new_rule_suggestions = []
        rule_update_suggestions = []
        
        # Get columns and sample data
        columns = schema_info.get("columns", [])
        sample_data = schema_info.get("sample_data", [])
        
        # Generate simple rule suggestions based on column types
        for column in columns:
            col_name = column.get("column_name")
            data_type = column.get("data_type", "").lower()
            is_nullable = column.get("is_nullable") == "YES"
            
            # Basic not-null rules for important columns
            if not is_nullable or col_name.endswith("_id") or col_name == "id":
                new_rule_suggestions.append({
                    "rule_type": "expect_column_values_to_not_be_null",
                    "column": col_name,
                    "columns": None,
                    "description": f"The {col_name} column should not contain null values.",
                    "rule_config": [{
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "kwargs": {"column": col_name}
                    }],
                    "confidence": 90
                })
            
            # Type-based rules
            if "int" in data_type or data_type in ("bigint", "smallint", "decimal", "numeric"):
                # Check if it's an ID column (likely unique)
                if col_name.endswith("_id") or col_name == "id":
                    new_rule_suggestions.append({
                        "rule_type": "expect_column_values_to_be_unique",
                        "column": col_name,
                        "columns": None,
                        "description": f"The {col_name} column should contain unique values.",
                        "rule_config": [{
                            "expectation_type": "expect_column_values_to_be_unique",
                            "kwargs": {"column": col_name}
                        }],
                        "confidence": 85
                    })
            elif "char" in data_type or "text" in data_type:
                # For email columns
                if "email" in col_name.lower():
                    new_rule_suggestions.append({
                        "rule_type": "expect_column_values_to_match_regex",
                        "column": col_name,
                        "columns": None,
                        "description": f"The {col_name} column should contain valid email addresses.",
                        "rule_config": [{
                            "expectation_type": "expect_column_values_to_match_regex",
                            "kwargs": {
                                "column": col_name, 
                                "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                            }
                        }],
                        "confidence": 80
                    })
            
        return {
            "new_rule_suggestions": new_rule_suggestions,
            "rule_update_suggestions": rule_update_suggestions,
            "error": None
        } 