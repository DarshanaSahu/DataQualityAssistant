import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.expectations.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.data_context.ephemeral_data_context import EphemeralDataContext
from great_expectations.data_context.types.base import DataContextConfig, InMemoryStoreBackendDefaults
from great_expectations.datasource.fluent.interfaces import Datasource as GXDatasource
from great_expectations.execution_engine import PandasExecutionEngine
import great_expectations as ge
from sqlalchemy import create_engine, text
from app.core.config import settings
from typing import List, Dict, Any
import json
from datetime import datetime
import os
import ruamel.yaml
import uuid
import time

class DataQualityEngine:
    def __init__(self):
        try:
            self.engine = create_engine(settings.DATABASE_URL)
        except Exception as e:
            print(f"Warning: Could not connect to database: {str(e)}")
            self.engine = None
        self._initialize_great_expectations()

    def _initialize_great_expectations(self):
        """Initialize Great Expectations context."""
        # Create a DataContext with in-memory storage
        store_backend_defaults = InMemoryStoreBackendDefaults()
        
        data_context_config = DataContextConfig(
            config_version=3.0,
            expectations_store_name=store_backend_defaults.expectations_store_name,
            validation_results_store_name=store_backend_defaults.validation_results_store_name,
            checkpoint_store_name=store_backend_defaults.checkpoint_store_name,
            stores=store_backend_defaults.stores,
            data_docs_sites={},
            analytics_enabled=False,
            fluent_datasources={
                "pandas_datasource": {
                    "class_name": "PandasDatasource",
                    "module_name": "great_expectations.datasource.fluent",
                    "execution_engine": {
                        "class_name": "PandasExecutionEngine",
                        "module_name": "great_expectations.execution_engine"
                    },
                    "data_connectors": {
                        "runtime_data_connector": {
                            "class_name": "RuntimeDataConnector",
                            "batch_identifiers": ["default_identifier_name"]
                        }
                    }
                }
            }
        )
            
        # Create an EphemeralDataContext
        try:
            self.context = EphemeralDataContext(
                project_config=data_context_config
            )
            print("Great Expectations context initialized successfully")
        except Exception as e:
            print(f"Error initializing context: {str(e)}")
            # As a fallback, try to get the global context
            try:
                self.context = ge.get_context()
                print("Using global Great Expectations context")
            except Exception as e2:
                print(f"Failed to get global context: {str(e2)}")
                raise

    def get_context(self):
        """Get a valid Great Expectations context or create a new one."""
        # Try to use existing context
        try:
            # Check if context is valid by calling a method
            self.context.list_expectation_suite_names()
            return self.context
        except Exception as e:
            print(f"Context invalid, reinitializing: {str(e)}")
            # Reinitialize the context
            for attempt in range(3):  # Try 3 times
                try:
                    self._initialize_great_expectations()
                    return self.context
                except Exception as e:
                    print(f"Failed to initialize context (attempt {attempt+1}): {str(e)}")
                    time.sleep(0.5)  # Wait a bit before retrying
            
            # Last resort: try to get global context
            try:
                return ge.get_context()
            except:
                raise RuntimeError("Failed to create or get a Great Expectations context")

    def execute_rules(self, table_name: str, rule_ids: List[int]) -> Dict[str, Any]:
        """Execute specified rules against the table."""
        results = []
        total_start_time = datetime.now()
        
        # Get rules from database
        with self.engine.connect() as connection:
            # Verify table exists
            table_exists = connection.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                )
            """), {"table_name": table_name}).scalar()
            
            if not table_exists:
                raise ValueError(f"Table '{table_name}' does not exist in the database")
            
            # Get rules - explicitly converting rule_config to JSON string
            query = text("""
                SELECT id, name, rule_config::text as rule_config
                FROM rules
                WHERE id = ANY(:rule_ids)
                AND is_active = true
            """)
            rules = connection.execute(query, {"rule_ids": rule_ids}).fetchall()
            
            if not rules:
                raise ValueError("No active rules found with the specified IDs")
            
            # Get table data
            data_query = text("SELECT * FROM " + table_name)
            df = pd.read_sql(data_query, connection)
            
            # Execute each rule
            for rule in rules:
                rule_start_time = datetime.now()
                rule_result = self._execute_single_rule(df, rule)
                rule_result["execution_time"] = (datetime.now() - rule_start_time).total_seconds()
                results.append(rule_result)
        
        # Calculate overall statistics
        total_rules = len(results)
        successful_rules = sum(1 for r in results if r["success"])
        failed_rules = total_rules - successful_rules
        
        return {
            "table_name": table_name,
            "execution_time": datetime.now().isoformat(),
            "total_duration": (datetime.now() - total_start_time).total_seconds(),
            "total_rules": total_rules,
            "successful_rules": successful_rules,
            "failed_rules": failed_rules,
            "success_rate": (successful_rules / total_rules * 100) if total_rules > 0 else 0,
            "results": results
        }

    def _execute_single_rule(self, df: pd.DataFrame, rule: Any) -> Dict[str, Any]:
        """Execute a single rule against the dataframe."""
        try:
            # Make sure rule_config is a list
            # Convert from SQLAlchemy result to Python object if needed
            if hasattr(rule, '_asdict'):
                rule_dict = rule._asdict()
                rule_id = rule_dict.get('id')
                rule_name = rule_dict.get('name')
                rule_configs = rule_dict.get('rule_config')
            else:
                rule_id = rule.id
                rule_name = rule.name
                rule_configs = rule.rule_config
            
            # If rule_configs is a string (JSON), parse it
            if isinstance(rule_configs, str):
                rule_configs = json.loads(rule_configs)
                
            # Make sure it's a list
            if not isinstance(rule_configs, list):
                rule_configs = [rule_configs]
                
            # ----- NEW DIRECT APPROACH WITHOUT VALIDATOR -----
            # Process each expectation manually
            validation_results = []
            rule_success = True
            
            # Define a simple wrapper to evaluate common expectations
            def evaluate_expectation(expectation_type, kwargs):
                try:
                    # Get the column to validate
                    column = kwargs.get('column')
                    
                    # HANDLE SPECIFIC EXPECTATION TYPES
                    if expectation_type == 'expect_column_values_to_not_be_null':
                        # Check for null values
                        null_count = df[column].isnull().sum()
                        success = null_count == 0
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": len(df),
                                "unexpected_count": null_count,
                                "unexpected_percent": (null_count / len(df) * 100) if len(df) > 0 else 0
                            }
                        }
                        
                    elif expectation_type == 'expect_column_values_to_be_in_set':
                        # Check if values are in the specified set
                        value_set = kwargs.get('value_set', [])
                        unexpected_values = df[df[column].notnull() & ~df[column].isin(value_set)][column]
                        unexpected_count = len(unexpected_values)
                        success = unexpected_count == 0
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": len(df),
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": (unexpected_count / len(df) * 100) if len(df) > 0 else 0,
                                "unexpected_values": unexpected_values.head(10).tolist() if not success else []
                            }
                        }
                    
                    elif expectation_type == 'expect_column_values_to_be_between':
                        # Check if values are between min_value and max_value
                        min_value = kwargs.get('min_value')
                        max_value = kwargs.get('max_value')
                        
                        # Handle cases where min or max might be None
                        if min_value is not None and max_value is not None:
                            unexpected_values = df[(df[column] < min_value) | (df[column] > max_value)][column]
                        elif min_value is not None:
                            unexpected_values = df[df[column] < min_value][column]
                        elif max_value is not None:
                            unexpected_values = df[df[column] > max_value][column]
                        else:
                            unexpected_values = pd.Series([])  # No constraints
                            
                        unexpected_count = len(unexpected_values)
                        success = unexpected_count == 0
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": len(df),
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": (unexpected_count / len(df) * 100) if len(df) > 0 else 0,
                                "unexpected_values": unexpected_values.head(10).tolist() if not success else []
                            }
                        }
                        
                    elif expectation_type == 'expect_column_values_to_match_regex':
                        # Check if values match the regex
                        import re
                        regex = kwargs.get('regex')
                        pattern = re.compile(regex)
                        
                        # Apply the regex check to non-null values
                        mask = df[column].notnull()
                        unexpected_values = df[mask & ~df[column].astype(str).str.match(pattern)][column]
                        unexpected_count = len(unexpected_values)
                        success = unexpected_count == 0
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": len(df),
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": (unexpected_count / len(df) * 100) if len(df) > 0 else 0,
                                "unexpected_values": unexpected_values.head(10).tolist() if not success else []
                            }
                        }
                        
                    elif expectation_type == 'expect_column_values_to_be_unique':
                        # Check for duplicate values
                        value_counts = df[column].value_counts()
                        duplicates = value_counts[value_counts > 1].index.tolist()
                        duplicate_count = sum(value_counts[value_counts > 1] - 1)
                        success = duplicate_count == 0
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": len(df),
                                "unexpected_count": duplicate_count,
                                "unexpected_percent": (duplicate_count / len(df) * 100) if len(df) > 0 else 0,
                                "unexpected_values": duplicates[:10] if not success else []
                            }
                        }
                        
                    # Add more expectation types as needed
                    
                    else:
                        # Default for unimplemented expectation types
                        return {
                            "expectation_type": expectation_type,
                            "success": False,
                            "error": f"Expectation type '{expectation_type}' not implemented in direct evaluation mode"
                        }
                        
                except Exception as e:
                    return {
                        "expectation_type": expectation_type,
                        "success": False,
                        "error": str(e)
                    }
            
            # Process each expectation
            for expectation_config in rule_configs:
                if isinstance(expectation_config, dict):
                    expectation_type = expectation_config.get("expectation_type")
                    kwargs = expectation_config.get("kwargs", {})
                else:
                    expectation_type = expectation_config.expectation_type
                    kwargs = expectation_config.kwargs
                
                # Evaluate this expectation
                result = evaluate_expectation(expectation_type, kwargs)
                validation_results.append(result)
                
                if not result.get("success", False):
                    rule_success = False
            
            return {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "success": rule_success,
                "statistics": {
                    "evaluated_expectations": len(validation_results),
                    "successful_expectations": sum(1 for r in validation_results if r.get("success", False)),
                    "unsuccessful_expectations": sum(1 for r in validation_results if not r.get("success", False)),
                    "total_rows": len(df)
                },
                "results": validation_results
            }
            
        except Exception as e:
            import traceback
            print(f"Error executing rule: {str(e)}")
            print(traceback.format_exc())
            rule_id = getattr(rule, 'id', 'unknown')
            rule_name = getattr(rule, 'name', 'unknown')
            return {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "success": False,
                "error": str(e)
            }

    def generate_report(self, execution_results: Dict[str, Any], output_path: str):
        """Generate Excel report from execution results."""
        # Create overall summary sheet
        overall_summary = [{
            "Table Name": execution_results["table_name"],
            "Execution Time": execution_results["execution_time"],
            "Duration (seconds)": execution_results["total_duration"],
            "Total Rules": execution_results["total_rules"],
            "Successful Rules": execution_results["successful_rules"],
            "Failed Rules": execution_results["failed_rules"],
            "Success Rate (%)": execution_results["success_rate"]
        }]
        
        # Create rules summary sheet
        rules_summary = []
        for result in execution_results["results"]:
            rules_summary.append({
                "Rule ID": result["rule_id"],
                "Rule Name": result["rule_name"],
                "Status": "Success" if result["success"] else "Failed",
                "Error": result.get("error", ""),
                "Execution Time (seconds)": result.get("execution_time", 0),
                "Evaluated Expectations": result.get("statistics", {}).get("evaluated_expectations", 0),
                "Successful Expectations": result.get("statistics", {}).get("successful_expectations", 0),
                "Unsuccessful Expectations": result.get("statistics", {}).get("unsuccessful_expectations", 0),
                "Total Rows": result.get("statistics", {}).get("total_rows", 0)
            })
        
        # Create detailed results sheet
        detailed_data = []
        for result in execution_results["results"]:
            if "results" in result:
                for validation_result in result["results"]:
                    # Extract common fields
                    data = {
                        "Rule ID": result["rule_id"],
                        "Rule Name": result["rule_name"],
                        "Expectation Type": validation_result.get("expectation_type", ""),
                        "Success": validation_result.get("success", False),
                    }
                    
                    # Add specific result details based on what's available
                    for key, value in validation_result.items():
                        if key not in ["expectation_type", "success"]:
                            # Format lists and dicts as JSON strings
                            if isinstance(value, (list, dict)):
                                value = json.dumps(value)
                            data[key] = value
                    
                    detailed_data.append(data)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(overall_summary).to_excel(writer, sheet_name="Overall Summary", index=False)
            pd.DataFrame(rules_summary).to_excel(writer, sheet_name="Rules Summary", index=False)
            
            if detailed_data:
                pd.DataFrame(detailed_data).to_excel(writer, sheet_name="Detailed Results", index=False)
            else:
                # Create an empty sheet if no detailed data
                pd.DataFrame().to_excel(writer, sheet_name="Detailed Results", index=False) 