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
            
            # Helper function to convert sample rows to JSON-serializable format
            def prepare_sample_rows(rows):
                if not rows:
                    return []
                
                result = []
                for row in rows:
                    # Convert each row to a serializable format
                    serializable_row = {}
                    for key, value in row.items():
                        # Handle dates, datetimes and other special types
                        if hasattr(value, 'isoformat'):
                            serializable_row[key] = value.isoformat()
                        elif pd.isna(value):
                            serializable_row[key] = None
                        elif isinstance(value, (pd.Series, pd.DataFrame)):
                            serializable_row[key] = "COMPLEX_DATA"
                        # Handle numpy types (including numpy.bool_)
                        elif str(type(value)).startswith("<class 'numpy."):
                            # Convert numpy types to native Python types
                            serializable_row[key] = value.item() if hasattr(value, 'item') else str(value)
                        else:
                            serializable_row[key] = value
                    result.append(serializable_row)
                return result
            
            # New helper function to get complete row samples in a consistent format
            def get_consistent_sample_rows(failing_mask, column_name, message=None, max_samples=5):
                """Return complete table row samples for failing rows in a consistent format."""
                if failing_mask.sum() == 0:  # No failures
                    return []
                    
                # Get up to 5 complete rows from the dataframe where the validation failed
                sample_rows = df[failing_mask].head(max_samples).to_dict('records')
                
                # Prepare the samples for serialization
                return prepare_sample_rows(sample_rows)
                
            # ----- NEW DIRECT APPROACH WITHOUT VALIDATOR -----
            # Process each expectation manually
            validation_results = []
            rule_success = True
            
            # Define a simple wrapper to evaluate common expectations
            def evaluate_expectation(expectation_type, kwargs):
                try:
                    # Get the column to validate
                    column = kwargs.get('column')
                    
                    # Get the "mostly" parameter if it exists
                    mostly = kwargs.get('mostly', 1.0)  # Default to 1.0 (100% must pass)
                    
                    # HANDLE SPECIFIC EXPECTATION TYPES
                    if expectation_type == 'expect_column_values_to_not_be_null':
                        # Check for null values
                        null_mask = df[column].isnull()
                        null_count = null_mask.sum()
                        element_count = len(df)
                        unexpected_percent = (null_count / element_count * 100) if element_count > 0 else 0
                        
                        # Determine success based on mostly parameter
                        success = unexpected_percent <= (1 - mostly) * 100
                        
                        # Get sample rows with failures - using consistent format
                        sample_rows = []
                        if null_count > 0:  # Always get samples if there are failures, regardless of mostly
                            sample_rows = get_consistent_sample_rows(null_mask, column)
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": element_count,
                                "unexpected_count": null_count,
                                "unexpected_percent": unexpected_percent
                            },
                            "sample_rows": sample_rows,
                            "kwargs": kwargs
                        }
                        
                    elif expectation_type == 'expect_column_values_to_be_in_set':
                        # Check if values are in the specified set
                        value_set = kwargs.get('value_set', [])
                        unexpected_mask = df[column].notnull() & ~df[column].isin(value_set)
                        unexpected_values = df[unexpected_mask][column]
                        unexpected_count = len(unexpected_values)
                        element_count = len(df)
                        unexpected_percent = (unexpected_count / element_count * 100) if element_count > 0 else 0
                        
                        # Determine success based on mostly parameter
                        success = unexpected_percent <= (1 - mostly) * 100
                        
                        # Get sample rows with failures - using consistent format
                        sample_rows = []
                        if unexpected_count > 0:  # Always get samples if there are failures, regardless of mostly
                            sample_rows = get_consistent_sample_rows(unexpected_mask, column)
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": element_count,
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": unexpected_percent,
                                "unexpected_values": unexpected_values.head(10).tolist() if unexpected_count > 0 else []
                            },
                            "sample_rows": sample_rows,
                            "kwargs": kwargs
                        }
                        
                    elif expectation_type == 'expect_column_values_to_be_between':
                        # Check if values are in range
                        min_value = kwargs.get('min_value')
                        max_value = kwargs.get('max_value')
                        
                        # Initialize masks
                        below_min = pd.Series(False, index=df.index)
                        above_max = pd.Series(False, index=df.index)
                        
                        # Check min threshold if specified
                        if min_value is not None:
                            below_min = df[column].notnull() & (df[column] < min_value)
                            
                        # Check max threshold if specified  
                        if max_value is not None:
                            above_max = df[column].notnull() & (df[column] > max_value)
                            
                        # Combine masks for all out-of-range values
                        unexpected_mask = below_min | above_max
                        unexpected_values = df[unexpected_mask][column]
                        unexpected_count = len(unexpected_values)
                        element_count = len(df)
                        unexpected_percent = (unexpected_count / element_count * 100) if element_count > 0 else 0
                        
                        # Determine success based on mostly parameter
                        success = unexpected_percent <= (1 - mostly) * 100
                        
                        # Get sample rows with failures - using consistent format
                        sample_rows = []
                        if unexpected_count > 0:  # Always get samples if there are failures, regardless of mostly
                            sample_rows = get_consistent_sample_rows(unexpected_mask, column)
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": element_count,
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": unexpected_percent,
                                "unexpected_values": unexpected_values.head(10).tolist() if unexpected_count > 0 else []
                            },
                            "sample_rows": sample_rows,
                            "kwargs": kwargs
                        }
                        
                    elif expectation_type == 'expect_column_values_to_match_regex':
                        # Check if values match the regex
                        import re
                        regex = kwargs.get('regex')
                        pattern = re.compile(regex)
                        
                        # Apply the regex check to non-null values
                        mask = df[column].notnull()
                        unexpected_mask = mask & ~df[column].astype(str).str.match(pattern)
                        unexpected_values = df[unexpected_mask][column]
                        unexpected_count = len(unexpected_values)
                        element_count = len(df)
                        unexpected_percent = (unexpected_count / element_count * 100) if element_count > 0 else 0
                        
                        # Determine success based on mostly parameter
                        success = unexpected_percent <= (1 - mostly) * 100
                        
                        # Get sample rows with failures - using consistent format
                        sample_rows = []
                        if unexpected_count > 0:  # Always get samples if there are failures, regardless of mostly
                            sample_rows = get_consistent_sample_rows(unexpected_mask, column)
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": element_count,
                                "unexpected_count": unexpected_count,
                                "unexpected_percent": unexpected_percent,
                                "unexpected_values": unexpected_values.head(10).tolist() if unexpected_count > 0 else []
                            },
                            "sample_rows": sample_rows,
                            "kwargs": kwargs
                        }
                        
                    elif expectation_type == 'expect_column_values_to_be_unique':
                        # Find duplicated values
                        duplicated_mask = df[column].duplicated(keep=False)
                        duplicate_values = df[duplicated_mask][column].unique().tolist()
                        duplicate_count = duplicated_mask.sum() - len(duplicate_values)
                        element_count = len(df)
                        unexpected_percent = (duplicate_count / element_count * 100) if element_count > 0 else 0
                        
                        # Determine success based on mostly parameter
                        success = unexpected_percent <= (1 - mostly) * 100
                        
                        # Get sample rows with failures - using consistent format
                        sample_rows = []
                        if duplicate_count > 0:  # Always get samples if there are failures, regardless of mostly
                            # For uniqueness, we want to include examples of each duplicate value
                            # but still limit to a reasonable number of samples
                            samples = []
                            # Get up to 5 unique values that have duplicates
                            for dup_val in duplicate_values[:5]:
                                # Get up to 2 examples of each duplicate value
                                dup_mask = df[column] == dup_val
                                dup_samples = get_consistent_sample_rows(dup_mask, column, max_samples=2)
                                samples.extend(dup_samples[:2])  # Add up to 2 examples
                                if len(samples) >= 5:  # Cap at 5 total samples
                                    break
                            sample_rows = samples[:5]  # Ensure we don't exceed 5 samples
                        
                        return {
                            "expectation_type": expectation_type,
                            "success": success,
                            "result": {
                                "element_count": element_count,
                                "unexpected_count": duplicate_count,
                                "unexpected_percent": unexpected_percent,
                                "unexpected_values": duplicate_values[:10] if duplicate_count > 0 else []
                            },
                            "sample_rows": sample_rows,
                            "kwargs": kwargs
                        }
                    
                    # Add more expectation types as needed
                    
                    else:
                        # Default for unimplemented expectation types
                        return {
                            "expectation_type": expectation_type,
                            "success": False,
                            "error": f"Expectation type '{expectation_type}' not implemented in direct evaluation mode",
                            "sample_rows": [],
                            "kwargs": kwargs
                        }
                        
                except Exception as e:
                    # Return error details without sample rows
                    return {
                        "expectation_type": expectation_type,
                        "success": False,
                        "error": str(e),
                        "sample_rows": [],
                        "kwargs": kwargs
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
        # Helper function to safely serialize complex objects including dates
        def json_serialize_safe(obj):
            if hasattr(obj, 'isoformat'):  # For datetime and date objects
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif pd.isna(obj):  # Handle NaN and None
                return None
            else:
                return str(obj)  # Convert anything else to string
        
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
                        if key not in ["expectation_type", "success", "sample_rows"]:
                            # Format lists and dicts as JSON strings with safe serialization
                            if isinstance(value, (list, dict)):
                                value = json.dumps(value, default=json_serialize_safe)
                            data[key] = value
                    
                    detailed_data.append(data)
                    
        # Extract sample failed rows for separate sheet
        sample_rows_data = []
        for result in execution_results["results"]:
            if "results" in result:
                for validation_result in result["results"]:
                    if not validation_result.get("success", True) and validation_result.get("sample_rows"):
                        for sample_row in validation_result.get("sample_rows", []):
                            # Convert sample row to a flattened format
                            row_data = {
                                "Rule ID": result["rule_id"],
                                "Rule Name": result["rule_name"],
                                "Expectation Type": validation_result.get("expectation_type", ""),
                            }
                            
                            # Add the row details
                            if isinstance(sample_row, dict):
                                # Add all row values
                                for key, value in sample_row.items():
                                    # Handle nested dictionaries
                                    if isinstance(value, dict):
                                        for sub_key, sub_value in value.items():
                                            row_data[f"{key}_{sub_key}"] = sub_value
                                    else:
                                        row_data[key] = value
                            
                            sample_rows_data.append(row_data)
                                                
        # Create visualization data for charts
        rule_success_data = {
            "Rule Name": [result["rule_name"] for result in execution_results["results"]],
            "Success": [1 if result["success"] else 0 for result in execution_results["results"]]
        }
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Overall Summary Sheet
            summary_df = pd.DataFrame(overall_summary)
            summary_df.to_excel(writer, sheet_name="Overall Summary", index=False)
            
            # Apply formatting to the overall summary sheet
            workbook = writer.book
            summary_sheet = writer.sheets["Overall Summary"]
            
            # Add title
            title_format = workbook.add_format({'bold': True, 'font_size': 14})
            summary_sheet.write(0, 0, f"Data Quality Report for {execution_results['table_name']}", title_format)
            summary_sheet.write(1, 0, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_sheet.write(3, 0, "Overall Summary:")
            
            # Write the summary data with offset
            summary_df.to_excel(writer, sheet_name="Overall Summary", startrow=4, index=False)
            
            # Format headers
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(4, col_num, value, header_format)
            
            # Format the success rate with color coding
            success_rate = execution_results["success_rate"]
            cell_format = workbook.add_format()
            if success_rate >= 90:
                cell_format.set_bg_color('#C6EFCE')  # Green for good
            elif success_rate >= 70:
                cell_format.set_bg_color('#FFEB9C')  # Yellow for warning
            else:
                cell_format.set_bg_color('#FFC7CE')  # Red for bad
            
            # Find the success rate column index
            success_rate_col = summary_df.columns.get_loc("Success Rate (%)")
            summary_sheet.write(5, success_rate_col, success_rate, cell_format)
            
            # Rules Summary Sheet
            rules_df = pd.DataFrame(rules_summary)
            rules_df.to_excel(writer, sheet_name="Rules Summary", index=False)
            
            # Format the Rules Summary sheet
            rules_sheet = writer.sheets["Rules Summary"]
            for col_num, value in enumerate(rules_df.columns.values):
                rules_sheet.write(0, col_num, value, header_format)
                
            # Color code the status column
            status_col = rules_df.columns.get_loc("Status")
            for row_num, status in enumerate(rules_df["Status"]):
                if status == "Success":
                    cell_format = workbook.add_format({'bg_color': '#C6EFCE'})
                else:
                    cell_format = workbook.add_format({'bg_color': '#FFC7CE'})
                rules_sheet.write(row_num + 1, status_col, status, cell_format)
            
            # Detailed Results sheet
            if detailed_data:
                detail_df = pd.DataFrame(detailed_data)
                detail_df.to_excel(writer, sheet_name="Detailed Results", index=False)
                
                # Format the Detailed Results sheet
                detail_sheet = writer.sheets["Detailed Results"]
                for col_num, value in enumerate(detail_df.columns.values):
                    detail_sheet.write(0, col_num, value, header_format)
                
                # Color code the success column
                success_col = detail_df.columns.get_loc("Success")
                for row_num, success in enumerate(detail_df["Success"]):
                    if success:
                        cell_format = workbook.add_format({'bg_color': '#C6EFCE'})
                    else:
                        cell_format = workbook.add_format({'bg_color': '#FFC7CE'})
                    detail_sheet.write(row_num + 1, success_col, success, cell_format)
            else:
                # Create an empty sheet if no detailed data
                pd.DataFrame().to_excel(writer, sheet_name="Detailed Results", index=False)
            
            # Sample Failed Rows sheet
            if sample_rows_data:
                # Create a DataFrame from the sample rows
                sample_df = pd.DataFrame(sample_rows_data)
                sample_df.to_excel(writer, sheet_name="Failed Data Samples", index=False)
                
                # Format the Sample Failed Rows sheet
                sample_sheet = writer.sheets["Failed Data Samples"]
                for col_num, value in enumerate(sample_df.columns.values):
                    sample_sheet.write(0, col_num, value, header_format)
            
            # Data Quality Metrics Visualization
            metrics_data = []
            for result in execution_results["results"]:
                if "statistics" in result:
                    stats = result["statistics"]
                    metrics_data.append({
                        "Rule Name": result["rule_name"],
                        "Total Rows": stats.get("total_rows", 0),
                        "Pass Rate (%)": (stats.get("successful_expectations", 0) / 
                                        max(stats.get("evaluated_expectations", 1), 1)) * 100
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name="Quality Metrics", index=False)
                
                # Format the Metrics sheet
                metrics_sheet = writer.sheets["Quality Metrics"]
                for col_num, value in enumerate(metrics_df.columns.values):
                    metrics_sheet.write(0, col_num, value, header_format)
                
                # Add a bar chart to visualize pass rates
                chart = workbook.add_chart({'type': 'bar'})
                chart.add_series({
                    'name': 'Pass Rate (%)',
                    'categories': ['Quality Metrics', 1, 0, len(metrics_data), 0],
                    'values': ['Quality Metrics', 1, 2, len(metrics_data), 2],
                })
                chart.set_title({'name': 'Data Quality Pass Rate by Rule'})
                chart.set_y_axis({'name': 'Rule'})
                chart.set_x_axis({'name': 'Pass Rate (%)'})
                metrics_sheet.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 2})
        
        return output_path 