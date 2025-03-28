import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.expectations.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.data_context.ephemeral_data_context import EphemeralDataContext
from great_expectations.data_context.types.base import DataContextConfig, InMemoryStoreBackendDefaults
from sqlalchemy import create_engine, text
from app.core.config import settings
from typing import List, Dict, Any
import json
from datetime import datetime
import os
import ruamel.yaml

class DataQualityEngine:
    def __init__(self):
        try:
            self.engine = create_engine(settings.DATABASE_URL)
        except Exception as e:
            print(f"Warning: Could not connect to database: {str(e)}")
            self.engine = None
        self.context = self._initialize_great_expectations()

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
            analytics_enabled=False
        )
            
        # Create an EphemeralDataContext
        return EphemeralDataContext(
            project_config=data_context_config
        )

    def execute_rules(self, table_name: str, rule_ids: List[int]) -> Dict[str, Any]:
        """Execute specified rules against the table."""
        results = []
        
        # Get rules from database
        with self.engine.connect() as connection:
            query = text("""
                SELECT id, name, rule_config
                FROM rules
                WHERE id = ANY(:rule_ids)
            """)
            rules = connection.execute(query, {"rule_ids": rule_ids}).fetchall()
            
            # Get table data
            data_query = text(f"SELECT * FROM {table_name}")
            df = pd.read_sql(data_query, connection)
            
            for rule in rules:
                rule_result = self._execute_single_rule(df, rule)
                results.append(rule_result)
        
        return {
            "table_name": table_name,
            "execution_time": datetime.now().isoformat(),
            "results": results
        }

    def _execute_single_rule(self, df: pd.DataFrame, rule: Any) -> Dict[str, Any]:
        """Execute a single rule against the dataframe."""
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name=rule.name,
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Create expectation suite
            suite = self.context.create_expectation_suite(
                expectation_suite_name=f"suite_{rule.id}",
                overwrite_existing=True
            )
            
            # Add expectations from rule config
            for expectation_config in rule.rule_config:
                expectation = ExpectationConfiguration(**expectation_config)
                suite.add_expectation(expectation)
            
            # Validate
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite.expectation_suite_name
            )
            results = validator.validate()
            
            return {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "success": results.success,
                "statistics": results.statistics,
                "results": results.results
            }
            
        except Exception as e:
            return {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "success": False,
                "error": str(e)
            }

    def generate_report(self, execution_results: Dict[str, Any], output_path: str):
        """Generate Excel report from execution results."""
        # Create summary sheet
        summary_data = []
        for result in execution_results["results"]:
            summary_data.append({
                "Rule ID": result["rule_id"],
                "Rule Name": result["rule_name"],
                "Status": "Success" if result["success"] else "Failed",
                "Error": result.get("error", ""),
                "Statistics": json.dumps(result.get("statistics", {}))
            })
        
        # Create detailed results sheet
        detailed_data = []
        for result in execution_results["results"]:
            if "results" in result:
                for validation_result in result["results"]:
                    detailed_data.append({
                        "Rule ID": result["rule_id"],
                        "Rule Name": result["rule_name"],
                        "Expectation Type": validation_result.get("expectation_type", ""),
                        "Success": validation_result.get("success", False),
                        "Details": json.dumps(validation_result.get("result", {}))
                    })
        
        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(detailed_data).to_excel(writer, sheet_name="Detailed Results", index=False) 