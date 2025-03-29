from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Union
import uvicorn
from app.core.config import settings
from app.db.session import get_db, engine
from app.services.rule_generator import AIRuleGenerator
from app.services.quality_engine import DataQualityEngine
from app.models.rule import Rule, RuleVersion
from pydantic import BaseModel
import os
from sqlalchemy import text, inspect, cast, String, JSON
from datetime import datetime
import pandas as pd
from fastapi.responses import JSONResponse
import logging
from app.services import db_utils
import json
from app.core.error_handling import handle_api_error, handle_database_error, format_error_response, get_error_message
import numpy as np
from sqlalchemy.exc import SQLAlchemyError

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add exception handlers for global error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle standard HTTP exceptions with structured responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle all other exceptions with a user-friendly message."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    error_message = get_error_message("internal_error")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": error_message,
            "status_code": 500
        }
    )

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "*"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With", "X-CSRF-Token", "*"],
    expose_headers=["Content-Type", "Content-Length", "Content-Disposition"],
    max_age=600,
)

# Pydantic models for request/response
class TableMetadata(BaseModel):
    column_name: str
    data_type: str
    is_nullable: bool
    column_default: str | None
    character_maximum_length: int | None

class TableSchema(BaseModel):
    table_name: str
    columns: List[TableMetadata]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, Any]]

class DatabaseConnectionResponse(BaseModel):
    status: str
    message: str
    database_url: str
    tables: List[str] = []

class ExpectationConfig(BaseModel):
    expectation_type: str
    kwargs: Dict[str, Any]

class RuleCreate(BaseModel):
    name: str
    description: str
    table_name: str
    rule_config: List[ExpectationConfig]  # Now a list of expectation configs

class NaturalLanguageRuleCreate(BaseModel):
    table_name: str
    rule_description: str
    rule_name: str = "Natural Language Generated Rule"

class RuleResponse(BaseModel):
    id: int
    name: str
    description: str
    table_name: str
    rule_config: List[ExpectationConfig]  # Now a list of expectation configs
    is_active: bool
    is_draft: bool = False
    confidence: int | None = None

    class Config:
        from_attributes = True

class RuleExecutionRequest(BaseModel):
    table_name: str
    rule_ids: List[int]

class RuleExecutionResponse(BaseModel):
    table_name: str
    execution_time: str
    total_duration: float
    total_rules: int
    successful_rules: int
    failed_rules: int
    success_rate: float
    results: List[dict]
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            # Handle numpy.bool_ and other numpy types
            'numpy.bool_': lambda v: bool(v),
            'numpy.integer': lambda v: int(v),
            'numpy.floating': lambda v: float(v),
            'numpy.ndarray': lambda v: v.tolist()
        }

class RuleUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    rule_config: List[ExpectationConfig] | None = None  # Now a list of expectation configs
    is_active: bool | None = None
    finalize_draft: bool = False  # Option to finalize a draft rule

class RuleClarificationResponse(BaseModel):
    rule_config: List[ExpectationConfig]  # Now a list of expectation configs
    needs_clarification: bool
    clarification_questions: List[str]
    confidence: int | None = None

class RuleConfig(BaseModel):
    expectation_type: str
    kwargs: Dict[str, Any]

class NewRuleSuggestion(BaseModel):
    rule_type: str
    column: str | None = None
    columns: List[str] | None = None
    description: str
    rule_config: List[ExpectationConfig]  # Now a list of expectation configs
    confidence: int

class RuleUpdateSuggestion(BaseModel):
    rule_id: int
    current_config: str | List[ExpectationConfig]  # Now a list of expectation configs
    suggested_config: List[ExpectationConfig]  # Now a list of expectation configs
    reason: str
    confidence: int

class RuleSuggestionResponse(BaseModel):
    table_name: str
    new_rule_suggestions: List[NewRuleSuggestion]
    rule_update_suggestions: List[RuleUpdateSuggestion]
    error: str | None = None

class ApplySuggestedRulesRequest(BaseModel):
    table_name: str
    new_rule_ids: List[int] = []  # IDs from the new_rule_suggestions list to apply
    update_rule_ids: List[int] = []  # IDs from the rule_update_suggestions list to apply

class ApplySuggestedRulesResponse(BaseModel):
    created_rules: List[RuleResponse]
    updated_rules: List[RuleResponse]
    errors: List[str] = []

class TableAnalysisRequest(BaseModel):
    table_name: str
    apply_suggestions: bool = False  # If true, automatically apply high-confidence suggestions

class DataStatistics(BaseModel):
    row_count: int
    column_stats: Dict[str, Any]
    multi_column_relationships: Dict[str, Any]

class TableAnalysisResponse(BaseModel):
    table_name: str
    column_analysis: Dict[str, Any]
    data_statistics: DataStatistics
    rule_suggestions: Dict[str, Any]
    existing_rules: List[Dict[str, Any]]
    applied_suggestions: ApplySuggestedRulesResponse | None = None

class RuleDetailResponse(BaseModel):
    id: int
    name: str
    description: str
    table_name: str
    rule_config: List[ExpectationConfig]  # Now a list of expectation configs
    is_active: bool
    is_draft: bool
    confidence: int | None
    created_at: datetime
    updated_at: datetime
    columns: List[str]
    versions: List[dict]

# Initialize services
try:
    rule_generator = AIRuleGenerator()
except Exception as e:
    print(f"Warning: Could not initialize AIRuleGenerator: {str(e)}")
    rule_generator = None

try:
    quality_engine = DataQualityEngine()
except Exception as e:
    print(f"Warning: Could not initialize DataQualityEngine: {str(e)}")
    quality_engine = None

@app.post(f"{settings.API_V1_STR}/rules/generate", response_model=List[RuleResponse])
async def generate_rules(table_name: str, db: Session = Depends(get_db)):
    """Generate rules for a table using AI."""
    rule_expectations = rule_generator.generate_rules(table_name)
    print(rule_expectations)
    created_rules = []
    
    # Group expectations by rule (they might come separated from the AI)
    rule_groups = {}
    
    for rule_config in rule_expectations:
        # Ensure rule_config is a valid dictionary
        if not isinstance(rule_config, dict):
            continue
            
        # Extract expectation info
        expectation_type = rule_config.get("expectation_type", "")
        kwargs = rule_config.get("kwargs", {})
        
        # Check if a similar rule already exists
        existing_rule = db.query(Rule).filter(
            Rule.table_name == table_name
        ).all()
        
        # Check if this expectation already exists in any rule
        skip_rule = False
        for rule in existing_rule:
            for existing_config in rule.rule_config:
                if (existing_config.get("expectation_type") == expectation_type and 
                    existing_config.get("kwargs") == kwargs):
                    print(f"Skipping duplicate expectation: {rule_config}")
                    skip_rule = True
                    break
            if skip_rule:
                break
                
        if skip_rule:
            continue
        
        # Determine the rule group based on columns involved
        column_key = ""
        if "column" in kwargs:
            column_key = f"single:{kwargs['column']}"
        elif "column_A" in kwargs and "column_B" in kwargs:
            column_key = f"pair:{kwargs['column_A']}:{kwargs['column_B']}"
        else:
            # Generic grouping for other cases
            column_key = f"other:{expectation_type}"
        
        # Add to appropriate rule group
        if column_key not in rule_groups:
            rule_groups[column_key] = []
        rule_groups[column_key].append({
            "expectation_type": expectation_type,
            "kwargs": kwargs
        })
    
    # Create rules from grouped expectations
    for column_key, expectations in rule_groups.items():
        # Skip empty groups
        if not expectations:
            continue
            
        # Create a descriptive name
        if column_key.startswith("single:"):
            column = column_key.split(":")[1]
            rule_name = f"AI Generated Rule for {column} in {table_name}"
        elif column_key.startswith("pair:"):
            parts = column_key.split(":")
            col1, col2 = parts[1], parts[2]
            rule_name = f"AI Generated Rule for relationship between {col1} and {col2} in {table_name}"
        else:
            rule_name = f"AI Generated Rule for {table_name}"
        
        # Create a new rule with the group of expectations
        rule = Rule(
            name=rule_name,
            description=f"Automatically generated rule with {len(expectations)} expectations based on data analysis",
            table_name=table_name,
            rule_config=expectations,
            is_active=True
        )
        
        try:
            db.add(rule)
            db.commit()
            db.refresh(rule)
            created_rules.append(rule)
        except Exception as e:
            db.rollback()
            print(f"Error creating rule: {str(e)}")
            continue
    
    if not created_rules:
        raise HTTPException(
            status_code=500,
            detail="No valid rules were created"
        )
        
    return created_rules

@app.get(f"{settings.API_V1_STR}/rules/check-outdated/{{rule_id}}")
async def check_rule_outdated(rule_id: int, db: Session = Depends(get_db)):
    """Check if a rule is outdated based on current data patterns."""
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        result = rule_generator.check_rule_outdated(rule_id, rule.table_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/rules/execute", response_model=RuleExecutionResponse)
async def execute_rules(request: RuleExecutionRequest):
    """Execute specified rules against a table."""
    try:
        # Execute rules and get results
        results = quality_engine.execute_rules(request.table_name, request.rule_ids)
        
        # Ensure all numpy types are converted to Python native types
        def sanitize_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: sanitize_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_numpy_types(item) for item in obj]
            elif str(type(obj)).startswith("<class 'numpy."):
                # Convert numpy types to native Python types
                if hasattr(obj, 'item'):
                    return obj.item()  # Convert numpy scalars to Python scalars
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()  # Convert numpy arrays to lists
                else:
                    return str(obj)  # Fallback to string conversion
            else:
                return obj
                
        # Apply the sanitization to the entire results
        results = sanitize_numpy_types(results)
        
        # Verify and ensure all validations with unexpected data have sample rows
        for rule_result in results.get("results", []):
            validations = rule_result.get("results", [])
            for validation in validations:
                # Check if there are unexpected values in this validation
                unexpected_count = validation.get("result", {}).get("unexpected_count", 0)
                # If there are unexpected values, ensure we have sample rows regardless of success
                if unexpected_count > 0:
                    # Ensure sample_rows exists and is not empty if we have unexpected data
                    if not validation.get("sample_rows"):
                        print(f"Warning: Validation has {unexpected_count} unexpected values but no sample rows")
                        validation["sample_rows"] = []  # Placeholder empty array
        
        # Generate report
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir,
            f"quality_report_{request.table_name}_{results['execution_time']}.xlsx"
        )
        quality_engine.generate_report(results, report_path)
        
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/tables/{{table_name}}/suggest-rules")
async def suggest_rules_for_table(table_name: str):
    """
    Generate rule suggestions for a given table based on current data patterns and anomalies.
    """
    try:
        # Check if the table exists
        if not db_utils.table_exists(table_name):
            return JSONResponse(
                status_code=404,
                content={
                    "table_name": table_name, 
                    "new_rule_suggestions": [], 
                    "rule_update_suggestions": [],
                    "error": get_error_message("table_not_found", table_name=table_name)
                }
            )

        # Generate rule suggestions
        try:
            suggestions = rule_generator.scan_table_for_rule_suggestions(table_name)
            
            # Validate and fix the response structure
            new_rule_suggestions = []
            rule_update_suggestions = []
            
            # Process new rule suggestions
            for rule in suggestions.get("new_rule_suggestions", []):
                try:
                    # Skip rules without rule_config
                    if "rule_config" not in rule:
                        continue
                        
                    # Ensure rule_config is a list
                    if not isinstance(rule["rule_config"], list):
                        rule["rule_config"] = [rule["rule_config"]]
                    
                    # Convert rule_config to string to prevent React rendering issues
                    stringified_rule_config = json.dumps(rule["rule_config"])
                    
                    # Add required fields if missing
                    new_rule = {
                        "rule_type": rule.get("rule_type", "unknown"),
                        "column": rule.get("column"),
                        "columns": rule.get("columns"),
                        "description": rule.get("description", "Auto-generated rule"),
                        "rule_config": stringified_rule_config,
                        "confidence": rule.get("confidence", 90)
                    }
                    new_rule_suggestions.append(new_rule)
                except Exception as e:
                    logger.error(f"Error processing rule suggestion: {str(e)}")
                    # Skip invalid rules
                    continue
                    
            # Process rule update suggestions
            for rule in suggestions.get("rule_update_suggestions", []):
                try:
                    # Skip rules without required fields
                    if "rule_id" not in rule or "suggested_config" not in rule:
                        continue
                        
                    # Ensure rule_id is an integer
                    if not isinstance(rule["rule_id"], int):
                        try:
                            rule["rule_id"] = int(rule["rule_id"])
                        except (ValueError, TypeError):
                            # Skip rules with invalid IDs
                            continue
                    
                    # Ensure suggested_config is a list
                    if not isinstance(rule["suggested_config"], list):
                        rule["suggested_config"] = [rule["suggested_config"]]
                    
                    # Convert suggested_config to string
                    stringified_suggested_config = json.dumps(rule["suggested_config"])
                    
                    # Format current_config
                    current_config = rule.get("current_config", "")
                    # If current_config is a list or dict, convert to string
                    if isinstance(current_config, (list, dict)):
                        current_config = json.dumps(current_config)
                    
                    update_rule = {
                        "rule_id": rule["rule_id"],
                        "current_config": current_config,
                        "suggested_config": stringified_suggested_config,
                        "reason": rule.get("reason", "Suggested based on data analysis"),
                        "confidence": rule.get("confidence", 90)
                    }
                    rule_update_suggestions.append(update_rule)
                except Exception as e:
                    logger.error(f"Error processing rule update suggestion: {str(e)}")
                    # Skip invalid rules
                    continue
            
            # Return formatted response
            return JSONResponse(
                status_code=200,
                content={
                    "table_name": table_name,
                    "new_rule_suggestions": new_rule_suggestions,
                    "rule_update_suggestions": rule_update_suggestions,
                    "error": suggestions.get("error")
                }
            )
        except Exception as e:
            logger.exception(f"Error generating rule suggestions for table {table_name}: {str(e)}")
            error_response = format_error_response(
                error=e, 
                error_key="rule_generation_error", 
                table_name=table_name, 
                detail=str(e)
            )
            return JSONResponse(
                status_code=500,
                content={
                    "table_name": table_name, 
                    "new_rule_suggestions": [], 
                    "rule_update_suggestions": [],
                    "error": error_response["error"]
                }
            )
    except Exception as e:
        logger.exception(f"Unhandled exception in suggest_rules_for_table endpoint: {str(e)}")
        error_response = format_error_response(
            error=e, 
            error_key="internal_error"
        )
        return JSONResponse(
            status_code=500,
            content={
                "table_name": table_name, 
                "new_rule_suggestions": [], 
                "rule_update_suggestions": [],
                "error": error_response["error"]
            }
        )

@app.post(f"{settings.API_V1_STR}/tables/{{table_name}}/apply-suggested-rules", response_model=ApplySuggestedRulesResponse)
async def apply_suggested_rules(
    table_name: str, 
    request: ApplySuggestedRulesRequest,
    db: Session = Depends(get_db)
):
    """Apply suggested new rules and rule updates."""
    try:
        # Verify table exists
        try:
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :table_name
                    )
                """), {"table_name": table_name})
                
                if not result.scalar():
                    raise handle_api_error(
                        error=ValueError(f"Table '{table_name}' not found"),
                        status_code=404,
                        error_key="table_not_found",
                        table_name=table_name
                    )
        except HTTPException:
            raise
        except SQLAlchemyError as e:
            raise handle_database_error(e, table_name=table_name)
        
        # Get the rule suggestions
        try:
            suggestions = rule_generator.scan_table_for_rule_suggestions(table_name)
        except Exception as e:
            logger.error(f"Error getting rule suggestions for table {table_name}: {str(e)}")
            raise handle_api_error(
                error=e, 
                status_code=500, 
                error_key="rule_generation_error", 
                table_name=table_name,
                detail=str(e)
            )
        
        created_rules = []
        updated_rules = []
        errors = []
        
        # Apply new rules
        for rule_id in request.new_rule_ids:
            try:
                # Find the corresponding rule suggestion
                rule_suggestion = next(
                    (r for i, r in enumerate(suggestions["new_rule_suggestions"]) if i == rule_id), 
                    None
                )
                
                if not rule_suggestion:
                    errors.append(get_error_message(
                        "resource_not_found", 
                        resource_type=f"rule suggestion with ID {rule_id}"
                    ))
                    continue
                
                # Extract rule configs - may now be a list of expectations
                rule_configs = rule_suggestion.get("rule_config", [])
                
                # Ensure rule_configs is a list
                if not isinstance(rule_configs, list):
                    rule_configs = [rule_configs]
                
                # Ensure all rule configs have proper structure
                clean_configs = []
                for config in rule_configs:
                    clean_configs.append({
                        "expectation_type": config.get("expectation_type", ""),
                        "kwargs": config.get("kwargs", {})
                    })
                
                # Check if a similar rule already exists by comparing each expectation
                # with existing rules
                skip_rule = False
                existing_rules = db.query(Rule).filter(Rule.table_name == table_name).all()
                
                for existing_rule in existing_rules:
                    if len(existing_rule.rule_config) != len(clean_configs):
                        continue
                    
                    match_count = 0
                    for new_expectation in clean_configs:
                        for existing_expectation in existing_rule.rule_config:
                            if (new_expectation["expectation_type"] == existing_expectation["expectation_type"] and
                                new_expectation["kwargs"] == existing_expectation["kwargs"]):
                                match_count += 1
                                break
                    
                    if match_count == len(clean_configs):
                        # This rule already exists
                        errors.append(f"Similar rule already exists (Rule ID: {existing_rule.id})")
                        skip_rule = True
                        break
                
                if skip_rule:
                    continue
                
                # Create the new rule
                rule_name = rule_suggestion.get("description", "").split(".")[0]
                if not rule_name:
                    rule_name = f"Rule for {table_name}"
                
                rule_description = rule_suggestion.get("description", f"Automatically created rule for {table_name}")
                
                new_rule = Rule(
                    name=f"AI Suggested Rule - {rule_name}",
                    description=rule_description,
                    table_name=table_name,
                    rule_config=clean_configs,
                    is_active=True,  # New rule is active by default
                    is_draft=False,
                    confidence=rule_suggestion.get("confidence", None)
                )
                
                db.add(new_rule)
                db.flush()  # Flush to get the ID, but don't commit yet
                
                # Create a version record
                version = RuleVersion(
                    rule_id=new_rule.id,
                    version_number=1,  # First version
                    rule_config=clean_configs,
                    is_current=True
                )
                
                db.add(version)
                db.commit()
                db.refresh(new_rule)
                
                created_rules.append(new_rule)
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error creating rule from suggestion {rule_id}: {str(e)}")
                errors.append(get_error_message("database_query", detail=str(e)))
            except Exception as e:
                db.rollback()
                logger.error(f"Error creating rule from suggestion {rule_id}: {str(e)}")
                errors.append(get_error_message("rule_generation_error", table_name=table_name, detail=str(e)))
        
        # Apply rule updates
        for rule_id in request.update_rule_ids:
            try:
                # Find the corresponding update suggestion
                update_suggestion = next(
                    (r for i, r in enumerate(suggestions["rule_update_suggestions"]) if i == rule_id), 
                    None
                )
                
                if not update_suggestion:
                    errors.append(get_error_message(
                        "resource_not_found", 
                        resource_type=f"rule update suggestion with ID {rule_id}"
                    ))
                    continue
                
                # Find the rule to update
                rule = db.query(Rule).filter(Rule.id == update_suggestion["rule_id"]).first()
                
                if not rule:
                    errors.append(get_error_message("rule_not_found", rule_id=update_suggestion["rule_id"]))
                    continue
                
                # Extract the suggested configs
                suggested_configs = update_suggestion.get("suggested_config", [])
                
                # Ensure it's a list
                if not isinstance(suggested_configs, list):
                    suggested_configs = [suggested_configs]
                
                # Ensure all configs have proper structure
                clean_configs = []
                for config in suggested_configs:
                    clean_configs.append({
                        "expectation_type": config.get("expectation_type", ""),
                        "kwargs": config.get("kwargs", {})
                    })
                
                # Update the rule
                rule.rule_config = clean_configs
                
                # Create a version record
                new_version = RuleVersion(
                    rule_id=rule.id,
                    version_number=len(rule.versions) + 1,
                    rule_config=rule.rule_config,
                    is_current=True
                )
                
                # Set all other versions as not current
                for version in rule.versions:
                    version.is_current = False
                
                db.add(new_version)
                db.commit()
                db.refresh(rule)
                updated_rules.append(rule)
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error updating rule {rule_id}: {str(e)}")
                errors.append(get_error_message("database_query", detail=str(e)))
            except Exception as e:
                db.rollback()
                logger.error(f"Error updating rule {rule_id}: {str(e)}")
                errors.append(get_error_message("rule_update_error", detail=str(e)))
        
        return {
            "created_rules": created_rules,
            "updated_rules": updated_rules,
            "errors": errors
        }
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception(f"Database error: {str(e)}")
        raise handle_database_error(e)
    except Exception as e:
        db.rollback()
        logger.exception(f"Unhandled error in apply_suggested_rules: {str(e)}")
        raise handle_api_error(
            error=e,
            status_code=500,
            error_key="internal_error"
        )

@app.post(f"{settings.API_V1_STR}/analyze-table", response_model=TableAnalysisResponse)
async def analyze_table(request: TableAnalysisRequest, db: Session = Depends(get_db)):
    """Perform comprehensive table analysis including rule suggestions."""
    try:
        logger.info(f"Starting analysis for table {request.table_name}")
        table_name = request.table_name
        
        # Verify table exists
        try:
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :table_name
                    )
                """), {"table_name": table_name})
                
                if not result.scalar():
                    raise handle_api_error(
                        error=ValueError(f"Table {table_name} not found"),
                        status_code=404,
                        error_key="table_not_found",
                        table_name=table_name
                    )
        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error checking table existence: {str(e)}")
            raise handle_database_error(e, table_name=table_name)
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            raise handle_api_error(
                error=e,
                status_code=500,
                error_key="database_query",
                detail=str(e)
            )
        
        # Get table schema and sample data
        try:
            schema_info = rule_generator.analyze_table_schema(table_name)
        except Exception as e:
            logger.error(f"Error analyzing table schema: {str(e)}")
            raise handle_api_error(
                error=e, 
                status_code=500, 
                error_key="sampling_error", 
                detail=f"Failed to analyze table schema: {str(e)}"
            )
        
        # Get column relationship insights for multi-column rules
        multi_column_insights = {}
        try:
            multi_column_insights = rule_generator._analyze_column_relationships(table_name, schema_info)
        except Exception as e:
            logger.warning(f"Error analyzing column relationships: {str(e)}")
            # Continue without these insights - non-critical error
        
        # Calculate basic data statistics on random sample of 100 rows for better performance
        row_count = 0
        column_stats = {}
        
        try:
            with engine.connect() as connection:
                # Get total row count (fast query)
                row_count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = row_count_result.scalar()
                
                # Get column statistics using random sampling for better performance
                for column in schema_info['columns']:
                    col_name = column['column_name']
                    data_type = column['data_type']
                    
                    # Get common statistics based on data type
                    # Using TABLESAMPLE to get a random sample for faster processing
                    stats_query = ""
                    
                    if data_type in ('integer', 'bigint', 'numeric', 'real', 'double precision'):
                        # Numeric column - apply random sampling for better performance
                        stats_query = f"""
                            SELECT 
                                MIN({col_name}) as min_value,
                                MAX({col_name}) as max_value,
                                AVG({col_name}) as avg_value,
                                COUNT({col_name}) as non_null_count,
                                COUNT(*) - COUNT({col_name}) as null_count
                            FROM (
                                SELECT {col_name} FROM {table_name} ORDER BY RANDOM() LIMIT 100
                            ) t
                        """
                    elif data_type in ('character varying', 'text', 'character', 'varchar'):
                        # Text column - apply random sampling for better performance
                        stats_query = f"""
                            SELECT 
                                MIN(LENGTH({col_name})) as min_length,
                                MAX(LENGTH({col_name})) as max_length,
                                AVG(LENGTH({col_name})) as avg_length,
                                COUNT({col_name}) as non_null_count,
                                COUNT(*) - COUNT({col_name}) as null_count
                            FROM (
                                SELECT {col_name} FROM {table_name} ORDER BY RANDOM() LIMIT 100
                            ) t
                        """
                    elif data_type in ('date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone'):
                        # Date/timestamp column - apply random sampling for better performance
                        stats_query = f"""
                            SELECT 
                                MIN({col_name}) as min_date,
                                MAX({col_name}) as max_date,
                                COUNT({col_name}) as non_null_count,
                                COUNT(*) - COUNT({col_name}) as null_count
                            FROM (
                                SELECT {col_name} FROM {table_name} ORDER BY RANDOM() LIMIT 100
                            ) t
                        """
                    elif data_type in ('boolean'):
                        # Boolean column - apply random sampling for better performance
                        stats_query = f"""
                            SELECT 
                                COUNT({col_name}) FILTER (WHERE {col_name} = true) as true_count,
                                COUNT({col_name}) FILTER (WHERE {col_name} = false) as false_count,
                                COUNT({col_name}) as non_null_count,
                                COUNT(*) - COUNT({col_name}) as null_count
                            FROM (
                                SELECT {col_name} FROM {table_name} ORDER BY RANDOM() LIMIT 100
                            ) t
                        """
                    else:
                        # Other column types - just count nulls
                        stats_query = f"""
                            SELECT 
                                COUNT({col_name}) as non_null_count,
                                COUNT(*) - COUNT({col_name}) as null_count
                            FROM (
                                SELECT {col_name} FROM {table_name} ORDER BY RANDOM() LIMIT 100
                            ) t
                        """
                    
                    if stats_query:
                        try:
                            stats_result = connection.execute(text(stats_query))
                            col_stats = dict(zip(stats_result.keys(), stats_result.fetchone()))
                            # Convert any non-serializable types
                            for k, v in col_stats.items():
                                if hasattr(v, 'isoformat'):  # Handle date/datetime
                                    col_stats[k] = v.isoformat()
                                elif v is None:
                                    col_stats[k] = None
                            
                            # Add note that stats are based on sample
                            col_stats["sample_size"] = 100
                            col_stats["is_sample"] = True
                            
                            column_stats[col_name] = col_stats
                        except Exception as e:
                            logger.warning(f"Error getting stats for column {col_name}: {str(e)}")
                            column_stats[col_name] = {"error": str(e), "is_sample": True}
        except SQLAlchemyError as e:
            logger.error(f"Database error calculating statistics: {str(e)}")
            # Continue with partial statistics - non-critical error
            if not column_stats:
                column_stats = {"error": str(e)}
        except Exception as e:
            logger.error(f"Error calculating data statistics: {str(e)}")
            # Continue with partial statistics - non-critical error
            if not column_stats:
                column_stats = {"error": str(e)}
        
        # Get existing rules
        existing_rules = []
        try:
            rules_result = db.query(Rule).filter(Rule.table_name == table_name).all()
            for rule in rules_result:
                # Ensure rule_config is in list format
                rule_config = rule.rule_config
                if rule_config and not isinstance(rule_config, list):
                    rule_config = [{
                        "expectation_type": rule_config.get("expectation_type", ""),
                        "kwargs": rule_config.get("kwargs", {})
                    }]
                
                existing_rules.append({
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "rule_config": rule_config,
                    "is_active": rule.is_active
                })
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching existing rules: {str(e)}")
            # Continue with empty existing rules - non-critical error
        except Exception as e:
            logger.error(f"Error fetching existing rules: {str(e)}")
            # Continue with empty existing rules - non-critical error
        
        # Get rule suggestions
        rule_suggestions = {"new_rule_suggestions": [], "rule_update_suggestions": []}
        try:
            if rule_generator:
                rule_suggestions = rule_generator.scan_table_for_rule_suggestions(table_name)
                
                # Validate rule suggestions format
                if not isinstance(rule_suggestions, dict):
                    logger.warning(f"Rule suggestions response is not a dictionary: {type(rule_suggestions)}")
                    rule_suggestions = {
                        "new_rule_suggestions": [],
                        "rule_update_suggestions": [],
                        "error": get_error_message("invalid_json", data_type="rule suggestions")
                    }
                
                if "new_rule_suggestions" not in rule_suggestions:
                    rule_suggestions["new_rule_suggestions"] = []
                
                if "rule_update_suggestions" not in rule_suggestions:
                    rule_suggestions["rule_update_suggestions"] = []
            else:
                logger.warning("AIRuleGenerator not available for rule suggestions")
                rule_suggestions = {
                    "new_rule_suggestions": [],
                    "rule_update_suggestions": [],
                    "error": get_error_message("ai_service_error")
                }
        except Exception as e:
            logger.error(f"Error generating rule suggestions: {str(e)}")
            rule_suggestions = {
                "new_rule_suggestions": [],
                "rule_update_suggestions": [],
                "error": get_error_message("rule_generation_error", table_name=table_name, detail=str(e))
            }
        
        # Apply high-confidence suggestions if requested
        applied_suggestions = None
        if request.apply_suggestions and rule_suggestions and not rule_suggestions.get("error"):
            try:
                # Automatically apply suggestions with confidence >= 90
                high_confidence_new_rules = [
                    i for i, r in enumerate(rule_suggestions.get("new_rule_suggestions", []))
                    if r.get("confidence", 0) >= 90
                ]
                
                high_confidence_updates = [
                    i for i, r in enumerate(rule_suggestions.get("rule_update_suggestions", []))
                    if r.get("confidence", 0) >= 90
                ]
                
                if high_confidence_new_rules or high_confidence_updates:
                    apply_request = ApplySuggestedRulesRequest(
                        table_name=table_name,
                        new_rule_ids=high_confidence_new_rules,
                        update_rule_ids=high_confidence_updates
                    )
                    
                    applied_suggestions = await apply_suggested_rules(
                        table_name=table_name,
                        request=apply_request,
                        db=db
                    )
            except Exception as e:
                logger.error(f"Error applying high-confidence suggestions: {str(e)}")
                # Continue without applied suggestions - non-critical error
        
        # Return the complete analysis response
        return {
            "table_name": table_name,
            "column_analysis": schema_info,
            "data_statistics": {
                "row_count": row_count,
                "column_stats": column_stats,
                "multi_column_relationships": multi_column_insights
            },
            "rule_suggestions": rule_suggestions,
            "existing_rules": existing_rules,
            "applied_suggestions": applied_suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error in analyze_table endpoint: {str(e)}")
        raise handle_api_error(
            error=e,
            status_code=500,
            error_key="internal_error"
        )

@app.get(f"{settings.API_V1_STR}/rules", response_model=List[RuleResponse])
async def list_rules(db: Session = Depends(get_db)):
    """List all rules."""
    try:
        rules = db.query(Rule).all()
        
        # Handle the transition to multi-expectation rule_config format
        for rule in rules:
            # If rule_config is not a list, convert it to a list with a single expectation
            if rule.rule_config and not isinstance(rule.rule_config, list):
                rule.rule_config = [{
                    "expectation_type": rule.rule_config.get("expectation_type", ""),
                    "kwargs": rule.rule_config.get("kwargs", {})
                }]
        
        return rules
    except Exception as e:
        # Log the error for debugging
        print(f"Error in list_rules: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving rules: {str(e)}"
        )

@app.get(f"{settings.API_V1_STR}/rules/{{rule_id}}", response_model=RuleDetailResponse)
async def get_rule(rule_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific rule."""
    try:
        # Get the rule with versions
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        
        if not rule:
            raise handle_api_error(
                error=ValueError(f"Rule with ID {rule_id} not found"),
                status_code=404,
                error_key="rule_not_found",
                rule_id=rule_id
            )
        
        # Extract columns from rule config
        rule_columns = set()
        for expectation in rule.rule_config:
            kwargs = expectation.get("kwargs", {})
            
            # Extract from common kwargs patterns
            if "column" in kwargs:
                rule_columns.add(kwargs["column"])
            if "column_A" in kwargs and "column_B" in kwargs:
                rule_columns.add(kwargs["column_A"])
                rule_columns.add(kwargs["column_B"])
            if "columns" in kwargs and isinstance(kwargs["columns"], list):
                rule_columns.update(kwargs["columns"])
        
        # Get all versions and format them
        versions = []
        for version in rule.versions:
            versions.append({
                "id": version.id,
                "version_number": version.version_number,
                "rule_config": version.rule_config,
                "is_current": version.is_current,
                "created_at": version.created_at
            })
        
        # Sort versions by number in descending order (newest first)
        versions.sort(key=lambda v: v["version_number"], reverse=True)
        
        # Return detailed rule information
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "table_name": rule.table_name,
            "rule_config": rule.rule_config,
            "is_active": rule.is_active,
            "is_draft": rule.is_draft,
            "confidence": rule.confidence,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at,
            "columns": list(rule_columns),
            "versions": versions
        }
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.exception(f"Database error fetching rule {rule_id}: {str(e)}")
        raise handle_database_error(e)
    except Exception as e:
        logger.exception(f"Error fetching rule {rule_id}: {str(e)}")
        raise handle_api_error(
            error=e,
            status_code=500,
            error_key="internal_error"
        )

@app.put(f"{settings.API_V1_STR}/rules/{{rule_id}}", response_model=RuleResponse)
async def update_rule(
    rule_id: int,
    rule_update: RuleUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing rule."""
    try:
        # Find the rule
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        
        if not rule:
            raise handle_api_error(
                error=ValueError(f"Rule with ID {rule_id} not found"),
                status_code=404,
                error_key="rule_not_found",
                rule_id=rule_id
            )
        
        # Update fields if provided
        if rule_update.name is not None:
            rule.name = rule_update.name
            
        if rule_update.description is not None:
            rule.description = rule_update.description
            
        if rule_update.is_active is not None:
            rule.is_active = rule_update.is_active
        
        # Create a new version if the rule config is updated
        if rule_update.rule_config is not None:
            # Validate rule configuration
            try:
                # Check rule_config is a list of ExpectationConfig objects
                rule_configs = rule_update.rule_config
                
                # Create a list of plain dictionaries from the Pydantic models
                clean_configs = []
                for config in rule_configs:
                    clean_configs.append({
                        "expectation_type": config.expectation_type,
                        "kwargs": config.kwargs
                    })
                
                # Apply the new rule configuration
                rule.rule_config = clean_configs
                
                # Create a new version
                version_number = 1
                if rule.versions:
                    version_number = max(v.version_number for v in rule.versions) + 1
                
                new_version = RuleVersion(
                    rule_id=rule.id,
                    version_number=version_number,
                    rule_config=rule.rule_config,
                    is_current=True
                )
                
                # Set all other versions as not current
                for version in rule.versions:
                    version.is_current = False
                
                db.add(new_version)
                
            except Exception as e:
                logger.error(f"Error updating rule configuration: {str(e)}")
                raise handle_api_error(
                    error=e,
                    status_code=400,
                    error_key="rule_update_error",
                    detail=f"Invalid rule configuration: {str(e)}"
                )
        
        # Finalize a draft rule if requested
        if rule_update.finalize_draft and rule.is_draft:
            rule.is_draft = False
            
            # If no name was provided, generate a better name
            if rule_update.name is None and not rule.name:
                # Extract column information from all expectations
                columns = []
                for expectation in rule.rule_config:
                    kwargs = expectation.get("kwargs", {})
                    
                    if "column" in kwargs:
                        columns.append(kwargs["column"])
                        
                    if "column_A" in kwargs and "column_B" in kwargs:
                        columns.append(f"{kwargs['column_A']}/{kwargs['column_B']}")
                
                # Create a descriptive name if possible
                if columns:
                    column_str = ", ".join(set(columns))
                    rule.name = f"Rule for {column_str}"
                else:
                    rule.name = f"Rule for {rule.table_name}"
        
        # Save changes and return the updated rule
        rule.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(rule)
        
        return rule
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception(f"Database error updating rule {rule_id}: {str(e)}")
        raise handle_database_error(e)
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating rule {rule_id}: {str(e)}")
        raise handle_api_error(
            error=e,
            status_code=500,
            error_key="rule_update_error",
            detail=str(e)
        )

@app.delete(f"{settings.API_V1_STR}/rules/{{rule_id}}")
async def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    """Delete a specific rule and its versions."""
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        # Delete all versions first
        db.query(RuleVersion).filter(RuleVersion.rule_id == rule_id).delete()
        
        # Delete the rule
        db.delete(rule)
        db.commit()
        
        return {"message": "Rule deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting rule: {str(e)}"
        )

@app.get(f"{settings.API_V1_STR}/database/tables", response_model=List[str])
async def list_tables():
    """List all tables in the database."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            return [row[0] for row in result]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tables: {str(e)}"
        )

@app.get(f"{settings.API_V1_STR}/database/tables/{{table_name}}/schema", response_model=TableSchema)
async def get_table_schema(table_name: str):
    """Get detailed schema information for a specific table."""
    try:
        with engine.connect() as connection:
            # Get column information
            columns_result = connection.execute(text("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})
            
            columns = [
                TableMetadata(
                    column_name=row[0],
                    data_type=row[1],
                    is_nullable=row[2] == 'YES',
                    column_default=row[3],
                    character_maximum_length=row[4]
                )
                for row in columns_result
            ]
            
            # Get primary key information
            pk_result = connection.execute(text("""
                SELECT c.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                    AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = :table_name
            """), {"table_name": table_name})
            
            primary_keys = [row[0] for row in pk_result]
            
            # Get foreign key information
            fk_result = connection.execute(text("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = :table_name
            """), {"table_name": table_name})
            
            foreign_keys = [
                {
                    "column_name": row[0],
                    "references_table": row[1],
                    "references_column": row[2]
                }
                for row in fk_result
            ]
            
            return TableSchema(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get table schema: {str(e)}"
        )

@app.post(f"{settings.API_V1_STR}/rules/generate-from-description", response_model=RuleResponse)
async def generate_rule_from_description(
    rule_data: NaturalLanguageRuleCreate,
    db: Session = Depends(get_db)
):
    """Generate a rule from natural language description."""
    try:
        # Basic input validation
        if not rule_data.rule_description:
            raise HTTPException(
                status_code=400,
                detail="Rule description cannot be empty"
            )
            
        # Check if table exists
        table_exists = False
        try:
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = :table_name
                """), {"table_name": rule_data.table_name})
                table_exists = result.first() is not None
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error checking table existence: {str(e)}"
            )
            
        if not table_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{rule_data.table_name}' does not exist"
            )
            
        # Generate rule using the rule generator
        rule_expectations = rule_generator.generate_rule_from_description(
            table_name=rule_data.table_name,
            rule_description=rule_data.rule_description
        )
        
        if not rule_expectations:
            raise HTTPException(
                status_code=400,
                detail="Could not generate valid rule from description"
            )
        
        # We might get multiple expectations for a single rule
        confidence = 0
        is_draft = False
        missing_columns = []
        
        # Extract all columns from all the expectations to validate they exist
        rule_columns = []
        
        for rule_config in rule_expectations:
            # Update overall confidence as the average of all expectations
            if "confidence" in rule_config:
                confidence += rule_config.get("confidence", 0)
            
            kwargs = rule_config.get("kwargs", {})
            
            # Check for single column in kwargs
            if "column" in kwargs:
                rule_columns.append(kwargs["column"])
            
            # Check for column pairs in kwargs
            if "column_A" in kwargs:
                rule_columns.append(kwargs["column_A"])
            if "column_B" in kwargs:
                rule_columns.append(kwargs["column_B"])
            
            # Check for multi-column expectations with arrays of columns
            if "columns" in kwargs:
                if isinstance(kwargs["columns"], list):
                    rule_columns.extend(kwargs["columns"])
        
        # Calculate average confidence
        if rule_expectations:
            confidence = confidence // len(rule_expectations)
        
        # Validate all extracted columns exist in the table
        if rule_columns:
            with engine.connect() as connection:
                for column_name in rule_columns:
                    result = connection.execute(text("""
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name
                    """), {"table_name": rule_data.table_name, "column_name": column_name})
                    
                    if result.first() is None:
                        is_draft = True
                        missing_columns.append(column_name)
        
        # Rules are in draft if: columns don't exist, confidence is low, or other parameters are missing
        if missing_columns or confidence < 70:
            is_draft = True
        
        # For columns that don't exist, save in draft mode
        if missing_columns:
            print(f"Warning: Columns {missing_columns} don't exist in table '{rule_data.table_name}'. Saving as draft.")
        
        # Check if a similar rule with the same set of expectations already exists
        similar_rule_exists = False
        existing_rules = db.query(Rule).filter(Rule.table_name == rule_data.table_name).all()
        
        for existing_rule in existing_rules:
            # Check if every expectation matches an existing rule's expectations
            if len(existing_rule.rule_config) == len(rule_expectations):
                match_count = 0
                for new_expectation in rule_expectations:
                    for existing_expectation in existing_rule.rule_config:
                        if (existing_expectation.get("expectation_type") == new_expectation.get("expectation_type") and 
                            existing_expectation.get("kwargs") == new_expectation.get("kwargs")):
                            match_count += 1
                            break
                
                if match_count == len(rule_expectations):
                    similar_rule_exists = True
                    existing_rule_id = existing_rule.id
                    existing_rule_name = existing_rule.name
                    break
        
        if similar_rule_exists:
            raise HTTPException(
                status_code=400,
                detail=f"A similar rule already exists: {existing_rule_name} (ID: {existing_rule_id})"
            )
        
        # Clean up the expectations by removing metadata
        clean_expectations = []
        for rule_config in rule_expectations:
            clean_expectations.append({
                "expectation_type": rule_config.get("expectation_type", ""),
                "kwargs": rule_config.get("kwargs", {})
            })
        
        # Create a descriptive name that mentions all columns
        rule_name = rule_data.rule_name
        if not rule_name:
            if len(rule_columns) > 0:
                rule_name = f"Rule for {rule_data.table_name}: {', '.join(set(rule_columns))}"
            else:
                rule_name = f"Rule for {rule_data.table_name}"
        
        rule = Rule(
            name=rule_name,
            description=rule_data.rule_description,
            table_name=rule_data.table_name,
            rule_config=clean_expectations,
            is_active=True,
            is_draft=is_draft,
            confidence=confidence
        )
        
        db.add(rule)
        db.commit()
        db.refresh(rule)
        
        return rule
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        db.rollback()
        print(f"Error generating rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating rule: {str(e)}"
        )

@app.put(f"{settings.API_V1_STR}/rules/{{rule_id}}/finish-draft", response_model=RuleResponse)
async def finish_draft_rule(
    rule_id: int,
    db: Session = Depends(get_db)
):
    """Mark a draft rule as finalized after user modifications."""
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
            
        if not rule.is_draft:
            return rule  # Already finalized
            
        # Check if all columns in the rule exist
        missing_columns = []
        all_columns = []
            
        # Extract all columns from all expectations
        for expectation in rule.rule_config:
            kwargs = expectation.get("kwargs", {})
            
            # Check for single column format
            if "column" in kwargs:
                all_columns.append(kwargs["column"])
                
            # Check for column pair format (multi-column)
            if "column_A" in kwargs:
                all_columns.append(kwargs["column_A"])
            if "column_B" in kwargs:
                all_columns.append(kwargs["column_B"])
                
            # Check for column list format
            if "columns" in kwargs and isinstance(kwargs["columns"], list):
                all_columns.extend(kwargs["columns"])
                
            # Check for compare_to format
            if "compare_to" in kwargs:
                all_columns.append(kwargs["compare_to"])
        
        # Verify all columns exist
        if all_columns:
            with engine.connect() as connection:
                for column_name in all_columns:
                    result = connection.execute(text("""
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name
                    """), {"table_name": rule.table_name, "column_name": column_name})
                    
                    if result.first() is None:
                        missing_columns.append(column_name)
                
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Columns {missing_columns} still don't exist in table '{rule.table_name}'. Cannot finalize draft."
            )
        
        # Update the rule
        rule.is_draft = False
        rule.confidence = 100  # User has verified the rule
        
        # Create a new version to track the change
        new_version = RuleVersion(
            rule_id=rule.id,
            version_number=len(rule.versions) + 1,
            rule_config=rule.rule_config,
            is_current=True
        )
        
        # Set all other versions as not current
        for version in rule.versions:
            version.is_current = False
            
        db.add(new_version)
        db.commit()
        db.refresh(rule)
        
        return rule
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error finalizing draft rule: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 