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

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

class RuleCreate(BaseModel):
    name: str
    description: str
    table_name: str
    rule_config: dict

class NaturalLanguageRuleCreate(BaseModel):
    table_name: str
    rule_description: str
    rule_name: str = "Natural Language Generated Rule"

class RuleResponse(BaseModel):
    id: int
    name: str
    description: str
    table_name: str
    rule_config: dict
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
    rule_config: dict | None = None
    is_active: bool | None = None
    finalize_draft: bool = False  # Option to finalize a draft rule

class RuleClarificationResponse(BaseModel):
    rule_config: dict
    needs_clarification: bool
    clarification_questions: List[str]
    confidence: int | None = None

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
    rules = rule_generator.generate_rules(table_name)
    print(rules)
    created_rules = []
    
    for rule_config in rules:
        # Ensure rule_config is a valid dictionary
        if not isinstance(rule_config, dict):
            continue
            
        # Check if a similar rule already exists
        existing_rule = db.query(Rule).filter(
            Rule.table_name == table_name,
            cast(Rule.rule_config['expectation_type'], String) == rule_config.get("expectation_type", ""),
            cast(Rule.rule_config['kwargs'], String) == str(rule_config.get("kwargs", {}))
        ).first()
        
        if existing_rule:
            print(f"Skipping duplicate rule: {rule_config}")
            continue
            
        # Create a new rule with proper structure
        rule = Rule(
            name=f"AI Generated Rule for {table_name}",
            description="Automatically generated rule based on data analysis",
            table_name=table_name,
            rule_config={
                "expectation_type": rule_config.get("expectation_type", ""),
                "kwargs": rule_config.get("kwargs", {})
            },
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

@app.get(f"{settings.API_V1_STR}/rules", response_model=List[RuleResponse])
async def list_rules(db: Session = Depends(get_db)):
    """List all rules."""
    rules = db.query(Rule).all()
    return rules

@app.get(f"{settings.API_V1_STR}/rules/{{rule_id}}", response_model=RuleResponse)
async def get_rule(rule_id: int, db: Session = Depends(get_db)):
    """Get a specific rule by ID."""
    rule = db.query(Rule).filter(Rule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule

@app.put(f"{settings.API_V1_STR}/rules/{{rule_id}}", response_model=RuleResponse)
async def update_rule(
    rule_id: int,
    rule_update: RuleUpdate,
    db: Session = Depends(get_db)
):
    """Update a specific rule."""
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        # Update fields if provided
        if rule_update.name is not None:
            rule.name = rule_update.name
        if rule_update.description is not None:
            rule.description = rule_update.description
        if rule_update.rule_config is not None:
            rule.rule_config = rule_update.rule_config
        if rule_update.is_active is not None:
            rule.is_active = rule_update.is_active
            
        # Check if we're finalizing a draft rule
        if rule.is_draft and rule_update.finalize_draft:
            # Verify column exists if this is a draft rule being finalized
            column_name = rule.rule_config.get("kwargs", {}).get("column")
            if column_name:
                with engine.connect() as connection:
                    result = connection.execute(text("""
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name
                    """), {"table_name": rule.table_name, "column_name": column_name})
                    column_exists = result.first() is not None
                    
                if not column_exists:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Column '{column_name}' doesn't exist in table '{rule.table_name}'. Cannot finalize draft."
                    )
            
            # If we get here, we can finalize the draft
            rule.is_draft = False
            rule.confidence = 100  # User has verified the rule
        
        # Create a new version of the rule
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
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error updating rule: {str(e)}"
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
        rules = rule_generator.generate_rule_from_description(
            table_name=rule_data.table_name,
            rule_description=rule_data.rule_description
        )
        
        if not rules:
            raise HTTPException(
                status_code=400,
                detail="Could not generate valid rule from description"
            )
        
        rule_config = rules[0]
        confidence = rule_config.get("confidence", 0)
        
        # Enhanced check for column existence in rule
        column_name = rule_config.get("kwargs", {}).get("column")
        column_exists = True
        if column_name:
            # Verify the column exists in the table
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name
                """), {"table_name": rule_data.table_name, "column_name": column_name})
                column_exists = result.first() is not None
        
        # Determine if the rule should be in draft mode
        # Rules are in draft if: column doesn't exist, confidence is low, or other parameters are missing
        is_draft = False
        if not column_exists or confidence < 70:
            is_draft = True
        
        # For columns that don't exist, save in draft mode
        if not column_exists:
            print(f"Warning: Column '{column_name}' doesn't exist in table '{rule_data.table_name}'. Saving as draft.")
        
        # Check if a similar rule already exists
        existing_rule = db.query(Rule).filter(
            Rule.table_name == rule_data.table_name,
            cast(Rule.rule_config['expectation_type'], String) == rule_config.get("expectation_type", ""),
            cast(Rule.rule_config['kwargs'], String) == str(rule_config.get("kwargs", {}))
        ).first()
        
        if existing_rule:
            raise HTTPException(
                status_code=400,
                detail=f"A similar rule already exists: {existing_rule.name} (ID: {existing_rule.id})"
            )
        
        # Create the rule in database - exclude the confidence metadata
        clean_rule_config = {
            "expectation_type": rule_config.get("expectation_type", ""),
            "kwargs": rule_config.get("kwargs", {})
        }
        
        rule = Rule(
            name=rule_data.rule_name or f"Rule for {rule_data.table_name}: {clean_rule_config.get('expectation_type', '')}",
            description=rule_data.rule_description,
            table_name=rule_data.table_name,
            rule_config=clean_rule_config,
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
            
        # Check if the column in the rule exists
        column_name = rule.rule_config.get("kwargs", {}).get("column")
        if column_name:
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name
                """), {"table_name": rule.table_name, "column_name": column_name})
                column_exists = result.first() is not None
                
            if not column_exists:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Column '{column_name}' still doesn't exist in table '{rule.table_name}'. Cannot finalize draft."
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