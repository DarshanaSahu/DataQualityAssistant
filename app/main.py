from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import uvicorn
from app.core.config import settings
from app.db.session import get_db, engine
from app.services.rule_generator import AIRuleGenerator
from app.services.quality_engine import DataQualityEngine
from app.models.rule import Rule, RuleVersion
from pydantic import BaseModel
import os
from sqlalchemy import text, inspect

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

class RuleResponse(BaseModel):
    id: int
    name: str
    description: str
    table_name: str
    rule_config: dict
    is_active: bool

    class Config:
        from_attributes = True

class RuleExecutionRequest(BaseModel):
    table_name: str
    rule_ids: List[int]

class RuleExecutionResponse(BaseModel):
    table_name: str
    execution_time: str
    results: List[dict]

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
    try:
        rules = rule_generator.generate_rules(table_name)
        created_rules = []
        
        for rule_config in rules:
            # Ensure rule_config is a valid dictionary
            if not isinstance(rule_config, dict):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        results = quality_engine.execute_rules(request.table_name, request.rule_ids)
        
        # Generate report
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir,
            f"quality_report_{request.table_name}_{results['execution_time']}.xlsx"
        )
        quality_engine.generate_report(results, report_path)
        
        return results
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

@app.get(f"{settings.API_V1_STR}/database/connect", response_model=DatabaseConnectionResponse)
async def connect_database():
    """Test database connection and return available tables."""
    try:
        # Test connection
        with engine.connect() as connection:
            # Get list of tables
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            
            return DatabaseConnectionResponse(
                status="success",
                message="Successfully connected to database",
                database_url=settings.DATABASE_URL,
                tables=tables
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to database: {str(e)}"
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 