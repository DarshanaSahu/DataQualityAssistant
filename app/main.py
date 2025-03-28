from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uvicorn
from app.core.config import settings
from app.db.session import get_db
from app.services.rule_generator import AIRuleGenerator
from app.services.quality_engine import DataQualityEngine
from app.models.rule import Rule, RuleVersion
from pydantic import BaseModel
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Pydantic models for request/response
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
            rule = Rule(
                name=f"AI Generated Rule for {table_name}",
                description="Automatically generated rule based on data analysis",
                table_name=table_name,
                rule_config=rule_config
            )
            db.add(rule)
            db.commit()
            db.refresh(rule)
            created_rules.append(rule)
        
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 