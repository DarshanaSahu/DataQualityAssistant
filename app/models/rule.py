from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    table_name = Column(String, nullable=False)
    # rule_config is now an array of expectation configurations
    rule_config = Column(JSON, nullable=False)  # Now structured as a list of expectation configs
    is_active = Column(Boolean, default=True)
    is_draft = Column(Boolean, default=False)
    confidence = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    versions = relationship("RuleVersion", back_populates="rule")

class RuleVersion(Base):
    __tablename__ = "rule_versions"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(Integer, ForeignKey("rules.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    rule_config = Column(JSON, nullable=False)  # Same structure as in Rule model
    is_current = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    rule = relationship("Rule", back_populates="versions") 