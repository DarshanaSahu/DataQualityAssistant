from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Data Quality Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/data_quality_db"
    
    # OpenAI
    ANTHROPIC_API_KEY: str
    
    # Great Expectations
    GE_DATA_DIR: str = "ge_data"
    
    class Config:
        env_file = ".env"

settings = Settings() 