from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.rule import Base
from app.core.config import settings

# Import all models here to ensure they are registered with SQLAlchemy
from app.models import *  # noqa: F403

def init_db():
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db() 