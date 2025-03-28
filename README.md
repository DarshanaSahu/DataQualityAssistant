# AI-Powered Data Quality Assistant

An intelligent data quality management system that leverages Claude AI to generate and manage data quality rules for PostgreSQL databases using Great Expectations.

## Features

### 1. AI Rule Generator
- Automatic table schema and sample data analysis
- Claude AI-powered Great Expectations rule generation
- Natural language to Great Expectations configuration conversion
- Rule management and versioning
- Outdated rule detection and recommendations

### 2. Data Quality Engine
- Great Expectations framework integration
- Automated rule execution
- Comprehensive quality reports in Excel format

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
ANTHROPIC_API_KEY=your_anthropic_api_key
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
.
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core functionality
│   ├── models/         # Data models
│   ├── services/       # Business logic
│   └── utils/          # Utility functions
├── tests/              # Test files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## API Documentation

Once the application is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 