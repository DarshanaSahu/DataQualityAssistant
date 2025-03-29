# Data Quality Assistant

A FastAPI-based application for monitoring data quality through rule-based validation. The application uses AI to suggest data quality rules and provides a comprehensive interface for managing and monitoring data quality.

## Features

- **Data Quality Rule Management**: Create, update, and manage rules for validating data quality.
- **AI-Powered Rule Suggestions**: Automatically generate rule suggestions based on table data.
- **Data Analysis**: Analyze table schema and sample data to understand patterns and anomalies.
- **User-Friendly Error Handling**: Standardized error responses with clear, actionable messages.
- **CORS Support**: Built-in support for cross-origin resource sharing.

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL or other compatible SQL database
- Dependencies listed in requirements.txt

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd DataQualityAssistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Copy `.env.example` to `.env` and update the variables with your configuration.

4. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Error Handling

The application implements a standardized error handling system that provides:

- User-friendly error messages
- Consistent error response structure
- Appropriate HTTP status codes
- Detailed logging for troubleshooting

For more details on the error handling implementation, see [ERROR_HANDLING_IMPROVEMENTS.md](docs/ERROR_HANDLING_IMPROVEMENTS.md).

## Testing

Two testing scripts are provided:

1. `testing_error_handling.sh`: Tests API endpoints with various error scenarios to verify error handling.

2. Run the error handling tests:
   ```
   ./testing_error_handling.sh
   ```

## Deployment

To deploy the error handling improvements, you can use the provided script:

```
./apply_error_handling_improvements.sh
```

This script:
- Backs up original files
- Creates the error handling module
- Copies documentation
- Attempts to restart the service

## License

[License information]

## Contributing

[Contribution guidelines] 