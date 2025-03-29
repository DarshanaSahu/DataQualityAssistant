#!/usr/bin/env python
"""
A simple script to run the FastAPI app directly for testing our fixes.
"""

import uvicorn
import sys

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        log_level="debug"
    ) 