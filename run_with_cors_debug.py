#!/usr/bin/env python
"""
Run the FastAPI server with CORS debugging enabled.

Usage:
    python run_with_cors_debug.py
"""

import uvicorn
import logging
import os
import sys

# Add CORS debug middleware to the app
def add_cors_debug_middleware():
    # Get the app's main module path
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "main.py")
    
    with open(app_path, "r") as f:
        lines = f.readlines()
    
    # Find where the app is defined
    app_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("app = FastAPI("):
            app_idx = i
            break
    
    if app_idx is None:
        print("Could not find FastAPI app definition in main.py")
        return False
    
    # Add CORS debug middleware code right after the app definition
    cors_debug_middleware = """
# CORS Debug Middleware
@app.middleware("http")
async def cors_debug_middleware(request, call_next):
    from fastapi.responses import JSONResponse
    import traceback
    
    # Log all incoming requests with headers
    print(f"\\n{'=' * 50}\\nIncoming request: {request.method} {request.url}\\n{'=' * 50}")
    print("Headers:")
    for name, value in request.headers.items():
        print(f"  {name}: {value}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response headers for debugging
        print(f"\\n{'=' * 50}\\nOutgoing response: {response.status_code}\\n{'=' * 50}")
        print("Response Headers:")
        for name, value in response.headers.items():
            print(f"  {name}: {value}")
        
        return response
    except Exception as e:
        # Log and return error details for debugging
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"\\n{'!' * 50}\\nERROR: {str(e)}\\n{'!' * 50}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content=error_details)

"""
    
    # Insert the middleware
    lines.insert(app_idx + 1, cors_debug_middleware)
    
    # Write back the modified file
    with open(app_path, "w") as f:
        f.writelines(lines)
    
    print("Successfully added CORS debug middleware to app/main.py")
    return True

if __name__ == "__main__":
    # Add CORS debug middleware (comment this out if you don't want to modify the file)
    add_cors_debug_middleware()
    
    # Set up logging for CORS issues
    logging.basicConfig(level=logging.DEBUG)
    
    # These specific loggers are helpful for debugging CORS issues
    logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
    logging.getLogger("fastapi").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
    
    # Print debug instructions
    print("\n" + "=" * 80)
    print("CORS DEBUG MODE ENABLED")
    print("=" * 80)
    print("""
To debug CORS issues:
1. Watch the server logs for request/response headers
2. Check for 'Access-Control-Allow-Origin' headers in responses
3. For preflight requests, look for OPTIONS requests and their responses
4. Use browser dev tools (Network tab) to inspect requests
    """)
    print("=" * 80 + "\n")
    
    # Run the server with hot reload
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8001,  # Use a different port to avoid conflicts
        reload=True, 
        log_level="debug"
    )
    
    print("Server running with CORS debugging enabled at http://localhost:8001") 