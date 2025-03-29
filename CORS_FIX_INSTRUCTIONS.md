# Fixing CORS and Database Migration Instructions

This guide provides step-by-step instructions to fix the CORS errors and update the database to support multi-expectation rules.

## 1. Updates and Fixes

The changes we've made to fix the issues:

1. **CORS Configuration**
   - Updated the CORS middleware in `app/main.py` to allow more origins
   - Added explicit headers, methods, and exposed headers
   - Set a max age for CORS caching

2. **Error Handling**
   - Added robust error handling to all API endpoints
   - Ensured that endpoints return valid JSON responses instead of throwing errors
   - Added detailed logging for better troubleshooting

3. **Database Migration**
   - Created a migration script to update existing rules to the multi-expectation format

4. **Rule Suggestion Improvements**
   - Added multiple levels of error handling in the rule suggestion process
   - Ensured that responses always have a valid structure even on error
   - Added safeguards against malformed rule suggestions

## 2. Run the Database Migration

Run the migration script to update existing rules in the database:

```bash
# Make sure you're in the project root directory
python migrate_rules.py
```

## 3. Restart the Server

You have two options for restarting the server:

### Option 1: Regular Start
```bash
# Stop the current server (Ctrl+C)
# Then restart with:
uvicorn app.main:app --reload
```

### Option 2: Debug Mode (Recommended for Troubleshooting)
```bash
# Stop the current server (Ctrl+C)
# Then start with CORS debugging enabled:
python run_with_cors_debug.py

# If additional debugging is needed, uncomment line 94 in run_with_cors_debug.py:
# add_cors_debug_middleware()
```

## 4. Verify the Fixes

1. Refresh your frontend application
2. Open the browser developer tools (F12)
3. Check these specific API endpoints in the Network tab:
   - `/api/v1/rules` - Should return a list of rules
   - `/api/v1/tables/{table_name}/suggest-rules` - Should return rule suggestions
   - `/api/v1/analyze-table` - Should perform a comprehensive analysis
4. Verify no CORS errors appear in the Console tab

## 5. Common Issues and Solutions

If you still see issues:

### a) CORS Errors Still Appearing
- Make sure the server was restarted after code changes
- Check that you're using the correct URL (http://localhost:3000)
- Try clearing browser cache or using incognito mode
- Check for mismatched credentials (withCredentials in frontend vs allow_credentials in backend)

### b) 500 Errors on Specific Endpoints
- Check the server logs for detailed error messages
- Try running the server with the debug middleware enabled
- If errors occur in rule generation, check the Claude API connection
- Make sure the table exists and has proper permissions

### c) Frontend Issues
- Clear browser cache
- Ensure the frontend is making requests to the correct API URL
- Check for any client-side errors in the console

## 6. Roll Back if Needed

If issues persist and you need to roll back:

```bash
# Option 1: Restore database backup if available

# Option 2: If no backup, manually update rule_config format in the database
# Use a database client to run this SQL (adjust as needed):
# UPDATE rules SET rule_config = JSON_BUILD_OBJECT('expectation_type', rule_config->>'expectation_type', 'kwargs', rule_config->'kwargs');
```

## 7. Additional Debugging

If you need to further debug CORS issues:

1. Run the server with the CORS debug middleware:
   - Uncomment the `add_cors_debug_middleware()` line in `run_with_cors_debug.py`
   - This will log all request/response headers

2. Use the browser's Network tab to inspect:
   - OPTIONS preflight requests
   - Response headers, particularly `Access-Control-Allow-*` headers

3. Check that your frontend is sending the right headers:
   - Origin header should match one of the allowed origins
   - If using credentials, make sure withCredentials is set 