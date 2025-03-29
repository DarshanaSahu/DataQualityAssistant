# Data Quality Assistant API Documentation

This document describes the APIs for managing data quality rules, with a focus on the new capabilities for multi-column and multi-expectation rules.

## Updated Database Schema

The database schema has been updated to support multiple expectations per rule:

* **rules** table:
  * `rule_config` column is now structured as an array/list of expectation configurations

## API Endpoints for Multi-Expectation Rules

### 1. GET `/api/v1/tables/{table_name}/suggest-rules`

Analyzes a table and suggests new rules or updates to existing rules, including multi-column and multi-expectation rules.

#### Request:
- **Method**: GET
- **URL Parameter**: `table_name` - The name of the database table to analyze
- **No request body required**

#### Response:
```json
{
  "table_name": "orders",
  "new_rule_suggestions": [
    {
      "rule_type": "expect_column_values_to_not_be_null",
      "column": "customer_id",
      "columns": null,
      "description": "Customer ID should not be null as it's a required field for order processing.",
      "rule_config": [
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "customer_id"
          }
        }
      ],
      "confidence": 95
    },
    {
      "rule_type": "multiple_date_validations",
      "column": null,
      "columns": ["start_date", "end_date"],
      "description": "Ensure dates are valid and properly ordered",
      "rule_config": [
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "start_date"
          }
        },
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "end_date"
          }
        },
        {
          "expectation_type": "expect_column_values_to_be_greater_than_other_column",
          "kwargs": {
            "column": "end_date",
            "compare_to": "start_date"
          }
        }
      ],
      "confidence": 98
    }
  ],
  "rule_update_suggestions": [
    {
      "rule_id": 123,
      "current_config": "{'expectation_type': 'expect_column_values_to_be_in_set', 'kwargs': {'column': 'status', 'value_set': ['pending', 'shipped']}}",
      "suggested_config": [
        {
          "expectation_type": "expect_column_values_to_be_in_set",
          "kwargs": {
            "column": "status",
            "value_set": ["pending", "shipped", "delivered", "cancelled"],
            "mostly": 0.98
          }
        }
      ],
      "reason": "Found additional status values in the data: 'delivered' and 'cancelled'",
      "confidence": 85
    }
  ],
  "error": null
}
```

### 2. POST `/api/v1/tables/{table_name}/apply-suggested-rules`

Applies selected rule suggestions (both single-column and multi-column, single-expectation and multi-expectation) to create or update rules.

#### Request:
```json
{
  "table_name": "orders",
  "new_rule_ids": [0, 1],
  "update_rule_ids": [0]
}
```
- **Method**: POST
- **URL Parameter**: `table_name` - The name of the database table
- **Body Parameters**:
  - `new_rule_ids`: Array of indexes from the `new_rule_suggestions` list to apply
  - `update_rule_ids`: Array of indexes from the `rule_update_suggestions` list to apply

#### Response:
```json
{
  "created_rules": [
    {
      "id": 456,
      "name": "AI Suggested Rule - Customer ID should not be null",
      "description": "Customer ID should not be null as it's a required field for order processing.",
      "table_name": "orders",
      "rule_config": [
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "customer_id"
          }
        }
      ],
      "is_active": true,
      "is_draft": false,
      "confidence": 95,
      "created_at": "2023-06-15T10:30:45.123456",
      "updated_at": "2023-06-15T10:30:45.123456"
    },
    {
      "id": 458,
      "name": "AI Suggested Rule for start_date, end_date",
      "description": "Ensure dates are valid and properly ordered",
      "table_name": "orders",
      "rule_config": [
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "start_date"
          }
        },
        {
          "expectation_type": "expect_column_values_to_not_be_null",
          "kwargs": {
            "column": "end_date"
          }
        },
        {
          "expectation_type": "expect_column_values_to_be_greater_than_other_column",
          "kwargs": {
            "column": "end_date",
            "compare_to": "start_date"
          }
        }
      ],
      "is_active": true,
      "is_draft": false,
      "confidence": 98,
      "created_at": "2023-06-15T10:30:45.123456",
      "updated_at": "2023-06-15T10:30:45.123456"
    }
  ],
  "updated_rules": [
    {
      "id": 123,
      "name": "Status Value Check",
      "description": "Checks that status values are valid",
      "table_name": "orders",
      "rule_config": [
        {
          "expectation_type": "expect_column_values_to_be_in_set",
          "kwargs": {
            "column": "status",
            "value_set": ["pending", "shipped", "delivered", "cancelled"],
            "mostly": 0.98
          }
        }
      ],
      "is_active": true,
      "is_draft": false,
      "confidence": 85,
      "created_at": "2023-06-01T15:22:33.123456",
      "updated_at": "2023-06-15T10:30:45.123456"
    }
  ],
  "errors": []
}
```

### 3. POST `/api/v1/analyze-table`

Performs a comprehensive analysis of a table including schema, data statistics, multi-column relationships, rule suggestions, and can automatically apply high-confidence suggestions.

#### Request:
```json
{
  "table_name": "orders",
  "apply_suggestions": false
}
```
- **Method**: POST
- **Body Parameters**:
  - `table_name`: The name of the database table to analyze
  - `apply_suggestions`: Boolean flag to automatically apply high-confidence suggestions (optional, defaults to false)

#### Response:
Response structure is similar to before, but now the `rule_config` fields in rule_suggestions and existing_rules will contain arrays of expectation configurations.

### 4. POST `/api/v1/rules/generate-from-description`

Generates a rule with one or more expectations based on a natural language description.

#### Request:
```json
{
  "table_name": "orders",
  "rule_name": "Date Range Validation",
  "rule_description": "The end_date should always be valid, not null, and after the start_date"
}
```
- **Method**: POST
- **Body Parameters**:
  - `table_name`: The name of the database table
  - `rule_name`: Optional custom name for the rule
  - `rule_description`: Natural language description of the rule to generate

#### Response:
```json
{
  "id": 459,
  "name": "Date Range Validation",
  "description": "The end_date should always be valid, not null, and after the start_date",
  "table_name": "orders",
  "rule_config": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "end_date"
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_greater_than_other_column",
      "kwargs": {
        "column": "end_date",
        "compare_to": "start_date"
      }
    }
  ],
  "is_active": true,
  "is_draft": false,
  "confidence": 95,
  "created_at": "2023-06-15T14:25:30.123456",
  "updated_at": "2023-06-15T14:25:30.123456"
}
```

### 5. GET `/api/v1/rules/{rule_id}`

Gets detailed information about a rule, including all columns involved in all expectations.

#### Request:
- **Method**: GET
- **URL Parameter**: `rule_id` - The ID of the rule to retrieve
- **No request body required**

#### Response:
```json
{
  "id": 459,
  "name": "Date Range Validation",
  "description": "The end_date should always be valid, not null, and after the start_date",
  "table_name": "orders",
  "rule_config": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "end_date"
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_greater_than_other_column",
      "kwargs": {
        "column": "end_date",
        "compare_to": "start_date"
      }
    }
  ],
  "is_active": true,
  "is_draft": false,
  "confidence": 95,
  "created_at": "2023-06-15T14:25:30.123456",
  "updated_at": "2023-06-15T14:25:30.123456",
  "columns": ["end_date", "start_date"],
  "versions": []
}
```

### 6. PUT `/api/v1/rules/{rule_id}`

Updates a rule, including its expectations.

#### Request:
```json
{
  "name": "Updated Rule Name",
  "description": "Updated description",
  "rule_config": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "end_date"
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_greater_than_other_column",
      "kwargs": {
        "column": "end_date",
        "compare_to": "start_date",
        "mostly": 0.99
      }
    }
  ],
  "is_active": true,
  "finalize_draft": true
}
```
- **Method**: PUT
- **URL Parameter**: `rule_id` - The ID of the rule to update
- **Body Parameters**:
  - `name`: Updated rule name (optional)
  - `description`: Updated rule description (optional)
  - `rule_config`: Updated array of expectation configurations (optional)
  - `is_active`: Whether the rule should be active (optional)
  - `finalize_draft`: Whether to finalize a draft rule (optional)

#### Response:
Returns the updated rule with the same structure as the GET endpoint.

### 7. PUT `/api/v1/rules/{rule_id}/finish-draft`

Finalizes a draft rule after user modifications, checking that all referenced columns exist.

#### Request:
- **Method**: PUT
- **URL Parameter**: `rule_id` - The ID of the rule to finalize
- **No request body required**

#### Response:
Returns the finalized rule with the same structure as the GET endpoint.

## Multi-Expectation Rule Types

The system now supports rules that combine multiple expectations:

1. **Date Validation Rules**:
   ```json
   {
     "rule_config": [
       {
         "expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {
           "column": "date_column"
         }
       },
       {
         "expectation_type": "expect_column_values_to_be_of_type",
         "kwargs": {
           "column": "date_column",
           "type_": "DATE"
         }
       }
     ]
   }
   ```

2. **Primary Key Validation Rules**:
   ```json
   {
     "rule_config": [
       {
         "expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {
           "column": "id_column"
         }
       },
       {
         "expectation_type": "expect_column_values_to_be_unique",
         "kwargs": {
           "column": "id_column"
         }
       }
     ]
   }
   ```

3. **Foreign Key Relationship Rules**:
   ```json
   {
     "rule_config": [
       {
         "expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {
           "column": "customer_id",
           "mostly": 0.99
         }
       },
       {
         "expectation_type": "expect_column_values_to_be_in_set",
         "kwargs": {
           "column": "customer_id",
           "value_set": {
             "query": "SELECT id FROM customers"
           }
         }
       }
     ]
   }
   ```

4. **Complex Date Range Rules**:
   ```json
   {
     "rule_config": [
       {
         "expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {
           "column": "start_date"
         }
       },
       {
         "expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {
           "column": "end_date"
         }
       },
       {
         "expectation_type": "expect_column_values_to_be_greater_than_other_column",
         "kwargs": {
           "column": "end_date",
           "compare_to": "start_date"
         }
       }
     ]
   }
   ``` 