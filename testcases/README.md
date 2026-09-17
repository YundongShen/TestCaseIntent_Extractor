# Test Cases Directory

This directory stores all test cases to be extracted.

## Directory Description

Each test case corresponds to a separate file, following the naming convention:
- `{feature_name}.test.js` or `{feature_name}.test.ts`

## Existing Test Cases

| Filename | Description | Type |
|----------|-------------|------|
| `admin-index-api.test.js` | Admin Index API integration test | JavaScript (Jest) |

## File Structure Example

```
testcases/
├── admin-index-api.test.js      # Admin Index API test
├── user-authentication.test.js  # User authentication test
├── product-catalog.test.js      # Product catalog test
└── ... (add more later)
```

## Adding New Test Cases

1. Create a new file in this directory, named `{feature_name}.test.js`
2. File content can be:
   - Jest test file
   - Test code from other testing frameworks
   - Any format of test cases

## Subsequent Processing

These test cases will be analyzed and extracted by the intent extraction system, with extraction content including:
- **Activity**: Operations executed by the test
- **Goal**: Test objectives and expected results
- **Object**: Main objects involved in the test

Reference main program: `../main.py`
