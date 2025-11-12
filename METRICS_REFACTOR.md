# Metrics Refactoring Guide

## Overview

This branch (`feature/metrics`) refactors the evaluation system to properly separate query, context, and output for more accurate metric scoring.

## What Changed

### 1. **New Request Structure**

**Before:**
```json
{
  "source": "mixed query and context",
  "output": "model response",
  "metric": "faithfulness"
}
```

**After:**
```json
{
  "query": "What is Selenium?",
  "context": ["Selenium is a web automation framework"],
  "output": "Selenium is for testing",
  "metric": "faithfulness"
}
```

### 2. **Removed Auto-defaulting of `expected_output`**

**Before:** `expected_output` was automatically set to `output` when not provided, causing metrics to always pass.

**After:** `expected_output` remains `None` unless explicitly provided, making evaluations more accurate.

### 3. **Metric-Specific Validation**

Each metric now enforces its required fields:

| Metric | Required Fields | Recommended Fields |
|--------|----------------|-------------------|
| `faithfulness` | `output` | `context`, `query` |
| `answer_relevancy` | `query`, `output` | `context` |
| `contextual_precision` | `context`, `output`, `expected_output` | `query` |
| `contextual_recall` | `context`, `output`, `expected_output` | `query` |

### 4. **Strict Mode Enabled**

All metrics use `strict_mode=True` for rigorous evaluation.

## Migration Guide

### API Changes

#### Faithfulness

**Old:**
```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Selenium is a web automation framework",
    "output": "Selenium is for testing",
    "metric": "faithfulness"
  }'
```

**New:**
```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Selenium?",
    "context": ["Selenium is a web automation framework for testing"],
    "output": "Selenium is for testing",
    "metric": "faithfulness"
  }'
```

#### Answer Relevancy

**Old:**
```bash
curl -X POST http://localhost:8000/eval \
  -d '{
    "source": "Can you help write Selenium code?",
    "output": "Yes, here is code",
    "metric": "answer_relevancy"
  }'
```

**New:**
```bash
curl -X POST http://localhost:8000/eval \
  -d '{
    "query": "Can you help write Selenium code?",
    "output": "Yes, here is code: from selenium import webdriver",
    "metric": "answer_relevancy"
  }'
```

⚠️ **Note:** `query` is now **required** for `answer_relevancy`

#### Contextual Precision/Recall

**Old:**
```bash
curl -X POST http://localhost:8000/eval \
  -d '{
    "source": "Context about Selenium",
    "output": "Selenium is for testing",
    "metric": "contextual_precision"
  }'
```

**New:**
```bash
curl -X POST http://localhost:8000/eval \
  -d '{
    "query": "What is Selenium?",
    "context": [
      "Selenium is a web automation framework",
      "Python is a programming language"
    ],
    "output": "Selenium is for web testing",
    "expected_output": "Selenium is for web automation",
    "metric": "contextual_precision"
  }'
```

⚠️ **Note:** `context` (as array) and `expected_output` are now **required**

## Why These Changes?

### Problem 1: Mixed Query and Context

**Before:** Using `source` for both query and context made faithfulness scoring weak.

**Example:**
```json
{
  "source": "Can you write Selenium code?",
  "output": "Hello Babu, all good",
  "metric": "faithfulness"
}
```
**Result:** Score = 1.0 ❌ (Wrong! Output doesn't address the query)

**After:** Separate `query` and `context` fields.
```json
{
  "query": "Can you write Selenium code?",
  "context": [],
  "output": "Hello Babu, all good",
  "metric": "answer_relevancy"
}
```
**Result:** Score = 0.1-0.2 ✅ (Correct! Output is irrelevant)

### Problem 2: Auto-defaulting expected_output

**Before:** `expected_output = output` made comparisons meaningless.

```python
# This would always pass:
if expected_output == output:  # Always true when auto-set!
    return 1.0
```

**After:** `expected_output` is only used when explicitly provided.

### Problem 3: No Field Validation

**Before:** Metrics would run with missing data, giving misleading scores.

**After:** Clear error messages:
```
ValueError: answer_relevancy requires 'query' (the user's question)
ValueError: contextual_precision requires 'context' (list of retrieved passages)
```

## Testing the Changes

### Test 1: Faithfulness (Correct Context)

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "context": ["Selenium WebDriver automates web browsers"],
    "output": "Selenium is used for browser automation",
    "metric": "faithfulness"
  }'
```

**Expected:** High score (0.9-1.0) - output aligns with context

### Test 2: Faithfulness (Hallucination)

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "context": ["Selenium WebDriver automates web browsers"],
    "output": "Selenium can automate mobile apps and desktop applications",
    "metric": "faithfulness"
  }'
```

**Expected:** Low score (0.0-0.4) - output invents mobile/desktop automation

### Test 3: Answer Relevancy (Good)

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I write Selenium code?",
    "output": "Here is a basic example: from selenium import webdriver; driver = webdriver.Chrome()",
    "metric": "answer_relevancy"
  }'
```

**Expected:** High score (0.9-1.0) - directly answers the question

### Test 4: Answer Relevancy (Off-topic)

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I write Selenium code?",
    "output": "As you mentioned earlier, you are automating Salesforce",
    "metric": "answer_relevancy"
  }'
```

**Expected:** Low score (0.1-0.3) - doesn't provide code or answer question

### Test 5: Contextual Precision

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Selenium?",
    "context": [
      "Selenium is a web automation testing framework",
      "Python is a programming language",
      "JavaScript is used for web development"
    ],
    "output": "Selenium is a web testing framework",
    "expected_output": "Selenium is used for web testing",
    "metric": "contextual_precision"
  }'
```

**Expected:** Medium score (0.6-0.7) - first context item is relevant, others aren't

## Backward Compatibility

⚠️ **Breaking Changes:**

1. `source` field is removed - use `query` and/or `context` instead
2. `context` is now an array of strings, not a single string
3. `answer_relevancy` now requires `query`
4. `contextual_*` metrics now require `context` and `expected_output`

## Benefits

✅ **More Accurate Scoring** - Proper separation of concerns
✅ **Better Debugging** - Clear error messages for missing fields
✅ **Strict Evaluation** - All metrics penalize hallucinations and irrelevance
✅ **Production Ready** - Reflects real-world quality standards
✅ **Self-Documenting** - Field names clearly indicate their purpose

## API Documentation

Visit `http://localhost:8000/metrics-info` for complete field requirements per metric.

## Next Steps

1. **Update Node.js client** to use new request structure
2. **Update tests** to use proper query/context/output separation
3. **Add integration tests** for each metric type
4. **Document real-world examples** in README

---

**Questions?** Check `STRICT_MODE.md` for evaluation details.
