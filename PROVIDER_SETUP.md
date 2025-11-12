# LLM Provider Configuration Guide

This project supports **two LLM providers** for evaluation: **Groq** and **OpenAI**. You can easily switch between them by changing the `EVAL_MODEL` environment variable.

## Quick Start

### Using Groq (Default)
```bash
# In .env file:
GROQ_API_KEY=gsk_your_groq_api_key_here
EVAL_MODEL=llama-3.3-70b-versatile
```

### Using OpenAI
```bash
# In .env file:
OPENAI_API_KEY=sk-your_openai_api_key_here
EVAL_MODEL=gpt-4o-mini
```

**That's it!** The code automatically detects which provider to use based on the model name.

---

## How It Works

The `init_evaluator_from_env()` function automatically determines the provider:

```python
evaluator = init_evaluator_from_env()
```

**Detection Logic:**
- If `EVAL_MODEL` starts with `llama-`, `mixtral-`, `gemma`, or `qwen` → **Uses Groq API**
- If `EVAL_MODEL` starts with `gpt-` → **Uses OpenAI API**

No code changes needed! Just update `.env` and restart the server.

---

## Supported Models

### Groq Models (Fast & Cost-Effective)
| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `llama-3.3-70b-versatile` | Medium | High | **Recommended default** |
| `llama-3.1-70b-versatile` | Medium | High | Alternative large model |
| `llama-3.1-8b-instant` | Very Fast | Good | Quick evaluations |
| `mixtral-8x7b-32768` | Fast | High | Long context support |
| `gemma2-9b-it` | Fast | Good | Smaller footprint |

### OpenAI Models (Production-Grade)
| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `gpt-4o-mini` | Fast | Very High | **Recommended for stricter scoring** |
| `gpt-4o` | Medium | Excellent | Best accuracy |
| `gpt-4-turbo` | Medium | Excellent | Long context |
| `gpt-3.5-turbo` | Very Fast | Good | Budget option |

---

## Evaluation Quality Comparison

### Groq Models
- ✅ **Pros:**
  - Very fast inference
  - Cost-effective
  - Good for development/testing
  - Strict mode enabled with custom system prompts
  
- ⚠️ **Considerations:**
  - May be more lenient on edge cases
  - Custom strict mode system messages help improve strictness

### OpenAI Models
- ✅ **Pros:**
  - **Stricter evaluation out of the box**
  - More consistent scoring
  - Better at detecting subtle hallucinations
  - Production-tested reliability
  
- ⚠️ **Considerations:**
  - Higher cost per evaluation
  - Slightly slower than Groq

---

## When to Use Each Provider

### Use Groq When:
- 🚀 **Development & Testing:** Fast iteration cycles
- 💰 **Budget Constraints:** Cost-effective alternative
- ⚡ **High Volume:** Need to process many evaluations quickly
- 🧪 **Experimentation:** Testing different prompts/configurations

### Use OpenAI When:
- 🎯 **Production Deployment:** Need maximum reliability
- 📊 **Strict Scoring Required:** Evaluating production LLM outputs
- 🔍 **Quality Assurance:** Detecting subtle hallucinations/errors
- 💼 **Customer-Facing:** Results need to be highly accurate

---

## Configuration Examples

### Example 1: Development with Groq
```bash
# .env
GROQ_API_KEY=gsk_your_groq_api_key_here
EVAL_MODEL=llama-3.3-70b-versatile
```

### Example 2: Production with OpenAI
```bash
# .env
OPENAI_API_KEY=sk-your_openai_api_key_here
EVAL_MODEL=gpt-4o-mini
```

### Example 3: Fast Testing with Groq
```bash
# .env
GROQ_API_KEY=gsk_your_groq_api_key_here
EVAL_MODEL=llama-3.1-8b-instant  # Fastest option
```

### Example 4: Maximum Quality with OpenAI
```bash
# .env
OPENAI_API_KEY=sk-your_openai_api_key_here
EVAL_MODEL=gpt-4o  # Best quality
```

---

## Testing the Configuration

### 1. Start the Server
```bash
python deepeval_server.py
```

### 2. Check Which Provider is Active
Look for log messages:
```
Using Groq API for evaluation with model: llama-3.3-70b-versatile
# or
Using OpenAI API for evaluation with model: gpt-4o-mini
```

### 3. Run Smoke Test
```bash
curl http://localhost:8000/example
```

This will evaluate a fixed example and show:
- Active configuration
- Score from the selected provider
- Confirmation that the setup works

---

## Switching Providers Mid-Project

You can switch providers at any time:

1. **Stop** the Python server (Ctrl+C)
2. **Update** `.env` file:
   ```bash
   # Change from:
   EVAL_MODEL=llama-3.3-70b-versatile
   # To:
   EVAL_MODEL=gpt-4o-mini
   # And add:
   OPENAI_API_KEY=sk-...
   ```
3. **Restart** the server:
   ```bash
   python deepeval_server.py
   ```

**Zero code changes required!** The evaluator automatically adapts.

---

## Cost Comparison

### Groq Pricing (Approximate)
- **Free tier:** Generous limits for development
- **Paid tier:** ~$0.05-0.10 per 1M tokens
- **Evaluation cost:** Very low (<$0.01 per 100 evaluations)

### OpenAI Pricing (Approximate)
- **gpt-4o-mini:** ~$0.15/$0.60 per 1M tokens (input/output)
- **gpt-4o:** ~$2.50/$10.00 per 1M tokens (input/output)
- **Evaluation cost:** Moderate (~$0.05-0.20 per 100 evaluations)

*Exact pricing varies by usage and provider updates. Check official pricing pages.*

---

## Troubleshooting

### Error: "GROQ_API_KEY environment variable is required"
- **Solution:** You set `EVAL_MODEL=llama-3.3-70b-versatile` but didn't provide `GROQ_API_KEY`
- **Fix:** Add `GROQ_API_KEY=gsk_...` to `.env`

### Error: "OPENAI_API_KEY environment variable is required"
- **Solution:** You set `EVAL_MODEL=gpt-4o-mini` but didn't provide `OPENAI_API_KEY`
- **Fix:** Add `OPENAI_API_KEY=sk-...` to `.env`

### Warning: "EVAL_MODEL 'xyz' is not a valid GPT model"
- **Solution:** You tried to use a non-GPT model name without proper provider detection
- **Fix:** Use valid model names from the tables above

### Scores Too Lenient with Groq
- **Solution 1:** Switch to OpenAI for stricter evaluation:
  ```bash
  EVAL_MODEL=gpt-4o-mini
  OPENAI_API_KEY=sk-...
  ```
- **Solution 2:** The custom `GroqModel` includes strict system prompts to improve strictness
- **Solution 3:** Use `gpt-4o` for maximum strictness

### Scores Too Strict with OpenAI
- **Solution:** Consider using Groq models if OpenAI is rejecting valid outputs
- **Note:** OpenAI's strictness is generally desirable for production evaluation

---

## Best Practices

1. **Development:** Use Groq for fast iteration
2. **CI/CD:** Use Groq for automated testing (cost-effective)
3. **Production:** Use OpenAI for final quality checks
4. **A/B Testing:** Run both providers and compare results
5. **Cost Optimization:** Use Groq for bulk evaluations, OpenAI for critical cases

---

## Advanced: Hybrid Approach

You can use **both providers** in parallel:

```python
# Initialize both evaluators
groq_evaluator = MetricEvaluator(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    use_groq=True
)

openai_evaluator = MetricEvaluator(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    use_groq=False
)

# Compare scores
groq_score, _ = groq_evaluator.evaluate(...)
openai_score, _ = openai_evaluator.evaluate(...)

print(f"Groq: {groq_score}, OpenAI: {openai_score}")
```

This is useful for:
- **Validation:** Ensure consistent results across providers
- **Quality Checks:** Use OpenAI as "ground truth" reference
- **Research:** Compare scoring behaviors

---

## Summary

| Aspect | Groq | OpenAI |
|--------|------|--------|
| **Setup** | Set `EVAL_MODEL=llama-*` | Set `EVAL_MODEL=gpt-*` |
| **Speed** | ⚡ Very Fast | 🐢 Moderate |
| **Cost** | 💰 Low | 💰💰 Higher |
| **Strictness** | 🔍 Good (with custom prompts) | 🔍🔍 Excellent |
| **Use Case** | Development, Testing | Production, QA |

**The best part?** You can switch anytime with **zero code changes**! 🎉
