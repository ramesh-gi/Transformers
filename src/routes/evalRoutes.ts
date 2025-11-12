import { Router, Request, Response, NextFunction } from "express";
import { callLLM } from "../services/llmClient.js";
import { evalWithMetric, evalFaithfulness } from "../services/evalClient.js";
import { retrieveContext } from "../services/ragService.js";
import { ENV } from "../config/env.js";

const router = Router();

/**
 * Error handler middleware for async routes
 */
const asyncHandler =
  (fn: (req: Request, res: Response) => Promise<any>) =>
  (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res)).catch(next);
  };

/**
 * POST /api/llm/eval
 * LLM-only evaluation endpoint
 *
 * Request body:
 * {
 *   prompt: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'faithfulness')
 * }
 *
 * Response:
 * {
 *   prompt: string,
 *   model: string,
 *   provider: string,
 *   llmResponse: string,
 *   metrics: { faithfulness?: number }
 * }
 */
router.post(
  "/llm/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { prompt, model, temperature, metric } = req.body;

    // Validation
    if (!prompt) {
      return res.status(400).json({
        error: "Missing required field: prompt"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "faithfulness";

    // Call LLM
    const llmResponse = await callLLM(prompt, effectiveModel, effectiveTemperature);
    console.log("LLM Response:", llmResponse);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // Evaluate with Deepeval using specified metric
    const evalResult = await evalWithMetric(prompt, llmResponse, effectiveMetric, provider);
    console.log("Evaluation Result:", evalResult);

    res.json({
      prompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation
      }
    });
  })
);

/**
 * POST /api/rag/eval
 * RAG + LLM evaluation endpoint
 *
 * Request body:
 * {
 *   query: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'faithfulness')
 * }
 *
 * Response:
 * {
 *   query: string,
 *   context: string,
 *   prompt: string,
 *   llmResponse: string,
 *   metrics: { faithfulness?: number }
 * }
 */
router.post(
  "/rag/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { query, model, temperature, metric } = req.body;

    // Validation
    if (!query) {
      return res.status(400).json({
        error: "Missing required field: query"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "faithfulness";

    // 1. Retrieve context from RAG
    const context = await retrieveContext(query);

    // 2. Build RAG prompt
    const ragPrompt = `You are a helpful QA assistant. Using ONLY the following context, answer the question as accurately as possible. If the context does not contain the answer, say "I don't have enough information to answer that."

CONTEXT:
${context}

QUESTION:
${query}

ANSWER:`;

    // 3. Call LLM with RAG prompt
    const llmResponse = await callLLM(ragPrompt, effectiveModel, effectiveTemperature);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // 4. Evaluate using specified metric: source = context, output = llmResponse
    const evalResult = await evalWithMetric(context, llmResponse, effectiveMetric, provider);

    res.json({
      query,
      context,
      prompt: ragPrompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation
      }
    });
  })
);

/**
 * GET /health
 * Health check endpoint
 */
router.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString()
  });
});

/**
 * POST /eval-only
 * Evaluate existing query-output pairs without LLM generation
 * 
 * Request body:
 * {
 *   query: string (required) - the input question/prompt,
 *   output: string (required) - the response to evaluate,
 *   context?: string (optional) - context for faithfulness evaluation,
 *   metric?: string (optional, defaults to 'faithfulness')
 * }
 *
 * Response:
 * {
 *   query: string,
 *   output: string,
 *   context?: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/eval-only",
  asyncHandler(async (req: Request, res: Response) => {
    const { query, output, context, metric } = req.body;

    // Validation
    if (!query) {
      return res.status(400).json({
        error: "Missing required field: query"
      });
    }

    if (!output) {
      return res.status(400).json({
        error: "Missing required field: output"
      });
    }

    // Determine effective parameters
    const effectiveMetric = metric || "faithfulness";
    const source = context || query; // Use context if provided, otherwise use query as source

    console.log(`Direct evaluation - Metric: ${effectiveMetric}`);
    console.log(`Query: ${query}`);
    console.log(`Output: ${output}`);
    console.log(`Source: ${source}`);

    // Evaluate using specified metric (no LLM generation needed)
    // Default to groq provider for evaluation
    const evalResult = await evalWithMetric(source, output, effectiveMetric, "groq");

    const response: any = {
      query,
      output,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation
      }
    };

    // Include context in response if it was provided
    if (context) {
      response.context = context;
    }

    res.json(response);
  })
);

/**
 * GET /metrics
 * Get available evaluation metrics for training
 */
router.get("/metrics", async (req: Request, res: Response) => {
  try {
    // Fetch metrics info from Deepeval service
    const response = await fetch(`${ENV.DEEPEVAL_URL.replace('/eval', '/metrics-info')}`);
    const metricsInfo = await response.json();
    
    res.json({
      ...metricsInfo,
      usage_examples: {
        faithfulness: "Measures alignment with provided context - ideal for RAG systems",
        answer_relevancy: "Measures how well the answer addresses the question - good for QA systems", 
        contextual_precision: "Measures precision of retrieved context - useful for retrieval evaluation",
        contextual_recall: "Measures coverage of expected information - helpful for completeness checking"
      }
    });
  } catch (error) {
    res.status(500).json({
      error: "Could not fetch metrics information",
      available_metrics: ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]
    });
  }
});

export default router;
