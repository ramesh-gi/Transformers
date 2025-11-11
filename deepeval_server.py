#!/usr/bin/env python3
"""
Deepeval FastAPI Sidecar Server
This runs separately from the Node.js server and provides LLM evaluation metrics.

Installation:
  pip install fastapi uvicorn deepeval

Usage:
  python deepeval_server.py
  # or
  uvicorn deepeval_server:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deepeval Evaluation Service",
    description="FastAPI sidecar for LLM evaluation using Deepeval",
    version="1.0.0"
)

# Add CORS middleware to allow Node.js calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvalRequest(BaseModel):
    """Request body for evaluation"""
    source: str  # context or prompt
    output: str  # model response
    provider: Optional[str] = None  # LLM provider: 'openai'
    metric: Optional[str] = "faithfulness"  # metric to evaluate: 'faithfulness', 'answer_relevancy', 'contextual_precision', 'contextual_recall'
    expected_output: Optional[str] = None  # expected/reference output for comparison (required for some metrics)


class EvalResponse(BaseModel):
    """Response with evaluation metrics"""
    metric_name: str
    score: Optional[float] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class MetricEvaluator:
    """Enterprise-grade metric evaluation system for training purposes."""
    
    SUPPORTED_METRICS = {
        "faithfulness": "Evaluates if the output is faithful to the source context",
        "answer_relevancy": "Evaluates how relevant the answer is to the input question", 
        "contextual_precision": "Evaluates the precision of retrieval in RAG systems (requires expected_output)",
        "contextual_recall": "Evaluates the recall of retrieval in RAG systems (requires expected_output)"
    }
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """Initialize the evaluator with OpenAI credentials."""
        if not openai_api_key or openai_api_key == "your-openai-api-key-here":
            raise ValueError("Valid OPENAI_API_KEY is required")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        from deepeval.models import GPTModel
        self.model = GPTModel(model=model_name)
        self.model_name = model_name
    
    def validate_metric(self, metric_name: str) -> bool:
        """Validate if the requested metric is supported."""
        return metric_name.lower() in self.SUPPORTED_METRICS
    
    def create_test_case(self, source: str, output: str, query: Optional[str] = None, expected_output: Optional[str] = None):
        """Create a standardized test case for evaluation."""
        from deepeval.test_case import LLMTestCase
        
        # For metrics that require expected_output but don't have one provided,
        # use the actual output as a reference point
        if expected_output is None:
            expected_output = output
        
        return LLMTestCase(
            input=query or source,
            actual_output=output,
            retrieval_context=[source],
            expected_output=expected_output
        )
    
    def evaluate_faithfulness(self, test_case) -> tuple[float, str]:
        """Evaluate faithfulness metric."""
        from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
        
        metric = FaithfulnessMetric(model=self.model)
        score = metric.measure(test_case)
        
        explanation = f"Faithfulness measures how well the output aligns with the provided context. Score: {score}/1.0"
        return score, explanation
    
    def evaluate_answer_relevancy(self, test_case) -> tuple[float, str]:
        """Evaluate answer relevancy metric.""" 
        from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
        
        metric = AnswerRelevancyMetric(model=self.model)
        score = metric.measure(test_case)
        
        explanation = f"Answer Relevancy measures how well the output addresses the input question. Score: {score}/1.0"
        return score, explanation
    
    def evaluate_contextual_precision(self, test_case) -> tuple[float, str]:
        """Evaluate contextual precision metric."""
        from deepeval.metrics.contextual_precision.contextual_precision import ContextualPrecisionMetric
        
        # Validate that required parameters are present
        if test_case.expected_output is None:
            raise ValueError("Contextual Precision metric requires an expected_output to compare against")
        
        metric = ContextualPrecisionMetric(model=self.model)
        score = metric.measure(test_case)
        
        explanation = f"Contextual Precision measures how precise the retrieved context is for generating the expected output. Score: {score}/1.0"
        return score, explanation
    
    def evaluate_contextual_recall(self, test_case) -> tuple[float, str]:
        """Evaluate contextual recall metric."""
        from deepeval.metrics.contextual_recall.contextual_recall import ContextualRecallMetric
        
        # Validate that required parameters are present
        if test_case.expected_output is None:
            raise ValueError("Contextual Recall metric requires an expected_output to compare against")
        
        metric = ContextualRecallMetric(model=self.model)
        score = metric.measure(test_case)
        
        explanation = f"Contextual Recall measures how well the context covers the expected answer. Score: {score}/1.0"
        return score, explanation
    
    def evaluate(self, metric_name: str, source: str, output: str, query: Optional[str] = None, expected_output: Optional[str] = None) -> tuple[float, str]:
        """Main evaluation method that routes to specific metric evaluators."""
        metric_name = metric_name.lower()
        
        if not self.validate_metric(metric_name):
            raise ValueError(f"Unsupported metric: {metric_name}. Supported: {list(self.SUPPORTED_METRICS.keys())}")
        
        test_case = self.create_test_case(source, output, query, expected_output)
        
        # Route to appropriate evaluation method
        if metric_name == "faithfulness":
            return self.evaluate_faithfulness(test_case)
        elif metric_name == "answer_relevancy":
            return self.evaluate_answer_relevancy(test_case)  
        elif metric_name == "contextual_precision":
            return self.evaluate_contextual_precision(test_case)
        elif metric_name == "contextual_recall":
            return self.evaluate_contextual_recall(test_case)
        else:
            raise ValueError(f"Metric {metric_name} is not implemented yet")


@app.post("/eval", response_model=EvalResponse)
async def evaluate_llm_response(req: EvalRequest):
    """
    Evaluate an LLM response using the specified metric.
    
    This enterprise-grade evaluation system supports multiple metrics for training purposes.
    Each metric can be used independently to teach specific evaluation concepts.
    
    Args:
        req: EvalRequest with source (context), output (response), metric type, and optional provider
        
    Returns:
        EvalResponse with metric score and explanation
    """
    try:
        # Validate inputs
        if not req.source or not req.output:
            raise ValueError("Source and output cannot be empty")
        
        if not req.metric:
            req.metric = "faithfulness"  # Default metric
            
        logger.info(f"Evaluating metric: {req.metric}")
        logger.info(f"Source length: {len(req.source)}")
        logger.info(f"Output length: {len(req.output)}")
        
        # Initialize evaluator
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        evaluator = MetricEvaluator(openai_api_key, openai_model)
        
        logger.info(f"Using evaluation model: {openai_model}")
        
        # Perform evaluation
        score, explanation = evaluator.evaluate(
            metric_name=req.metric,
            source=req.source,
            output=req.output,
            expected_output=req.expected_output
        )
        
        logger.info(f"Evaluation complete. Metric: {req.metric}, Score: {score}")
        
        return EvalResponse(
            metric_name=req.metric,
            score=score,
            explanation=explanation
        )
            
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Deepeval Evaluation Service",
        "version": "1.0.0"
    }


@app.get("/metrics-info")
async def metrics_info():
    """Get available metrics information for training purposes"""
    metrics = []
    for metric_name, description in MetricEvaluator.SUPPORTED_METRICS.items():
        metrics.append({
            "name": metric_name,
            "description": description,
            "endpoint": "/eval",
            "parameter": f'"metric": "{metric_name}"',
            "range": "0.0 to 1.0",
            "higher_is_better": True
        })
    
    return {
        "available_metrics": metrics,
        "usage": "Include 'metric' parameter in POST /eval request body",
        "training_note": "Each metric can be used independently for step-by-step learning",
        "example_request": {
            "source": "Context or retrieval content",
            "output": "LLM generated response", 
            "metric": "faithfulness",
            "expected_output": "Expected or reference response (required for contextual_precision and contextual_recall)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Deepeval Evaluation Service...")
    logger.info("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
