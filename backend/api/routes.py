"""
API routes for LLM Evaluation service.
"""
import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncGenerator

from backend.api.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    MetricsListResponse,
    ModelsListResponse,
    MetricType,
    JudgeModel,
    MetricResult,
)
from backend.api.security import (
    rate_limiter,
    get_client_identifier,
    mask_api_key,
    log_evaluation_request,
    SecureAPIKeyHandler,
    generate_request_id,
)
from backend.metrics.eval_templates import EvalMetricTemplates
from backend.metrics.eval_metrics import EvalMetric
from backend.llms.llm_client import LLMClient
from backend.evaluation.eval_engine import Evaluator

router = APIRouter()


# Metric name to template mapping
METRIC_TEMPLATES = {
    MetricType.COHERENCE: EvalMetricTemplates.PointwiseMetric.COHERENCE,
    MetricType.FLUENCY: EvalMetricTemplates.PointwiseMetric.FLUENCY,
    MetricType.SAFETY: EvalMetricTemplates.PointwiseMetric.SAFETY,
    MetricType.GROUNDEDNESS: EvalMetricTemplates.PointwiseMetric.GROUNDEDNESS,
    MetricType.INSTRUCTION_FOLLOWING: EvalMetricTemplates.PointwiseMetric.INSTRUCTION_FOLLOWING,
    MetricType.VERBOSITY: EvalMetricTemplates.PointwiseMetric.VERBOSITY,
    MetricType.TEXT_QUALITY: EvalMetricTemplates.PointwiseMetric.TEXT_QUALITY,
    MetricType.MULTI_TURN_CHAT_QUALITY: EvalMetricTemplates.PointwiseMetric.MULTI_TURN_CHAT_QUALITY,
    MetricType.MULTI_TURN_CHAT_SAFETY: EvalMetricTemplates.PointwiseMetric.MULTI_TURN_CHAT_SAFETY,
    MetricType.SUMMARIZATION_QUALITY: EvalMetricTemplates.PointwiseMetric.SUMMARIZATION_QUALITY,
    MetricType.QUESTION_ANSWERING_QUALITY: EvalMetricTemplates.PointwiseMetric.QUESTION_ANSWERING_QUALITY,
}

METRIC_DESCRIPTIONS = {
    "coherence": "Measures the logical flow and organization of ideas in the response",
    "fluency": "Measures language mastery, grammar, and natural flow",
    "safety": "Measures absence of harmful, toxic, or inappropriate content",
    "groundedness": "Measures if response only contains information from the prompt",
    "instruction_following": "Measures how well the response follows instructions",
    "verbosity": "Measures appropriate conciseness of the response",
    "text_quality": "Overall text quality combining multiple factors",
    "multi_turn_chat_quality": "Quality assessment for multi-turn conversations",
    "multi_turn_chat_safety": "Safety assessment for multi-turn conversations",
    "summarization_quality": "Quality of text summarization",
    "question_answering_quality": "Quality of question answering",
}


async def check_rate_limit(request: Request) -> str:
    """Dependency to check rate limiting."""
    client_id = get_client_identifier(request)
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "3600"}
        )
    
    return client_id


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.get("/metrics", response_model=MetricsListResponse, tags=["Info"])
async def list_metrics():
    """List all available evaluation metrics."""
    # Convert templates to string keys
    templates = {k.value: v for k, v in METRIC_TEMPLATES.items()}
    
    return MetricsListResponse(
        pointwise_metrics=[m.value for m in MetricType],
        description=METRIC_DESCRIPTIONS,
        templates=templates
    )


@router.get("/models", response_model=ModelsListResponse, tags=["Info"])
async def list_models():
    """List all supported judge models."""
    models = [m.value for m in JudgeModel]
    providers = list(set(m.value.split("/")[0] for m in JudgeModel))
    
    return ModelsListResponse(models=models, providers=providers)


@router.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate(
    request: Request,
    eval_request: EvaluationRequest,
    client_id: Annotated[str, Depends(check_rate_limit)]
):
    """
    Evaluate AI responses using LLM-as-a-judge methodology.
    
    Metrics can be:
    - Predefined metric names (strings): "coherence", "fluency", etc.
    - Custom metrics (objects): {"name": "my_metric", "template": "Evaluate {response}..."}
    
    Security notes:
    - API keys are used only for the current request and not stored
    - Dataset content is processed in-memory and not persisted
    - Rate limiting is applied per client IP
    """
    request_id = generate_request_id()
    api_key = eval_request.api_key
    
    # Track metric names for logging
    metric_names = []
    
    try:
        # Extract provider from model
        provider = eval_request.judge_model.value.split("/")[0]
        
        # Validate API key format
        if not SecureAPIKeyHandler.validate_key_format(api_key, provider):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid API key format for provider: {provider}"
            )
        
        # Check for multi-turn metrics that require history
        multi_turn_metric_names = {"multi_turn_chat_quality", "multi_turn_chat_safety"}
        has_multi_turn = any(
            m.get("name") in multi_turn_metric_names 
            for m in eval_request.metrics 
            if m.get("type") == "predefined"
        )
        
        if has_multi_turn:
            # Check if any row is missing history
            missing_history = [
                i for i, row in enumerate(eval_request.dataset)
                if row.history is None or row.history.strip() == ""
            ]
            if missing_history:
                raise HTTPException(
                    status_code=400,
                    detail=f"Multi-turn metrics require 'history' field. Missing in rows: {missing_history[:5]}{'...' if len(missing_history) > 5 else ''}"
                )
        
        # Convert dataset to DataFrame
        dataset_dicts = []
        for row in eval_request.dataset:
            row_dict = {"prompt": row.prompt, "response": row.response}
            if row.history:
                row_dict["history"] = row.history
            dataset_dicts.append(row_dict)
        
        dataset = pd.DataFrame(dataset_dicts)
        
        # Create evaluation metrics (both predefined and custom)
        eval_metrics = []
        for metric_info in eval_request.metrics:
            metric_type = metric_info.get("type")
            metric_name = metric_info.get("name")
            metric_names.append(metric_name)
            
            if metric_type == "predefined":
                # Get template from predefined templates
                try:
                    metric_enum = MetricType(metric_name)
                    template = METRIC_TEMPLATES.get(metric_enum)
                    if template:
                        eval_metrics.append(
                            EvalMetric(
                                metric_name=metric_name,
                                metric_prompt_template=template
                            )
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown predefined metric: {metric_name}"
                    )
            
            elif metric_type == "custom":
                # Use custom template, but wrap with standard instruction for response format
                custom_template = metric_info.get("template")
                if not custom_template:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Custom metric '{metric_name}' is missing template"
                    )
                # Standard instruction for all metrics (forces JSON with int rating and string explanation)
                standard_instruction = (
                    "You are an expert evaluator. "
                    "Your task is to evaluate the following AI response according to the custom metric described below. "
                    "Return your answer as a JSON object with two fields: "
                    "'rating' (an integer, 1-5) and 'explanation' (a string explaining your rating). "
                    "Do not return anything except the JSON object.\n\n"
                )
                wrapped_template = standard_instruction + custom_template.strip()
                eval_metrics.append(
                    EvalMetric(
                        metric_name=metric_name,
                        metric_prompt_template=wrapped_template
                    )
                )
        
        if not eval_metrics:
            raise HTTPException(
                status_code=400,
                detail="No valid metrics provided"
            )
        
        # Initialize LLM client with optional custom rate limits
        try:
            rate_limit_rpm = None
            rate_limit_rps = None
            
            if eval_request.rate_limits:
                rate_limit_rpm = eval_request.rate_limits.rpm
                rate_limit_rps = eval_request.rate_limits.rps
            
            llm_client = LLMClient(
                judge_model=eval_request.judge_model.value,
                api_key=api_key,
                requests_per_minute=rate_limit_rpm,
                requests_per_second=rate_limit_rps
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to initialize LLM client: {str(e)}"
            )
        
        # Run evaluation
        evaluator = Evaluator(llm_client=llm_client, eval_metrics=eval_metrics)
        eval_table, summary = evaluator.evaluate(dataset=dataset)
        
        # Convert results to response format
        results = eval_table.to_dict(orient="records")
        
        # Format summary
        formatted_summary = {}
        for metric_name, stats in summary.items():
            if metric_name != "num_samples" and isinstance(stats, dict):
                formatted_summary[metric_name] = MetricResult(
                    mean=float(stats["mean"]),
                    std=float(stats["std"])
                )
        
        # Log success (without sensitive data)
        log_evaluation_request(
            client_id=client_id,
            judge_model=eval_request.judge_model.value,
            metrics=metric_names,
            num_samples=len(eval_request.dataset),
            success=True
        )
        
        return EvaluationResponse(
            success=True,
            num_samples=len(eval_request.dataset),
            results=results,
            summary=formatted_summary,
            message=f"Evaluation completed successfully. Request ID: {request_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log failure (without sensitive data)
        log_evaluation_request(
            client_id=client_id,
            judge_model=eval_request.judge_model.value,
            metrics=metric_names,
            num_samples=len(eval_request.dataset),
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}. Request ID: {request_id}"
        )
    
    finally:
        # Clear API key from local scope
        SecureAPIKeyHandler.clear_key_from_memory(api_key)


@router.get("/rate-limit-status", tags=["System"])
async def rate_limit_status(request: Request):
    """Check current rate limit status for the client."""
    client_id = get_client_identifier(request)
    remaining = rate_limiter.get_remaining(client_id)
    
    return {
        "remaining_requests": remaining,
        "max_requests": rate_limiter.max_requests,
        "window_seconds": rate_limiter.window_seconds
    }


@router.post("/evaluate/stream", tags=["Evaluation"])
async def evaluate_stream(
    request: Request,
    eval_request: EvaluationRequest,
    client_id: Annotated[str, Depends(check_rate_limit)]
):
    """
    Evaluate AI responses with streaming progress updates via Server-Sent Events.
    
    Streams progress events as each API call completes, then sends the final results.
    
    Event types:
    - progress: {"completed": int, "total": int, "metric": str, "percent": float}
    - result: The final evaluation results
    - error: Error message if evaluation fails
    """
    request_id = generate_request_id()
    api_key = eval_request.api_key
    
    async def generate_events() -> AsyncGenerator[str, None]:
        metric_names = []
        
        try:
            # Extract provider from model
            provider = eval_request.judge_model.value.split("/")[0]
            
            # Validate API key format
            if not SecureAPIKeyHandler.validate_key_format(api_key, provider):
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid API key format for provider: {provider}'})}\n\n"
                return
            
            # Check for multi-turn metrics that require history
            multi_turn_metric_names = {"multi_turn_chat_quality", "multi_turn_chat_safety"}
            has_multi_turn = any(
                m.get("name") in multi_turn_metric_names 
                for m in eval_request.metrics 
                if m.get("type") == "predefined"
            )
            
            if has_multi_turn:
                missing_history = [
                    i for i, row in enumerate(eval_request.dataset)
                    if row.history is None or row.history.strip() == ""
                ]
                if missing_history:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Multi-turn metrics require history field. Missing in rows: {missing_history[:5]}'})}\n\n"
                    return
            
            # Convert dataset to DataFrame
            dataset_dicts = []
            for row in eval_request.dataset:
                row_dict = {"prompt": row.prompt, "response": row.response}
                if row.history:
                    row_dict["history"] = row.history
                dataset_dicts.append(row_dict)
            
            dataset = pd.DataFrame(dataset_dicts)
            
            # Create evaluation metrics
            eval_metrics = []
            for metric_info in eval_request.metrics:
                metric_type = metric_info.get("type")
                metric_name = metric_info.get("name")
                metric_names.append(metric_name)
                
                if metric_type == "predefined":
                    try:
                        metric_enum = MetricType(metric_name)
                        template = METRIC_TEMPLATES.get(metric_enum)
                        if template:
                            eval_metrics.append(
                                EvalMetric(
                                    metric_name=metric_name,
                                    metric_prompt_template=template
                                )
                            )
                    except ValueError:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown predefined metric: {metric_name}'})}\n\n"
                        return
                
                elif metric_type == "custom":
                    custom_template = metric_info.get("template")
                    if not custom_template:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Custom metric {metric_name} is missing template'})}\n\n"
                        return
                    standard_instruction = (
                        "You are an expert evaluator. "
                        "Your task is to evaluate the following AI response according to the custom metric described below. "
                        "Return your answer as a JSON object with two fields: "
                        "'rating' (an integer, 1-5) and 'explanation' (a string explaining your rating). "
                        "Do not return anything except the JSON object.\n\n"
                    )
                    wrapped_template = standard_instruction + custom_template.strip()
                    eval_metrics.append(
                        EvalMetric(
                            metric_name=metric_name,
                            metric_prompt_template=wrapped_template
                        )
                    )
            
            if not eval_metrics:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No valid metrics provided'})}\n\n"
                return
            
            # Send initial progress
            total_calls = len(dataset) * len(eval_metrics)
            yield f"data: {json.dumps({'type': 'progress', 'completed': 0, 'total': total_calls, 'metric': 'initializing', 'percent': 0})}\n\n"
            
            # Initialize LLM client with optional custom rate limits
            rate_limit_rpm = None
            rate_limit_rps = None
            
            if eval_request.rate_limits:
                rate_limit_rpm = eval_request.rate_limits.rpm
                rate_limit_rps = eval_request.rate_limits.rps
            
            try:
                llm_client = LLMClient(
                    judge_model=eval_request.judge_model.value,
                    api_key=api_key,
                    requests_per_minute=rate_limit_rpm,
                    requests_per_second=rate_limit_rps
                )
            except ValueError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to initialize LLM client: {str(e)}'})}\n\n"
                return
            
            # Run evaluation with progress streaming
            evaluator = Evaluator(llm_client=llm_client, eval_metrics=eval_metrics)
            
            # Use the generator-based evaluation
            eval_table = None
            summary = None
            
            for event in evaluator.evaluate_with_progress(dataset=dataset):
                if event.get("type") == "progress":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event.get("type") == "complete":
                    eval_table = event["dataset"]
                    summary = event["summaries"]
            
            if eval_table is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Evaluation did not complete'})}\n\n"
                return
            
            # Convert results to response format
            results = eval_table.to_dict(orient="records")
            
            # Format summary
            formatted_summary = {}
            for metric_name, stats in summary.items():
                if metric_name != "num_samples" and isinstance(stats, dict):
                    formatted_summary[metric_name] = {
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"])
                    }
            
            # Send final result
            final_result = {
                "type": "result",
                "success": True,
                "num_samples": len(eval_request.dataset),
                "results": results,
                "summary": formatted_summary,
                "message": f"Evaluation completed successfully. Request ID: {request_id}"
            }
            yield f"data: {json.dumps(final_result)}\n\n"
            
            # Log success
            log_evaluation_request(
                client_id=client_id,
                judge_model=eval_request.judge_model.value,
                metrics=metric_names,
                num_samples=len(eval_request.dataset),
                success=True
            )
            
        except Exception as e:
            log_evaluation_request(
                client_id=client_id,
                judge_model=eval_request.judge_model.value,
                metrics=metric_names,
                num_samples=len(eval_request.dataset),
                success=False
            )
            yield f"data: {json.dumps({'type': 'error', 'message': f'Evaluation failed: {str(e)}. Request ID: {request_id}'})}\n\n"
        
        finally:
            SecureAPIKeyHandler.clear_key_from_memory(api_key)
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
