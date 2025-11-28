"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Union
from enum import Enum


class MetricType(str, Enum):
    """Available evaluation metrics."""
    # Pointwise metrics
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    SAFETY = "safety"
    GROUNDEDNESS = "groundedness"
    INSTRUCTION_FOLLOWING = "instruction_following"
    VERBOSITY = "verbosity"
    TEXT_QUALITY = "text_quality"
    MULTI_TURN_CHAT_QUALITY = "multi_turn_chat_quality"
    MULTI_TURN_CHAT_SAFETY = "multi_turn_chat_safety"
    SUMMARIZATION_QUALITY = "summarization_quality"
    QUESTION_ANSWERING_QUALITY = "question_answering_quality"


class JudgeModel(str, Enum):
    """Supported judge models."""
    # Google models
    GEMINI_2_FLASH_LITE = "google/gemini-2.0-flash-lite"
    GEMINI_2_FLASH = "google/gemini-2.0-flash"
    GEMINI_2_PRO = "google/gemini-2.0-pro"
    # OpenAI models
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    GPT_4_TURBO = "openai/gpt-4-turbo"
    # Anthropic models
    CLAUDE_3_5_SONNET = "anthropic/claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-latest"


class DatasetRow(BaseModel):
    """Single row in the evaluation dataset."""
    prompt: str = Field(..., description="The user prompt/input")
    response: str = Field(..., description="The AI-generated response to evaluate")
    history: Optional[str] = Field(None, description="Conversation history for multi-turn metrics")


class CustomMetric(BaseModel):
    """Custom user-defined metric."""
    name: str = Field(..., min_length=1, max_length=100, description="Name of the custom metric")
    template: str = Field(..., min_length=10, description="Evaluation template with {prompt} and {response} placeholders")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid identifier-like."""
        clean_name = v.strip().lower().replace(' ', '_')
        if not clean_name:
            raise ValueError("Metric name cannot be empty")
        return clean_name
    
    @field_validator('template')
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Ensure template has required placeholders."""
        if '{prompt}' not in v.lower() and '{response}' not in v.lower():
            raise ValueError("Template should contain {prompt} and/or {response} placeholders")
        return v


class MetricInput(BaseModel):
    """Flexible metric input - can be a predefined metric name or a custom metric object."""
    predefined: Optional[MetricType] = Field(None, description="Predefined metric type")
    custom: Optional[CustomMetric] = Field(None, description="Custom metric definition")
    
    @model_validator(mode='after')
    def check_one_defined(self):
        if self.predefined is None and self.custom is None:
            raise ValueError("Either 'predefined' or 'custom' must be provided")
        if self.predefined is not None and self.custom is not None:
            raise ValueError("Only one of 'predefined' or 'custom' should be provided")
        return self


class RateLimitSettings(BaseModel):
    """Optional rate limit settings for API calls."""
    rpm: Optional[int] = Field(None, ge=1, le=1000, description="Requests per minute limit")
    rps: Optional[int] = Field(None, ge=1, le=100, description="Requests per second limit")


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""
    dataset: list[DatasetRow] = Field(..., min_length=1, max_length=1000, description="Dataset rows to evaluate")
    judge_model: JudgeModel = Field(..., description="The LLM model to use as judge")
    metrics: list[Union[str, dict]] = Field(..., min_length=1, description="List of metric names (strings) or custom metric objects")
    api_key: str = Field(..., min_length=10, description="API key for the judge model provider")
    rate_limits: Optional[RateLimitSettings] = Field(None, description="Optional custom rate limit settings")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Basic validation - ensure key is not obviously invalid."""
        if v.strip() == "" or v == "string":
            raise ValueError("Invalid API key provided")
        return v
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v: list) -> list:
        """Validate metrics list - can be strings (predefined) or dicts (custom)."""
        if not v:
            raise ValueError("At least one metric is required")
        
        validated = []
        for item in v:
            if isinstance(item, str):
                # Check if it's a valid predefined metric
                try:
                    MetricType(item)
                    validated.append({"type": "predefined", "name": item})
                except ValueError:
                    raise ValueError(f"Unknown predefined metric: {item}. Use a custom metric object for custom metrics.")
            elif isinstance(item, dict):
                # Custom metric
                if 'name' not in item or 'template' not in item:
                    raise ValueError("Custom metric must have 'name' and 'template' fields")
                validated.append({"type": "custom", "name": item['name'], "template": item['template']})
            else:
                raise ValueError(f"Invalid metric format: {item}")
        
        return validated


class MetricResult(BaseModel):
    """Result for a single metric."""
    mean: float = Field(..., description="Mean rating across all samples")
    std: float = Field(..., description="Standard deviation of ratings")


class EvaluationResultRow(BaseModel):
    """Single row in evaluation results."""
    prompt: str
    response: str
    history: Optional[str] = None
    # Dynamic fields for each metric's rating and explanation will be added


class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint."""
    success: bool = Field(..., description="Whether evaluation completed successfully")
    num_samples: int = Field(..., description="Number of samples evaluated")
    results: list[dict] = Field(..., description="Detailed results for each sample")
    summary: dict[str, MetricResult] = Field(..., description="Summary statistics per metric")
    message: Optional[str] = Field(None, description="Additional message or error details")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str


class MetricsListResponse(BaseModel):
    """Response model for available metrics endpoint."""
    pointwise_metrics: list[str]
    description: dict[str, str]


class ModelsListResponse(BaseModel):
    """Response model for available models endpoint."""
    models: list[str]
    providers: list[str]
