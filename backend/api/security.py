"""
Security utilities for the LLM Evaluation API.
Handles API key protection and request validation.
"""
import hashlib
import secrets
import logging
from functools import wraps
from typing import Callable
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging - never log sensitive data
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[datetime]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        current_requests = len([
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ])
        
        return max(0, self.max_requests - current_requests)


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)


def get_client_identifier(request: Request) -> str:
    """Get a unique identifier for the client (IP-based)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    return client_ip


def mask_api_key(api_key: str) -> str:
    """Mask API key for safe logging - show only first 4 and last 4 chars."""
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def hash_for_logging(data: str) -> str:
    """Create a hash for logging purposes without exposing actual data."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def validate_request_size(max_size_mb: float = 10.0):
    """Decorator to validate request size."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            content_length = request.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_size_mb:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request too large. Maximum size: {max_size_mb}MB"
                    )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def log_evaluation_request(
    client_id: str,
    judge_model: str,
    metrics: list[str],
    num_samples: int,
    success: bool
):
    """Log evaluation request without sensitive data."""
    logger.info(
        f"Evaluation request - Client: {hash_for_logging(client_id)}, "
        f"Model: {judge_model}, Metrics: {metrics}, "
        f"Samples: {num_samples}, Success: {success}"
    )


class SecureAPIKeyHandler:
    """
    Handles API keys securely - keys are used only for the duration
    of the request and not stored.
    """
    
    @staticmethod
    def validate_key_format(api_key: str, provider: str) -> bool:
        """
        Basic format validation for API keys.
        Does NOT validate if key is actually valid with the provider.
        """
        if not api_key or len(api_key) < 10:
            return False
        
        # Basic format checks per provider
        if provider == "google":
            # Google API keys typically start with 'AIza'
            return api_key.startswith("AIza") or len(api_key) >= 20
        elif provider == "openai":
            # OpenAI keys start with 'sk-'
            return api_key.startswith("sk-") or len(api_key) >= 20
        elif provider == "anthropic":
            # Anthropic keys start with 'sk-ant-'
            return api_key.startswith("sk-ant-") or len(api_key) >= 20
        
        return True  # Allow unknown providers with basic length check
    
    @staticmethod
    def clear_key_from_memory(api_key: str) -> None:
        """
        Attempt to clear API key from memory.
        Note: Python doesn't guarantee immediate garbage collection,
        but this helps mark the data for cleanup.
        """
        # Overwrite the string content (limited effectiveness in Python)
        # This is mainly symbolic - proper cleanup happens via GC
        del api_key


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return secrets.token_hex(16)
