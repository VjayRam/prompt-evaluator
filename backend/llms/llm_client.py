from pydantic import BaseModel, Field
from openai import OpenAI
import anthropic
from google import genai
import time
import threading
from collections import deque
from typing import Optional


class LLMRateLimiter:
    """
    Customizable rate limiter for LLM API calls.
    Supports requests per minute (RPM) and requests per second (RPS) limits.
    Thread-safe implementation using sliding window algorithm.
    """
    
    # Default rate limits per provider (requests per minute)
    DEFAULT_LIMITS = {
        "google": {"rpm": 60, "rps": 10},
        "openai": {"rpm": 60, "rps": 10},
        "anthropic": {"rpm": 60, "rps": 10},
    }
    
    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_second: Optional[int] = None,
        provider: str = "default"
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute. If None, uses provider default.
            requests_per_second: Maximum requests allowed per second. If None, uses provider default.
            provider: The LLM provider name for default limits.
        """
        defaults = self.DEFAULT_LIMITS.get(provider, {"rpm": 60, "rps": 10})
        
        self.rpm_limit = requests_per_minute if requests_per_minute is not None else defaults["rpm"]
        self.rps_limit = requests_per_second if requests_per_second is not None else defaults["rps"]
        
        # Sliding window queues for tracking request timestamps
        self._minute_window: deque = deque()
        self._second_window: deque = deque()
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Stats tracking
        self._total_requests = 0
        self._total_wait_time = 0.0
    
    def _clean_windows(self, current_time: float) -> None:
        """Remove expired timestamps from sliding windows."""
        # Clean minute window (keep last 60 seconds)
        while self._minute_window and current_time - self._minute_window[0] > 60:
            self._minute_window.popleft()
        
        # Clean second window (keep last 1 second)
        while self._second_window and current_time - self._second_window[0] > 1:
            self._second_window.popleft()
    
    def _calculate_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before next request is allowed."""
        wait_time = 0.0
        
        # Check RPM limit
        if len(self._minute_window) >= self.rpm_limit:
            oldest_in_minute = self._minute_window[0]
            wait_for_rpm = 60 - (current_time - oldest_in_minute)
            wait_time = max(wait_time, wait_for_rpm)
        
        # Check RPS limit
        if len(self._second_window) >= self.rps_limit:
            oldest_in_second = self._second_window[0]
            wait_for_rps = 1 - (current_time - oldest_in_second)
            wait_time = max(wait_time, wait_for_rps)
        
        return max(0, wait_time)
    
    def acquire(self) -> float:
        """
        Acquire permission to make a request. Blocks if rate limit is exceeded.
        
        Returns:
            The time waited in seconds (0 if no wait was needed).
        """
        with self._lock:
            current_time = time.time()
            self._clean_windows(current_time)
            
            wait_time = self._calculate_wait_time(current_time)
            
            if wait_time > 0:
                # Release lock while sleeping to allow other threads
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()
                
                current_time = time.time()
                self._clean_windows(current_time)
                self._total_wait_time += wait_time
            
            # Record this request
            self._minute_window.append(current_time)
            self._second_window.append(current_time)
            self._total_requests += 1
            
            return wait_time
    
    def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.
        
        Returns:
            True if request is allowed, False if rate limit would be exceeded.
        """
        with self._lock:
            current_time = time.time()
            self._clean_windows(current_time)
            
            if self._calculate_wait_time(current_time) > 0:
                return False
            
            self._minute_window.append(current_time)
            self._second_window.append(current_time)
            self._total_requests += 1
            return True
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            current_time = time.time()
            self._clean_windows(current_time)
            
            return {
                "rpm_limit": self.rpm_limit,
                "rps_limit": self.rps_limit,
                "requests_in_last_minute": len(self._minute_window),
                "requests_in_last_second": len(self._second_window),
                "total_requests": self._total_requests,
                "total_wait_time_seconds": round(self._total_wait_time, 2),
            }
    
    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._minute_window.clear()
            self._second_window.clear()
            self._total_requests = 0
            self._total_wait_time = 0.0


class LLMClient:
    class LLMResponse(BaseModel):
        rating: int = Field(description="Rating score assigned by the LLM judge")
        explanation: str = Field(description="Explanation provided by the LLM judge for the rating")
        
    def __init__(
        self,
        judge_model: str,
        api_key: str,
        requests_per_minute: Optional[int] = 10,
        requests_per_second: Optional[int] = 1,
        enable_rate_limiting: bool = True
    ):
        """
        Initialize the LLM client with optional rate limiting.
        
        Args:
            judge_model: Model identifier in format "provider/model_name"
            api_key: API key for the provider
            requests_per_minute: Custom RPM limit (uses provider default if None)
            requests_per_second: Custom RPS limit (uses provider default if None)
            enable_rate_limiting: Whether to enable rate limiting (default: True)
        """
        self.model = judge_model
        self.api_key = api_key

        # Extract provider from model name (format: provider/model_name)
        if '/' in judge_model:
            self.model_provider = judge_model.split('/')[0]
            self.model_name = judge_model.split('/')[1]
        else:
            self.model_provider = "unknown"
            self.model_name = judge_model

        # Initialize rate limiter
        self.enable_rate_limiting = enable_rate_limiting
        if enable_rate_limiting:
            self.rate_limiter = LLMRateLimiter(
                requests_per_minute=requests_per_minute,
                requests_per_second=requests_per_second,
                provider=self.model_provider
            )
        else:
            self.rate_limiter = None

        if self.model_provider == "openai":
            try: 
                self.client = OpenAI(api_key=self.api_key)
                print("Client initialized: OpenAI")
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {e}")

        elif self.model_provider == "anthropic":
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("Client initialized: Anthropic")
            except Exception as e:
                raise ValueError(f"Failed to initialize Anthropic client: {e}")

        elif self.model_provider == "google":
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("Client initialized: Google Gemini")
            except Exception as e:
                raise ValueError(f"Failed to initialize Google client: {e}")
        
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        if enable_rate_limiting:
            print(f"Rate limiting enabled: {self.rate_limiter.rpm_limit} RPM, {self.rate_limiter.rps_limit} RPS")

    def get_rate_limit_stats(self) -> Optional[dict]:
        """Get current rate limiter statistics."""
        if self.rate_limiter:
            return self.rate_limiter.get_stats()
        return None

    def generate(self, prompt: str) -> str:
        # Apply rate limiting before making the API call
        if self.rate_limiter:
            wait_time = self.rate_limiter.acquire()
            if wait_time > 0:
                print(f"Rate limit: waited {wait_time:.2f}s before request")
        if self.model_provider == "openai":
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                text_format=self.LLMResponse,
            )
            return response.output_parsed
        
        elif self.model_provider == "anthropic":
            response = self.client.beta.messages.create(
                model=self.model_name,
                betas = ["structured-outputs-2025-11-13"],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                output_format={
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rating": {"type": "integer"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["rating", "explanation"],
                        "additionalProperties": False
                    }
                }
            )
            return response.content[0].text
        
        elif self.model_provider == "google":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": self.LLMResponse.model_json_schema(),
                }
            )
            return self.LLMResponse.model_validate_json(response.text)
        
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
