"""
LLM-as-a-Judge Evaluation API

A FastAPI backend for evaluating AI-generated responses using LLM judges.

Security Features:
- API keys are never stored and only used for the duration of each request
- Dataset content is processed in-memory only
- Rate limiting to prevent abuse
- Request size limits
- Secure logging without sensitive data exposure
"""
import sys
sys.dont_write_bytecode = True

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from .api.routes import router
from .api.security import generate_request_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LLM Evaluation API...")
    yield
    logger.info("Shutting down LLM Evaluation API...")


# Create FastAPI app
app = FastAPI(
    title="LLM-as-a-Judge Evaluation API",
    description="""
    API for evaluating AI-generated responses using LLM judges.
    
    ## Features
    - Multiple evaluation metrics (coherence, fluency, safety, etc.)
    - Support for multiple LLM providers (Google, OpenAI, Anthropic)
    - Secure API key handling (keys are never stored)
    - Rate limiting and request validation
    
    ## Security
    - API keys are used only for the duration of each request
    - Dataset content is processed in-memory and not persisted
    - All sensitive data is masked in logs
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add security middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["localhost", "127.0.0.1", "your-domain.com"]
# )


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add request ID for tracking
    response.headers["X-Request-ID"] = generate_request_id()
    
    return response


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent DoS attacks."""
    max_size = 10 * 1024 * 1024  # 10 MB
    
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum size: 10MB"}
        )
    
    return await call_next(request)


# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM-as-a-Judge Evaluation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
