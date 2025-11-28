"""
LLM-as-a-Judge Evaluation API Runner

Run from project root with: python run.py
"""
import sys
sys.dont_write_bytecode = True

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
