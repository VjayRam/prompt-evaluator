# PromptEval

PromptEval is a secure, privacy-focused platform for evaluating Large Language Model (LLM) responses using an "LLM-as-a-Judge" approach. It allows users to upload datasets, select evaluation metrics, and run evaluations using various LLM providers (Google Gemini, OpenAI, Anthropic) without storing sensitive data or API keys.

**[Live Demo](https://prompt-eval-hub.vercel.app/)**

![PromptEval Screenshot](frontend/public/logo.png)

## Features

*   **LLM-as-a-Judge**: Uses advanced LLMs to evaluate the quality of responses based on predefined or custom metrics.
*   **Privacy-First Architecture**:
    *   **Zero Persistence**: API keys and dataset content are processed in-memory and never stored in a database.
    *   **Client-Side Storage**: User preferences and non-sensitive state are stored in the browser's `localStorage`.
*   **Multi-Provider Support**:
    *   Google Gemini (Flash, Pro, Flash-Lite)
    *   OpenAI (GPT-4o, GPT-4-Turbo, GPT-4o-mini)
    *   Anthropic (Claude 3.5 Sonnet, Haiku)
*   **Comprehensive Metrics**:
    *   **Predefined**: Coherence, Fluency, Safety, Groundedness, Instruction Following, Verbosity, Text Quality, Summarization Quality, QA Quality.
    *   **Custom**: Create your own metrics with custom system prompts.
*   **Real-time Evaluation**: Streaming progress updates using Server-Sent Events (SSE) with detailed progress tracking.
*   **Rate Limiting**:
    *   Built-in provider-specific rate limits (RPM/RPS).
    *   Customizable rate limits for advanced users.
*   **Data Management**:
    *   CSV dataset upload (requires `prompt` and `response` columns).
    *   Detailed results view with filtering and sorting.
    *   Export results to CSV or JSON.
    *   Optional local persistence of small datasets.

## Technical Stack

### Frontend
*   **Framework**: [Astro](https://astro.build/) (v5) - chosen for performance and static site generation capabilities.
*   **Styling**: [Tailwind CSS](https://tailwindcss.com/) (v4) - for a modern, responsive, and dark-themed UI.
*   **State Management**: Vanilla JavaScript with reactive UI updates and `localStorage` persistence.
*   **Communication**: `fetch` API for REST endpoints and `EventSource` pattern for SSE streaming.

### Backend
*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - high-performance, async Python web framework.
*   **Language**: Python 3.10+
*   **LLM Integration**:
    *   `openai` SDK
    *   `anthropic` SDK
    *   `google-genai` SDK
*   **Data Processing**: `pandas` for efficient dataset handling.
*   **Security**:
    *   In-memory processing.
    *   Strict CORS policies.
    *   Security headers (HSTS, X-Frame-Options, etc.).
    *   Request size limiting.
*   **Concurrency**: Async request handling and thread-safe rate limiting implementation.

## Project Structure

```
llm-eval/
├── backend/                 # Python FastAPI Backend
│   ├── api/                 # API Routes and Models
│   │   ├── models.py        # Pydantic data models
│   │   ├── routes.py        # API endpoints
│   │   └── security.py      # Security utilities (rate limiting, key masking)
│   ├── evaluation/          # Evaluation Logic
│   │   └── eval_engine.py   # Core evaluation engine
│   ├── llms/                # LLM Client Wrappers
│   │   └── llm_client.py    # Unified client for OpenAI, Anthropic, Google
│   ├── metrics/             # Evaluation Metrics
│   │   ├── eval_metrics.py  # Metric definitions
│   │   └── eval_templates.py# System prompt templates
│   └── main.py              # Application entry point
│
└── frontend/                # Astro Frontend
    ├── public/              # Static assets
    ├── src/
    │   ├── layouts/         # Page layouts
    │   ├── pages/           # Application pages (index, templates)
    │   └── styles/          # Global CSS
    ├── astro.config.mjs     # Astro configuration
    └── package.json         # Frontend dependencies
```

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites
*   **Node.js** (v18 or higher)
*   **Python** (v3.10 or higher)
*   **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/VjayRam/prompt-evaluator.git
cd prompt-evaluator
```

### 2. Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3.  Activate the virtual environment:
    *   **Windows**:
        ```bash
        venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you create a `requirements.txt` if one doesn't exist, containing: `fastapi`, `uvicorn`, `pandas`, `openai`, `anthropic`, `google-genai`, `python-multipart`)*

5.  Run the backend server:
    ```bash
    # Run from the project root (one level up from backend/)
    cd ..
    python -m backend.main
    ```
    The backend will start at `http://localhost:8000`.

### 3. Frontend Setup

1.  Open a new terminal and navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Configure the API Endpoint:
    *   Open `frontend/src/pages/index.astro` and `frontend/src/pages/templates.astro`.
    *   Locate the `const API` variable.
    *   Change it to point to your local backend:
        ```javascript
        const API = 'http://localhost:8000/api/v1';
        ```

4.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:4321`.

## Usage Guide

1.  **Configuration**:
    *   Select your LLM Provider (e.g., OpenAI).
    *   Select a specific Model (e.g., GPT-4o).
    *   Enter your API Key (stored locally only).

2.  **Dataset**:
    *   Upload a CSV file containing at least `prompt` and `response` columns.
    *   (Optional) `history` column for multi-turn chat evaluation.

3.  **Metrics**:
    *   Select one or more predefined metrics (e.g., Coherence, Safety).
    *   Or create a Custom Metric with your own evaluation criteria.

4.  **Run Evaluation**:
    *   Click "Run Evaluation".
    *   Watch the real-time progress bar.

5.  **Analyze Results**:
    *   View the summary statistics (Mean, Standard Deviation).
    *   Inspect individual row results with ratings and explanations.
    *   Download the full report as CSV or JSON.

## Deployment

*   **Frontend**: Deploy to Vercel, Netlify, or any static site host.
*   **Backend**: Deploy to Vercel (using Python runtime), Railway, Render, or any container platform (Docker).

## License

MIT License. See [LICENSE](LICENSE) for more information.

