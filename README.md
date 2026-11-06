 FastAPI Streaming, Token Counting & Cost Tracking

## Project Overview

This project demonstrates how to build an LLM-powered application using FastAPI, OpenRouter and Server-Sent Events (SSE).

It implements real-time streaming responses, token counting, API cost calculation, SQLite usage tracking and daily budget management.

## Objectives

- Create a FastAPI streaming endpoint using `StreamingResponse`.
- Implement Server-Sent Events (SSE).
- Display LLM responses progressively in a browser.
- Estimate input tokens using `tiktoken`.
- Compare estimated tokens with actual API usage.
- Calculate API costs using configurable model pricing.
- Store API usage history in SQLite.
- Implement a daily API spending limit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| FastAPI | REST API development |
| OpenRouter | LLM API access |
| OpenAI SDK | API communication |
| StreamingResponse | Real-time response streaming |
| Server-Sent Events | Sending streaming chunks |
| tiktoken | Token estimation |
| SQLite | Usage and cost logging |
| HTML, CSS, JavaScript | Browser streaming interface |
| Uvicorn | ASGI server |

## Project Structure

```text
week5_day2/
│
├── app.py
├── config.json
├── requirements.txt
├── .env
├── .gitignore
├── usage.db
│
└── static/
    └── index.html
```

## Task 1 – Streaming Endpoint

### Endpoint

`POST /chat/stream`

The endpoint uses FastAPI's `StreamingResponse` with:

```python
media_type="text/event-stream"
```

### Features

- Streams LLM-generated text as individual chunks.
- Uses an asynchronous generator.
- Sends each chunk using SSE format.
- Sends a final `[DONE]` event.
- Handles client disconnections.
- Uses a JavaScript `fetch()` request and stream reader to display text progressively.

### Example Request

```json
{
  "prompt": "Explain machine learning",
  "model": "openai/gpt-4o-mini",
  "max_tokens": 200
}
```

### SSE Response Format

```text
data: {"text": "Machine"}

data: {"text": " learning"}

data: {"text": " is"}

data: [DONE]
```

## Task 2 – Token Counting and Cost Tracking

### Token Counting

Implemented:

```python
count_tokens(messages, model)
```

Uses `tiktoken` to estimate input tokens before sending the request.

The application checks the configured model context limit and rejects requests that exceed it.

### Cost Calculation

API costs are calculated using:

```text
Input Cost =
Input Tokens × Input Price / 1,000,000

Output Cost =
Output Tokens × Output Price / 1,000,000

Total Cost = Input Cost + Output Cost
```

Model pricing is maintained in `config.json`.

**Note:** Update the example pricing with the current rates for your selected OpenRouter model.

### SQLite Usage Logging

Each API call records:

- Timestamp
- Model name
- Estimated input tokens
- Actual or estimated prompt tokens
- Completion tokens
- Total tokens
- API cost

### Usage Endpoint

`GET /usage`

Returns today's:

- Total prompt tokens
- Total completion tokens
- Total tokens
- Total API cost
- Daily budget
- Remaining budget

### Daily Budget Management

The application estimates the maximum cost of a request before sending it.

If the estimated request cost exceeds the remaining daily budget, the API returns:

`429 Too Many Requests`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Browser streaming interface |
| POST | `/chat` | Non-streaming LLM response |
| POST | `/chat/stream` | Real-time streaming response |
| GET | `/usage` | Daily token and cost report |
| GET | `/docs` | FastAPI Swagger documentation |

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd week5_day2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Never upload `.env` or expose your API key on GitHub.

### 5. Run the application

```bash
python -m uvicorn app:app --reload
```

## Testing

### Browser Streaming

Open:

`http://127.0.0.1:8000/`

Enter a prompt and click **Start Streaming**.

The generated response appears progressively in the browser.

### Swagger UI

Open:

`http://127.0.0.1:8000/docs`

Test the `/chat` and `/chat/stream` endpoints.

### Test with curl

```powershell
$body = @{
    prompt = "Explain machine learning"
    model = "openai/gpt-4o-mini"
    max_tokens = 200
} | ConvertTo-Json -Compress

$body | curl.exe -N -X POST "http://127.0.0.1:8000/chat/stream" -H "Content-Type: application/json" --data-binary "@-"
```

### Check Usage

Open:

`http://127.0.0.1:8000/usage`

## Concepts Learned

1. **StreamingResponse:** Sends generated content progressively instead of waiting for the entire response.
2. **Server-Sent Events:** Provides a simple event-based format for streaming data from the server.
3. **Async Generators:** Generate response chunks asynchronously.
4. **tiktoken:** Estimates the number of tokens in input messages.
5. **Token Usage:** Compares estimated tokens with provider-reported usage.
6. **Cost Tracking:** Calculates API costs based on input and output tokens.
7. **SQLite:** Maintains persistent API usage records.
8. **Budget Management:** Prevents requests that exceed the configured daily spending limit.

## Learning Outcome

Successfully developed an LLM application with real-time streaming, token estimation, API cost calculation and persistent usage tracking using FastAPI and OpenRouter.

## Author

**Dhanush S**

Week 5 – Day 2 | LLM Application Development
