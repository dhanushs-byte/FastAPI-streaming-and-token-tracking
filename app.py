
import os
import json
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "usage.db"

with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in .env")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

app = FastAPI(
    title="LLM Streaming and Token Cost Tracking",
    version="1.0"
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str = "You are a helpful assistant."
    model: str = CONFIG["default_model"]
    max_tokens: int = Field(default=500, ge=1)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                local_date TEXT,
                model TEXT,
                estimated_input_tokens INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL
            )
        """)


@app.on_event("startup")
def startup():
    initialize_database()


# --------------------------------------------------
# TOKEN COUNTING
# --------------------------------------------------

def get_model_config(model):
    models = CONFIG["models"]

    if model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model not configured: {model}"
        )

    return models[model]


def count_tokens(messages, model):
    get_model_config(model)

    try:
        encoding = tiktoken.encoding_for_model(
            model.split("/")[-1]
        )
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 3

    for message in messages:
        total += 4
        total += len(encoding.encode(message["role"]))
        total += len(encoding.encode(message["content"]))

    return total


def check_context_limit(messages, model, max_tokens):
    estimated = count_tokens(messages, model)

    context_limit = get_model_config(model)["context_limit"]

    if estimated + max_tokens > context_limit:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Context limit exceeded",
                "estimated_tokens": estimated,
                "context_limit": context_limit
            }
        )

    return estimated


# --------------------------------------------------
# COST CALCULATION
# --------------------------------------------------

def calculate_cost(model, input_tokens, output_tokens):
    prices = get_model_config(model)["price_per_1m_tokens_usd"]

    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000

    return input_cost + output_cost


# --------------------------------------------------
# DAILY USAGE
# --------------------------------------------------

def get_today_usage():
    today = datetime.now().astimezone().date().isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                COALESCE(SUM(prompt_tokens), 0),
                COALESCE(SUM(completion_tokens), 0),
                COALESCE(SUM(total_tokens), 0),
                COALESCE(SUM(cost_usd), 0)
            FROM usage_logs
            WHERE local_date = ?
        """, (today,)).fetchone()

    spent = float(row[3])
    budget = float(CONFIG["daily_budget_usd"])

    return {
        "date": today,
        "prompt_tokens": row[0],
        "completion_tokens": row[1],
        "total_tokens": row[2],
        "total_cost_usd": round(spent, 8),
        "daily_budget_usd": budget,
        "remaining_budget_usd": round(max(0, budget - spent), 8)
    }


def check_daily_budget(model, estimated_input, max_tokens):
    usage = get_today_usage()

    estimated_cost = calculate_cost(
        model,
        estimated_input,
        max_tokens
    )

    if usage["total_cost_usd"] + estimated_cost > usage["daily_budget_usd"]:
        raise HTTPException(
            status_code=429,
            detail="Daily API budget exceeded"
        )


# --------------------------------------------------
# SAVE USAGE TO SQLITE
# --------------------------------------------------

def save_usage(model, estimated, actual_input, actual_output):
    cost = calculate_cost(
        model,
        actual_input,
        actual_output
    )

    current = datetime.now().astimezone()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO usage_logs (
                timestamp,
                local_date,
                model,
                estimated_input_tokens,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            current.isoformat(),
            current.date().isoformat(),
            model,
            estimated,
            actual_input,
            actual_output,
            actual_input + actual_output,
            cost
        ))

    return cost


# --------------------------------------------------
# BUILD MESSAGES
# --------------------------------------------------

def build_messages(body):
    return [
        {"role": "system", "content": body.system},
        {"role": "user", "content": body.prompt}
    ]


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    path = BASE_DIR / "static" / "index.html"
    return path.read_text(encoding="utf-8")


# --------------------------------------------------
# TASK 1: NON-STREAMING CHAT
# --------------------------------------------------

@app.post("/chat")
async def chat(body: ChatRequest):

    try:
        messages = build_messages(body)

        estimated = check_context_limit(
            messages, body.model, body.max_tokens
        )

        check_daily_budget(
            body.model, estimated, body.max_tokens
        )

        response = await client.chat.completions.create(
            model=body.model,
            messages=messages,
            temperature=0.7,
            max_tokens=body.max_tokens
        )

        usage = response.usage

        if usage is None:
            raise HTTPException(
                status_code=502,
                detail="Provider did not return usage information"
            )

        actual_input = usage.prompt_tokens or 0
        actual_output = usage.completion_tokens or 0

        cost = save_usage(
            body.model,
            estimated,
            actual_input,
            actual_output
        )

        return {
            "model": response.model,
            "answer": response.choices[0].message.content or "",
            "estimated_input_tokens": estimated,
            "actual_prompt_tokens": actual_input,
            "actual_completion_tokens": actual_output,
            "total_tokens": actual_input + actual_output,
            "cost_usd": round(cost, 8),
            "estimate_difference": actual_input - estimated
        }

    except HTTPException:
        raise

    except Exception as error:
        print("CHAT ERROR:", repr(error))

        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {str(error)}"
        )


# --------------------------------------------------
# TASK 1: STREAMING ENDPOINT WITH SSE
# --------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):

    messages = build_messages(body)

    estimated = check_context_limit(
        messages, body.model, body.max_tokens
    )

    check_daily_budget(
        body.model, estimated, body.max_tokens
    )

    async def generate():

        chunks = []
        actual_input = 0
        actual_output = 0

        try:
            stream = await client.chat.completions.create(
                model=body.model,
                messages=messages,
                temperature=0.7,
                max_tokens=body.max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in stream:

                if await request.is_disconnected():
                    break

                if chunk.choices:
                    content = chunk.choices[0].delta.content

                    if content:
                        chunks.append(content)

                        event_data = json.dumps({
                            "text": content
                        })

                        yield f"data: {event_data}\n\n"

                if getattr(chunk, "usage", None):
                    actual_input = chunk.usage.prompt_tokens or 0
                    actual_output = chunk.usage.completion_tokens or 0

            # Fallback when provider does not send stream usage.
            usage_is_estimated = (
                actual_input == 0 and actual_output == 0
            )

            if usage_is_estimated:
                actual_input = estimated

                try:
                    encoding = tiktoken.encoding_for_model(
                        body.model.split("/")[-1]
                    )
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")

                actual_output = len(
                    encoding.encode("".join(chunks))
                )

            if chunks:
                cost = save_usage(
                    body.model,
                    estimated,
                    actual_input,
                    actual_output
                )

                usage_data = {
                    "estimated_input_tokens": estimated,
                    "actual_prompt_tokens": actual_input,
                    "actual_completion_tokens": actual_output,
                    "cost_usd": round(cost, 8),
                    "usage_is_estimated": usage_is_estimated
                }

                if not await request.is_disconnected():
                    yield (
                        "event: usage\n"
                        f"data: {json.dumps(usage_data)}\n\n"
                    )

                    yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            raise

        except Exception as error:
            print("STREAM ERROR:", repr(error))

            yield (
                "event: error\n"
                f"data: {json.dumps({'error': str(error)})}\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# --------------------------------------------------
# TASK 2: USAGE ENDPOINT
# --------------------------------------------------

@app.get("/usage")
async def usage():
    return get_today_usage()
