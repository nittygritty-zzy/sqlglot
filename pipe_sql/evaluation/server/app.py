"""FastAPI server for pipe SQL model inference and tool execution.

Provides:
- POST /v1/chat/completions  — OpenAI-compatible chat endpoint
- POST /tools/{name}         — Tool execution endpoints
- GET  /benchmark/questions   — Benchmark question loading
- GET  /health               — Health check
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pipe_sql.evaluation.server.benchmark_routes import load_bird_dev, load_spider_dev
from pipe_sql.evaluation.server.benchmark_routes import router as benchmark_router
from pipe_sql.evaluation.server.inference import PipeSQLModel
from pipe_sql.evaluation.server.tool_routes import db_path_cache, tables_cache
from pipe_sql.evaluation.server.tool_routes import router as tool_router
from pipe_sql.training.schema_extractor import build_db_path_cache, build_tables_cache

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Model instance — loaded at startup
model: PipeSQLModel | None = None

MODEL_PATH = os.environ.get("MODEL_PATH", "pipe_sql/finetuning_output/merged")
BENCHMARK = os.environ.get("BENCHMARK", "spider")
SPIDER_DB_DIR = os.environ.get("SPIDER_DB_DIR", "data/spider/database")
SPIDER_DIR = os.environ.get("SPIDER_DIR", "data/spider")
BIRD_DB_DIR = os.environ.get("BIRD_DB_DIR", "data/bird/dev_20240627/dev_databases")
BIRD_DIR = os.environ.get("BIRD_DIR", "data/bird/dev_20240627")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and caches at startup."""
    global model

    # Build DB caches based on benchmark
    if BENCHMARK == "bird":
        logger.info(f"Using BIRD benchmark, db_dir={BIRD_DB_DIR}")
        db_dirs = [BIRD_DB_DIR]
    else:
        logger.info(f"Using Spider benchmark, db_dir={SPIDER_DB_DIR}")
        db_dirs = [SPIDER_DB_DIR]
        # Also check for Spider2-lite databases
        spider2_dir = os.environ.get("SPIDER2_DB_DIR", "")
        if spider2_dir and os.path.isdir(spider2_dir):
            db_dirs.append(spider2_dir)

    tc = build_tables_cache(db_dirs)
    dpc = build_db_path_cache(db_dirs)
    tables_cache.update(tc)
    db_path_cache.update(dpc)
    logger.info(f"Cached {len(tables_cache)} database schemas, {len(db_path_cache)} db paths")

    # Load benchmark questions
    if BENCHMARK == "bird":
        load_bird_dev(BIRD_DIR)
    else:
        load_spider_dev(SPIDER_DIR)

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    model = PipeSQLModel(MODEL_PATH)
    model.load()
    logger.info("Server ready")

    yield

    logger.info("Shutting down")


app = FastAPI(title="Pipe SQL Evaluation Server", lifespan=lifespan)
app.include_router(tool_router)
app.include_router(benchmark_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and model.is_loaded,
        "databases": len(db_path_cache),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Accepts messages in OpenAI format, applies training chat template,
    generates via the fine-tuned model, and returns OpenAI format response.
    """
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.1)

    if model is None or not model.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"},
        )

    # Generate response using training template
    parsed = model.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    # Build OpenAI-compatible response
    message = {"role": "assistant", "content": parsed.get("content", "")}

    if parsed.get("tool_calls"):
        message["tool_calls"] = parsed["tool_calls"]

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_PATH,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if parsed.get("tool_calls") else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("pipe_sql.evaluation.server.app:app", host="0.0.0.0", port=port)
