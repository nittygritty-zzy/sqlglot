"""Benchmark question loading routes."""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Loaded at startup
_questions: list[dict] = []


def load_spider_dev(spider_dir: str = "data/spider"):
    """Load Spider 1.0 dev questions."""
    global _questions
    dev_path = os.path.join(spider_dir, "dev.json")
    if not os.path.exists(dev_path):
        logger.warning(f"Spider dev.json not found at {dev_path}")
        return

    with open(dev_path) as f:
        raw = json.load(f)

    _questions = []
    for i, item in enumerate(raw):
        _questions.append({
            "id": f"spider_dev_{i}",
            "question": item["question"],
            "db_id": item["db_id"],
            "gold_sql": item["query"],
            "source": "spider_dev",
        })

    logger.info(f"Loaded {len(_questions)} Spider dev questions")


@router.get("/questions")
def get_questions(
    limit: int = Query(default=0, ge=0),
    offset: int = Query(default=0, ge=0),
):
    """Return benchmark questions with optional pagination."""
    subset = _questions[offset:]
    if limit > 0:
        subset = subset[:limit]
    return {"questions": subset, "total": len(_questions)}
