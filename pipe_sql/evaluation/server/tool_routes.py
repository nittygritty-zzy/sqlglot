"""Tool execution routes reusing pipe_sql/training/tool_executor.py functions."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipe_sql.training.schema_extractor import TableSchema
from pipe_sql.training.tool_executor import (
    describe_table_result,
    execute_pipe_sql_result,
    list_tables_result,
    sample_data_result,
    validate_pipe_sql_result,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])

# These are populated at startup by app.py
tables_cache: dict[str, list[TableSchema]] = {}
db_path_cache: dict[str, str] = {}


class ToolRequest(BaseModel):
    arguments: dict


class ToolResponse(BaseModel):
    result: str


def _get_tables(db_id: str) -> list[TableSchema]:
    if db_id not in tables_cache:
        raise HTTPException(status_code=404, detail=f"Unknown db_id: {db_id}")
    return tables_cache[db_id]


def _get_db_path(db_id: str) -> str:
    if db_id not in db_path_cache:
        raise HTTPException(status_code=404, detail=f"Unknown db_id: {db_id}")
    return db_path_cache[db_id]


@router.post("/list_tables", response_model=ToolResponse)
def tool_list_tables(req: ToolRequest):
    db_id = req.arguments.get("db_id", "")
    tables = _get_tables(db_id)
    return ToolResponse(result=list_tables_result(tables))


@router.post("/describe_table", response_model=ToolResponse)
def tool_describe_table(req: ToolRequest):
    db_id = req.arguments.get("db_id", "")
    table_name = req.arguments.get("table_name", "")
    tables = _get_tables(db_id)
    return ToolResponse(result=describe_table_result(tables, table_name))


@router.post("/sample_data", response_model=ToolResponse)
def tool_sample_data(req: ToolRequest):
    db_id = req.arguments.get("db_id", "")
    table_name = req.arguments.get("table_name", "")
    limit = req.arguments.get("limit", 5)
    db_path = _get_db_path(db_id)
    return ToolResponse(result=sample_data_result(db_path, table_name, limit))


@router.post("/execute_pipe_sql", response_model=ToolResponse)
def tool_execute_pipe_sql(req: ToolRequest):
    db_id = req.arguments.get("db_id", "")
    pipe_sql = req.arguments.get("pipe_sql", "")
    db_path = _get_db_path(db_id)
    return ToolResponse(result=execute_pipe_sql_result(db_path, pipe_sql))


@router.post("/validate_pipe_sql", response_model=ToolResponse)
def tool_validate_pipe_sql(req: ToolRequest):
    pipe_sql = req.arguments.get("pipe_sql", "")
    return ToolResponse(result=validate_pipe_sql_result(pipe_sql))
