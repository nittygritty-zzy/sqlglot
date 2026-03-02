"""Extract compact schema strings from SQLite databases for prompt inclusion."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field


@dataclass
class ColumnInfo:
    name: str
    col_type: str
    is_pk: bool


@dataclass
class ForeignKey:
    from_col: str
    ref_table: str
    ref_col: str


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


# SQLite type affinity normalization
_TYPE_MAP = {
    "VARCHAR": "TEXT",
    "NVARCHAR": "TEXT",
    "CHAR": "TEXT",
    "CHARACTER": "TEXT",
    "CLOB": "TEXT",
    "STRING": "TEXT",
    "DOUBLE": "REAL",
    "FLOAT": "REAL",
    "DECIMAL": "REAL",
    "NUMERIC": "REAL",
    "NUMBER": "REAL",
    "BOOLEAN": "INTEGER",
    "BOOL": "INTEGER",
    "INT": "INTEGER",
    "BIGINT": "INTEGER",
    "SMALLINT": "INTEGER",
    "TINYINT": "INTEGER",
    "MEDIUMINT": "INTEGER",
    "INT2": "INTEGER",
    "INT8": "INTEGER",
    "UNSIGNED": "INTEGER",
}


def _normalize_type(raw_type: str) -> str:
    if not raw_type:
        return ""
    # Strip parenthesized size, e.g. VARCHAR(100) -> VARCHAR
    base = raw_type.split("(")[0].strip().upper()
    return _TYPE_MAP.get(base, base)


def extract_db_schema(db_path: str) -> list[TableSchema]:
    """Extract all table schemas from a SQLite database file."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [row[0] for row in cursor.fetchall()]

        tables = []
        for table_name in table_names:
            if table_name.startswith("sqlite_"):
                continue

            # Get columns
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = []
            for row in cursor.fetchall():
                # row: (cid, name, type, notnull, dflt_value, pk)
                columns.append(ColumnInfo(
                    name=row[1],
                    col_type=_normalize_type(row[2]),
                    is_pk=bool(row[5]),
                ))

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            fks = []
            for row in cursor.fetchall():
                # row: (id, seq, table, from, to, on_update, on_delete, match)
                fks.append(ForeignKey(
                    from_col=row[3],
                    ref_table=row[2],
                    ref_col=row[4],
                ))

            tables.append(TableSchema(name=table_name, columns=columns, foreign_keys=fks))

        return tables
    finally:
        conn.close()


def format_schema_compact(tables: list[TableSchema]) -> str:
    """Format table schemas as a compact prompt string."""
    fk_lookup: dict[str, ForeignKey] = {}
    for table in tables:
        for fk in table.foreign_keys:
            fk_lookup[f"{table.name}.{fk.from_col}"] = fk

    lines = []
    for table in tables:
        col_parts = []
        for col in table.columns:
            part = col.name
            if col.col_type:
                part += f" {col.col_type}"
            if col.is_pk:
                part += " PK"
            fk = fk_lookup.get(f"{table.name}.{col.name}")
            if fk:
                part += f" FK->{fk.ref_table}.{fk.ref_col}"
            col_parts.append(part)
        lines.append(f"{table.name}({', '.join(col_parts)})")

    return "\n".join(lines)


def build_schema_cache(db_dirs: list[str]) -> dict[str, str]:
    """Pre-cache {db_id: schema_str} across multiple DB directories."""
    cache: dict[str, str] = {}
    for db_dir in db_dirs:
        if not os.path.isdir(db_dir):
            continue
        for db_id in os.listdir(db_dir):
            if db_id in cache:
                continue
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if not os.path.isfile(db_path):
                continue
            try:
                tables = extract_db_schema(db_path)
                schema_str = format_schema_compact(tables)
                if schema_str:
                    cache[db_id] = schema_str
            except Exception:
                continue
    return cache


def build_tables_cache(db_dirs: list[str]) -> dict[str, list[TableSchema]]:
    """Pre-cache {db_id: list[TableSchema]} across multiple DB directories."""
    cache: dict[str, list[TableSchema]] = {}
    for db_dir in db_dirs:
        if not os.path.isdir(db_dir):
            continue
        for db_id in os.listdir(db_dir):
            if db_id in cache:
                continue
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if not os.path.isfile(db_path):
                continue
            try:
                tables = extract_db_schema(db_path)
                if tables:
                    cache[db_id] = tables
            except Exception:
                continue
    return cache


def build_db_path_cache(db_dirs: list[str]) -> dict[str, str]:
    """Pre-cache {db_id: absolute_path_to_sqlite} across multiple DB directories."""
    cache: dict[str, str] = {}
    for db_dir in db_dirs:
        if not os.path.isdir(db_dir):
            continue
        for db_id in os.listdir(db_dir):
            if db_id in cache:
                continue
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if not os.path.isfile(db_path):
                continue
            cache[db_id] = os.path.abspath(db_path)
    return cache
