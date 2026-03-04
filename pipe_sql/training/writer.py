"""Write output JSONL files with train/dev split and statistics."""

from __future__ import annotations

import json
import os
import random
from collections import Counter

from pipe_sql.training.formatter import ChatSample


def _split_by_question_id(
    samples: list[ChatSample],
    train_ratio: float,
    seed: int,
) -> tuple[list[ChatSample], list[ChatSample]]:
    """Split samples into train/dev by question_id, stratified by db_id."""
    # Group question_ids by db_id
    qid_to_dbid: dict[str, str] = {}
    for s in samples:
        qid = s.metadata["question_id"]
        if qid not in qid_to_dbid:
            qid_to_dbid[qid] = s.metadata["db_id"]

    dbid_to_qids: dict[str, list[str]] = {}
    for qid, dbid in qid_to_dbid.items():
        dbid_to_qids.setdefault(dbid, []).append(qid)

    rng = random.Random(seed)
    train_qids: set[str] = set()
    dev_qids: set[str] = set()

    for dbid in sorted(dbid_to_qids):
        qids = sorted(dbid_to_qids[dbid], key=str)
        rng.shuffle(qids)
        split_idx = max(1, int(len(qids) * train_ratio))
        # Ensure at least one in train per db
        train_qids.update(qids[:split_idx])
        dev_qids.update(qids[split_idx:])

    train = [s for s in samples if s.metadata["question_id"] in train_qids]
    dev = [s for s in samples if s.metadata["question_id"] in dev_qids]
    return train, dev


def _sample_to_dict(sample: ChatSample) -> dict:
    d = {"messages": sample.messages, "metadata": sample.metadata}
    if sample.tools is not None:
        d["tools"] = sample.tools
    return d


def _compute_stats(
    train: list[ChatSample],
    dev: list[ChatSample],
    errors_skipped: int,
    missing_db: int,
) -> dict:
    all_samples = train + dev
    op_counter: Counter = Counter()
    steps_per_query: Counter = Counter()
    db_counter: Counter = Counter()

    tool_calling_count = 0
    seen_qids: set[str] = set()
    for s in all_samples:
        if s.metadata.get("sample_type") == "tool_calling":
            tool_calling_count += 1
        else:
            op_counter[s.metadata["op_type"]] += 1
        db_counter[s.metadata["db_id"]] += 1
        qid = s.metadata["question_id"]
        if qid not in seen_qids:
            seen_qids.add(qid)
            total_steps = s.metadata.get("total_steps")
            if total_steps is not None:
                steps_per_query[total_steps] += 1

    return {
        "total_samples": len(all_samples),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "total_queries": len(seen_qids),
        "errors_skipped": errors_skipped,
        "missing_db": missing_db,
        "operator_distribution": dict(sorted(op_counter.items())),
        "steps_per_query_distribution": {
            str(k): v for k, v in sorted(steps_per_query.items())
        },
        "database_count": len(db_counter),
        "tool_calling_samples": tool_calling_count,
    }


def write_output(
    samples: list[ChatSample],
    output_dir: str,
    train_ratio: float = 0.95,
    seed: int = 42,
    errors_skipped: int = 0,
    missing_db: int = 0,
) -> dict:
    """Write train.jsonl, dev.jsonl, and stats.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    train, dev = _split_by_question_id(samples, train_ratio, seed)

    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for s in train:
            f.write(json.dumps(_sample_to_dict(s)) + "\n")

    dev_path = os.path.join(output_dir, "dev.jsonl")
    with open(dev_path, "w") as f:
        for s in dev:
            f.write(json.dumps(_sample_to_dict(s)) + "\n")

    stats = _compute_stats(train, dev, errors_skipped, missing_db)

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
