"""Repository paths and compact experiment I/O."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ("qwen3-8b", "qwen3-4b")
STRUCTURAL = (
    "baseline",
    "no-gold",
    "sub-operations",
    "sub-operations-no-gold",
    "sub-operations-no-first",
    "sub-operations-first-only",
    "sub-operations-no-last",
)
SEMANTIC = (
    "state-gold-only",
    "state-gold-decoy",
    "state-decoy-failure",
    "state-decoy-success",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=8)
def load_registry(family, data=DATA):
    # Sessions clone the initial state; immutable tool definitions can be reused.
    from envtoolbench.v2.registry import build_v2_registry
    from envtoolbench.v2.stateful_operations import build_stateful_registry

    settings = read_json(Path(data) / family / "registry.json")
    if family == "structural":
        return build_v2_registry(seed=settings["seed"], domains=settings["domains"])
    return build_stateful_registry(seed=settings["seed"])


def model_runtime(model, model_dir=None):
    from stage1.runtime import LocalTransformersRuntime

    settings = read_json(ROOT / "stage1/models.json")[model]
    return LocalTransformersRuntime(
        model_key=model,
        model_path=model_dir or ROOT / settings["local_path"],
        protocol="qwen3",
        max_context_tokens=32768,
        max_new_tokens=512,
        attn_implementation="sdpa",
    )
