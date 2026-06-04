from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def normalize_prompt(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def task_fingerprint(prompt: str) -> str:
    norm = normalize_prompt(prompt)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def load_benchmark_fingerprints(data_dir: Path, splits: list[str] | None = None) -> set[str]:
    cached = data_dir / "benchmark_task_fingerprints.json"
    if cached.exists():
        with open(cached) as f:
            return set(json.load(f))
    fps: set[str] = set()
    bench_dir = data_dir / "hf_benchmark"
    if not bench_dir.exists():
        return fps
    for path in sorted(bench_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        if splits and path.stem not in splits:
            continue
        with open(path) as f:
            rows = json.load(f)
        for row in rows:
            fps.add(task_fingerprint(row["query"]))
    return fps


def save_benchmark_fingerprints(data_dir: Path, splits: list[str]) -> Path:
    fps = load_benchmark_fingerprints(data_dir, splits=None)
    if not fps:
        bench_dir = data_dir / "hf_benchmark"
        for split in splits:
            p = bench_dir / f"{split}.json"
            if not p.exists():
                continue
            with open(p) as f:
                for row in json.load(f):
                    fps.add(task_fingerprint(row["query"]))
    out = data_dir / "benchmark_task_fingerprints.json"
    with open(out, "w") as f:
        json.dump(sorted(fps), f, indent=2)
    return out


def dedupe_samples_by_task(
    samples: list[dict],
    exclude_fingerprints: set[str],
    *,
    prompt_key: str = "prompt",
) -> tuple[list[dict], dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    stats = {"input": len(samples), "excluded_benchmark": 0, "excluded_duplicate": 0, "kept": 0}
    for s in samples:
        fp = task_fingerprint(s[prompt_key])
        if fp in exclude_fingerprints:
            stats["excluded_benchmark"] += 1
            continue
        if fp in seen:
            stats["excluded_duplicate"] += 1
            continue
        seen.add(fp)
        kept.append(s)
        stats["kept"] += 1
    return kept, stats
