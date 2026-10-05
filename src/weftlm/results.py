"""Raw experiment artifacts and transparent summary calculations."""

import csv
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment_metadata() -> dict:
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "hardware_model": command_output(["sysctl", "-n", "hw.model"]),
        "source_fingerprint": source.hexdigest(),
        "mlx": version("mlx"),
        "weftlm": version("weftlm"),
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "tracked_source_dirty": bool(
            command_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "pyproject.toml",
                    "uv.lock",
                ]
            )
        ),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "swap": command_output(["sysctl", "-n", "vm.swapusage"]),
    }


def summarize_samples(samples_ns: list[int]) -> dict:
    if not samples_ns or any(not isinstance(x, int) or x <= 0 for x in samples_ns):
        raise ValueError("Timing samples must be positive integer nanoseconds")
    ordered = sorted(samples_ns)
    return {
        "repeats": len(ordered),
        "median_ms": statistics.median(ordered) / 1e6,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1] / 1e6,
        "min_ms": ordered[0] / 1e6,
    }


def maximum_rank_samples(per_rank: list[list[int]]) -> list[int]:
    if not per_rank or not per_rank[0]:
        raise ValueError("Participant samples cannot be empty")
    for samples in per_rank:
        if len(samples) != len(per_rank[0]):
            raise ValueError("Participant measurement counts must match")
        summarize_samples(samples)
    return [max(values) for values in zip(*per_rank, strict=True)]


def write_results(
    directory: Path,
    metadata: dict,
    samples_ns: list[int],
    *,
    per_rank_samples: list[list[int]] | None = None,
) -> dict:
    if per_rank_samples is not None:
        if len(per_rank_samples) != metadata["world_size"]:
            raise ValueError("Participant count must match world size")
        if maximum_rank_samples(per_rank_samples) != samples_ns:
            raise ValueError(
                "Global samples must be the per-iteration participant maximum"
            )
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata)
    metadata["peak_process_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin"
        else None
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = [{"iteration": i, "elapsed_ns": ns} for i, ns in enumerate(samples_ns)]
    if per_rank_samples is not None:
        for i, row in enumerate(rows):
            row["participants_elapsed_ns"] = [
                samples[i] for samples in per_rank_samples
            ]
            row["rank_elapsed_ns"] = per_rank_samples[metadata["rank"]][i]
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = summarize_samples(samples_ns)
    summary.update(
        {
            "mode": metadata["mode"],
            "length": metadata["config"]["length"],
            "dtype": metadata["config"]["dtype"],
            "rank": metadata.get("rank", 0),
            "latency_scope": metadata.get("latency_scope", "local_operation"),
            "correctness_passed": metadata.get(
                "all_participants_passed", metadata["error"]["passed"]
            ),
        }
    )
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    return summary


def write_failure(directory: Path, metadata: dict, error: Exception) -> None:
    """Keep a failed configuration visible without inventing timing samples."""
    directory.mkdir(parents=True, exist_ok=False)
    record = {**metadata, "status": "failed", "reason": str(error)}
    (directory / "failure.json").write_text(json.dumps(record, indent=2) + "\n")


def summarize_directory(directory: Path) -> list[dict]:
    summaries = []
    for path in sorted(directory.rglob("samples.jsonl")):
        metadata = json.loads((path.parent / "metadata.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != metadata["repeats"]:
            raise ValueError(f"Incomplete measurement series in {path}")
        if [row["iteration"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Duplicate or missing iterations in {path}")
        if metadata.get("latency_scope") == "max_participant_per_iteration":
            for row in rows:
                samples = row["participants_elapsed_ns"]
                if (
                    len(samples) != metadata["world_size"]
                    or maximum_rank_samples([[ns] for ns in samples])[0]
                    != row["elapsed_ns"]
                    or samples[metadata["rank"]] != row["rank_elapsed_ns"]
                ):
                    raise ValueError(f"Invalid participant reduction in {path}")
        summary = summarize_samples([row["elapsed_ns"] for row in rows])
        summary.update(
            {
                "mode": metadata["mode"],
                "length": metadata["config"]["length"],
                "dtype": metadata["config"]["dtype"],
                "rank": metadata.get("rank", 0),
                "latency_scope": metadata.get("latency_scope", "local_operation"),
                "correctness_passed": metadata.get(
                    "all_participants_passed", metadata["error"]["passed"]
                ),
                "path": str(path.parent),
            }
        )
        summaries.append(summary)
    for path in sorted(directory.rglob("failure.json")):
        record = json.loads(path.read_text())
        summaries.append(
            {
                "status": record["status"],
                "reason": record["reason"],
                "mode": record["mode"],
                "length": record["config"]["length"],
                "rank": record["rank"],
                "path": str(path.parent),
            }
        )
    if not summaries:
        raise ValueError("No raw timing artifacts found")
    return summaries
