"""Raw experiment artifacts and transparent summary calculations."""

import csv
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
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "mlx": version("mlx"),
        "weftlm": version("weftlm"),
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "tracked_source_dirty": bool(
            command_output(
                ["git", "diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"]
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


def write_results(directory: Path, metadata: dict, samples_ns: list[int]) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata)
    metadata["peak_process_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin"
        else None
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = [{"iteration": i, "elapsed_ns": ns} for i, ns in enumerate(samples_ns)]
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
        }
    )
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def summarize_directory(directory: Path) -> list[dict]:
    summaries = []
    for path in sorted(directory.rglob("samples.jsonl")):
        metadata = json.loads((path.parent / "metadata.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != metadata["repeats"]:
            raise ValueError(f"Incomplete measurement series in {path}")
        if [row["iteration"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Duplicate or missing iterations in {path}")
        summary = summarize_samples([row["elapsed_ns"] for row in rows])
        summary.update(
            {
                "mode": metadata["mode"],
                "length": metadata["config"]["length"],
                "dtype": metadata["config"]["dtype"],
                "rank": metadata.get("rank", 0),
                "path": str(path.parent),
            }
        )
        summaries.append(summary)
    if not summaries:
        raise ValueError("No raw timing artifacts found")
    return summaries
