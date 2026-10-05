"""Small reference outputs for validation without full KV on distributed ranks."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from .fixtures import AttentionConfig


def write_reference(directory: Path, config: AttentionConfig, expected: mx.array):
    artifact = {
        "format_version": 1,
        "fixture_version": 1,
        "source": "independent-full-reference",
        "config": asdict(config),
        "values": expected.astype(mx.float32).tolist(),
    }
    (directory / "reference.json").write_text(json.dumps(artifact) + "\n")


def load_reference(directory: Path, config: AttentionConfig) -> tuple[mx.array, str]:
    if not (directory / "metadata.json").is_file():
        directory = directory / f"single-{config.length}"
    metadata = json.loads((directory / "metadata.json").read_text())
    content = (directory / "reference.json").read_bytes()
    artifact = json.loads(content)
    if not all(isinstance(value, dict) for value in (metadata, artifact)):
        raise ValueError("Reference metadata must be JSON objects")
    if (
        not isinstance(metadata.get("config"), dict)
        or not isinstance(metadata.get("error"), dict)
        or not isinstance(artifact.get("config"), dict)
        or not isinstance(artifact.get("values"), list)
    ):
        raise ValueError("Reference artifact is missing required fields")
    identity = ("length", "hq", "hkv", "dim", "dtype", "seed")
    if (
        metadata.get("mode") != "single"
        or metadata["error"].get("passed") is not True
        or metadata.get("fixture_version") != 1
        or artifact.get("format_version") != 1
        or artifact.get("fixture_version") != 1
        or artifact.get("source") != "independent-full-reference"
        or any(
            artifact.get("config", {}).get(name) != getattr(config, name)
            or metadata.get("config", {}).get(name) != getattr(config, name)
            for name in identity
        )
    ):
        raise ValueError(
            "Reference artifact does not match a verified single-mode fixture"
        )
    with mx.stream(mx.cpu):
        expected = mx.array(artifact["values"], dtype=config.mlx_dtype)
        if expected.shape != (1, config.hq, 1, config.dim):
            raise ValueError("Reference output has the wrong shape")
        if not mx.all(mx.isfinite(expected)).item():
            raise ValueError("Reference output must be finite")
    return expected, hashlib.sha256(content).hexdigest()
