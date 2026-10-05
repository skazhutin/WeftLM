import json

import mlx.core as mx
import pytest
from test_distributed import launch

from weftlm.benchmark import benchmark_local
from weftlm.distributed_benchmark import benchmark_distributed
from weftlm.fixtures import AttentionConfig
from weftlm.golden import load_reference
from weftlm.results import maximum_rank_samples, summarize_directory, summarize_samples


def benchmark_arguments(mode, directory, *, length=7):
    return [
        "bench",
        "--mode",
        mode,
        "--device",
        "cpu",
        "--lengths",
        str(length),
        "--hq",
        "4",
        "--hkv",
        "2",
        "--dim",
        "8",
        "--block-size",
        "2",
        "--warmup",
        "1",
        "--repeats",
        "3",
        "--topology",
        "local-processes",
        "--link-description",
        "loopback TCP",
        "--output",
        str(directory),
    ]


@pytest.mark.parametrize("mode", ["context", "heads"])
def test_distributed_benchmark_keeps_raw_participants_and_global_maximum(
    tmp_path, mode
):
    output = tmp_path / mode
    rows = launch(*benchmark_arguments(mode, output))
    assert sorted(row["rank"] for row in rows) == [0, 1]
    records, identifiers = [], []
    for rank in range(2):
        directory = output / f"rank-{rank}" / f"{mode}-7"
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["error"]["passed"] and metadata["all_participants_passed"]
        assert metadata["topology"] == "local-processes"
        assert metadata["source_fingerprint"]
        assert metadata["latency_scope"] == "max_participant_per_iteration"
        assert metadata["payload_bytes_per_rank"] == (160 if mode == "context" else 32)
        assert not metadata["wire_traffic_measured"]
        identifiers.append(metadata["run_id"])
        samples = [
            json.loads(line)
            for line in (directory / "samples.jsonl").read_text().splitlines()
        ]
        assert len(samples) == 3
        for row in samples:
            assert row["elapsed_ns"] == max(row["participants_elapsed_ns"])
            assert row["rank_elapsed_ns"] == row["participants_elapsed_ns"][rank]
        records.append([row["participants_elapsed_ns"] for row in samples])
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        assert set(diagnostics) == {
            "local_compute",
            "communication",
            "merge_or_assembly",
            "barrier",
        }
        for phase in diagnostics.values():
            assert (
                maximum_rank_samples(phase["participants_samples_ns"])
                == phase["global_samples_ns"]
            )
            assert summarize_samples(phase["global_samples_ns"]) == phase["summary"]
    assert identifiers[0] == identifiers[1]
    assert records[0] == records[1]
    summaries = summarize_directory(output)
    assert len(summaries) == 2
    assert summaries[0]["median_ms"] == summaries[1]["median_ms"]
    # A corrupt total cannot be accepted just because it looks like a valid time.
    path = output / "rank-0" / f"{mode}-7" / "samples.jsonl"
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    samples[0]["elapsed_ns"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in samples))
    with pytest.raises(ValueError, match="Invalid participant reduction"):
        summarize_directory(output)


def test_maximum_is_reduced_before_summary():
    participants = [[1, 9, 1], [9, 1, 1]]
    assert maximum_rank_samples(participants) == [9, 9, 1]
    assert summarize_samples(maximum_rank_samples(participants))["median_ms"] == 9e-6
    assert max(summarize_samples(row)["median_ms"] for row in participants) == 1e-6
    with pytest.raises(ValueError, match="counts must match"):
        maximum_rank_samples([[1], [1, 2]])


def test_large_context_uses_verified_small_reference_artifact(tmp_path):
    config = AttentionConfig(length=4097, hq=4, hkv=2, dim=8, block_size=2)
    baseline = tmp_path / "baseline" / "single-4097"
    benchmark_local(
        config, mode="single", output=baseline, device=mx.cpu, warmup=0, repeats=1
    )
    expected, digest = load_reference(baseline, config)
    assert expected.shape == (1, 4, 1, 8)
    output = tmp_path / "distributed"
    # Larger kernel blocks are permitted: they do not change canonical inputs.
    arguments = benchmark_arguments("context", output, length=4097)
    arguments[arguments.index("--block-size") + 1] = "512"
    rows = launch(*arguments, "--reference", str(baseline.parent))
    assert len(rows) == 2
    for rank in range(2):
        metadata = json.loads(
            (output / f"rank-{rank}" / "context-4097" / "metadata.json").read_text()
        )
        assert metadata["error"]["passed"]
        assert metadata["reference_identity"] == digest
    with pytest.raises(ValueError, match="does not match"):
        load_reference(
            baseline, AttentionConfig(length=4097, hq=4, hkv=2, dim=8, seed=1)
        )


def test_large_context_cannot_build_full_reference_on_a_distributed_rank(tmp_path):
    with pytest.raises(ValueError, match="require --reference"):
        benchmark_distributed(
            AttentionConfig(length=4097),
            group=None,
            mode="context",
            output=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()
