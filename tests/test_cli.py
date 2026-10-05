import json

import pytest

from weftlm.__main__ import main
from weftlm.results import summarize_directory, summarize_samples


def test_cpu_cli_produces_recomputable_raw_artifacts(tmp_path, capsys):
    output = tmp_path / "run"
    assert (
        main(
            [
                "bench",
                "--device",
                "cpu",
                "--mode",
                "local-context",
                "--lengths",
                "8",
                "--hq",
                "4",
                "--hkv",
                "2",
                "--dim",
                "8",
                "--warmup",
                "1",
                "--repeats",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    metadata = json.loads((output / "local-context-8" / "metadata.json").read_text())
    assert metadata["error"]["passed"]
    assert metadata["world_size"] == 1
    assert metadata["device"] == "cpu"
    summary = summarize_directory(output)[0]
    assert summary["repeats"] == 3
    assert summary["p95_ms"] >= summary["median_ms"] > 0
    assert main(["summarize", "--input", str(output)]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["bench", "--lengths", "8", "--output", str(output)])


def test_summary_nearest_rank_p95():
    summary = summarize_samples([i * 1_000_000 for i in range(1, 21)])
    assert summary["median_ms"] == 10.5
    assert summary["p95_ms"] == 19


def test_check_command_and_invalid_input(capsys, tmp_path):
    assert main(["check", "--device", "cpu", "--length", "7", "--dim", "8"]) == 0
    assert all(
        value["passed"] for value in json.loads(capsys.readouterr().out).values()
    )
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["bench", "--lengths", "0", "--output", str(output)])
    assert not output.exists()


def test_truncated_measurements_are_rejected(tmp_path, capsys):
    output = tmp_path / "run"
    main(
        [
            "bench",
            "--device",
            "cpu",
            "--lengths",
            "2",
            "--hq",
            "1",
            "--hkv",
            "1",
            "--dim",
            "4",
            "--warmup",
            "0",
            "--repeats",
            "2",
            "--output",
            str(output),
        ]
    )
    path = output / "single-2" / "samples.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_directory(output)
