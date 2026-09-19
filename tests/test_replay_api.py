import json
import struct
from types import SimpleNamespace

import pytest

import adaptq.replay as replay_api


def _mock_run_factory(stdout):
    calls = []

    def run(command, *, capture_output, text):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run, calls


def test_replay_forwards_output_format_and_path(monkeypatch, tmp_path):
    summary = {
        "strategy": "har_fixed",
        "n_tokens_replayed": 12,
        "wall_time_ms": 1.5,
    }
    run, calls = _mock_run_factory(json.dumps(summary))
    monkeypatch.setattr(replay_api, "_get_binary", lambda: "adapTQ_demo")
    monkeypatch.setattr(replay_api.subprocess, "run", run)

    output_path = tmp_path / "replay.md"
    result = replay_api.ReplayEngine().replay(
        "session.aqss",
        output_format="md",
        output_path=output_path,
    )

    assert result.raw == summary
    assert result.strategy == "har_fixed"
    assert calls == [[
        "adapTQ_demo",
        "replay",
        "session.aqss",
        "--format",
        "md",
        "--output",
        str(output_path),
        "--summary-json",
    ]]


def test_compare_forwards_output_format_and_path(monkeypatch, tmp_path):
    rows = [
        {
            "strategy": "har_fixed",
            "n_tokens": 12,
            "wall_time_ms": 1.5,
            "avg_latency_us": 4.0,
            "avg_quality": 0.98,
            "avg_bits_per_dim": 4.0,
        }
    ]
    run, calls = _mock_run_factory(json.dumps(rows))
    monkeypatch.setattr(replay_api, "_get_binary", lambda: "adapTQ_demo")
    monkeypatch.setattr(replay_api.subprocess, "run", run)

    output_path = tmp_path / "compare.tex"
    result = replay_api.ReplayEngine().compare(
        "session.aqss",
        strategies=["har_fixed"],
        output_format="tex",
        output_path=output_path,
    )

    assert result.rows == rows
    assert calls == [[
        "adapTQ_demo",
        "compare",
        "session.aqss",
        "--strategies",
        "har_fixed",
        "--format",
        "tex",
        "--output",
        str(output_path),
        "--summary-json",
    ]]


@pytest.mark.parametrize("output_format", ["yaml", "html", "", "JSON"])
def test_replay_rejects_unsupported_output_format(monkeypatch, output_format):
    monkeypatch.setattr(replay_api, "_get_binary", lambda: "should-not-run")

    with pytest.raises(ValueError, match="unsupported output_format"):
        replay_api.ReplayEngine().replay(
            "session.aqss",
            output_format=output_format,
        )


def test_replay_rejects_negative_branch_token(monkeypatch):
    monkeypatch.setattr(replay_api, "_get_binary", lambda: (_ for _ in ()).throw(
        AssertionError("negative branch must fail before CLI lookup")
    ))

    with pytest.raises(ValueError, match="from_token must be non-negative"):
        replay_api.ReplayEngine().replay("session.aqss", from_token=-1)


def test_snapshot_info_reads_header_without_replay(monkeypatch, tmp_path):
    snapshot = tmp_path / "session.aqss"
    flags = 1 | 2
    header = struct.pack(
        "<IIiiiiiiQ",
        replay_api._SNAPSHOT_MAGIC,
        replay_api._SNAPSHOT_VERSION,
        2,
        4,
        128,
        4,
        37,
        8,
        flags,
    )
    snapshot.write_bytes(header + b"payload that must not be parsed")

    def fail_binary_lookup():
        raise AssertionError("snapshot_info must not invoke the CLI")

    monkeypatch.setattr(replay_api, "_get_binary", fail_binary_lookup)

    assert replay_api.snapshot_info(snapshot) == {
        "n_tokens": 37,
        "n_layers": 2,
        "n_heads": 4,
        "dim": 128,
        "bits": 4,
        "version": replay_api._SNAPSHOT_VERSION,
        "file_size_bytes": snapshot.stat().st_size,
        "has_token_log": True,
        "has_strategy_state": True,
    }


def test_snapshot_info_rejects_bad_header(tmp_path):
    snapshot = tmp_path / "bad.aqss"
    snapshot.write_bytes(b"AQSS")

    with pytest.raises(RuntimeError, match="truncated snapshot header"):
        replay_api.snapshot_info(snapshot)


def test_snapshot_info_rejects_invalid_magic(tmp_path):
    snapshot = tmp_path / "bad_magic.aqss"
    header = struct.pack(
        "<IIiiiiiiQ",
        0xDEADBEEF,
        replay_api._SNAPSHOT_VERSION,
        1,
        1,
        64,
        4,
        1,
        1,
        0,
    )
    snapshot.write_bytes(header)

    with pytest.raises(RuntimeError, match="invalid magic"):
        replay_api.snapshot_info(snapshot)


def test_snapshot_to_json_exports_json_and_file(tmp_path):
    snapshot = tmp_path / "session.aqss"
    header = struct.pack(
        "<IIiiiiiiQ",
        replay_api._SNAPSHOT_MAGIC,
        replay_api._SNAPSHOT_VERSION,
        2,
        4,
        128,
        4,
        37,
        8,
        3,
    )
    snapshot.write_bytes(header)

    json_str = replay_api.snapshot_to_json(snapshot)
    data = json.loads(json_str)
    assert data["n_tokens"] == 37
    assert data["dim"] == 128
    assert data["bits"] == 4
    assert data["version"] == replay_api._SNAPSHOT_VERSION

    out_file = tmp_path / "info.json"
    json_str2 = replay_api.snapshot_to_json(snapshot, json_path=out_file)
    assert json_str2 == json_str
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == data
