import json
from pathlib import Path

from src.experiments import build_run_manifest, file_sha256, write_run_manifest


def test_manifest_hashes_inputs_and_serializes(tmp_path: Path):
    panel = tmp_path / "panel.csv"
    benchmark = tmp_path / "benchmark.csv"
    panel.write_text("date,ticker\n2024-01-01,ABC\n", encoding="utf-8")
    benchmark.write_text("date,close\n2024-01-01,100\n", encoding="utf-8")

    manifest = build_run_manifest(
        parameters={"model": "ridge"},
        input_paths={"panel": panel, "benchmark": benchmark},
        repository=tmp_path,
    )
    output = tmp_path / "run_manifest.json"
    write_run_manifest(manifest, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["parameters"] == {"model": "ridge"}
    assert saved["inputs"]["panel"]["sha256"] == file_sha256(panel)
    assert saved["git_revision"] is None


def test_manifest_captures_signal_screen_assumptions(tmp_path: Path):
    panel = tmp_path / "panel.csv"
    benchmark = tmp_path / "benchmark.csv"
    panel.write_text("date,ticker\\n2024-01-01,ABC\\n", encoding="utf-8")
    benchmark.write_text("date,close\\n2024-01-01,100\\n", encoding="utf-8")
    parameters = {
        "signals": "short_horizon_reversal",
        "include_neutralized": True,
        "cost_bps": 10.0,
        "execution": "next_session_open",
        "label_horizon_days": 5,
    }
    manifest = build_run_manifest(
        parameters=parameters,
        input_paths={"panel": panel, "benchmark": benchmark},
        repository=tmp_path,
    )
    assert manifest["parameters"] == parameters
    assert set(manifest["inputs"]) == {"panel", "benchmark"}
