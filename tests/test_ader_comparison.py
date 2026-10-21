"""Completed-window aggregation and comparison provenance boundaries."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "compare_ader_runs", scripts / "compare_ader_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runs(module, directory):
    for method, value in (("ader", 0.25), ("dropout", 0.20)):
        run = directory / method
        run.mkdir()
        windows = [
            {
                "train_period": i,
                "test_period": i + 1,
                "test_metrics": dict.fromkeys(module.PAPER[method], value),
                "validation_sha256": "shared" if i == 0 else method,
                "continuation": {"active_items": 100 + i},
            }
            for i in range(16)
        ]
        report = {
            "status": "passed",
            "returncode": 0,
            "method": method,
            "windows": windows,
            "mean_test_metrics": module.mean_metrics(windows),
        }
        manifest = {
            "method": method,
            "windows": 16,
            "dataset": "DIGINETICA",
            "command": module.build_command(method),
            "source_commit": "test-fixture",
            "source_hashes": {},
            "input_audit": {},
            "script_sha256": {},
            "packages": {},
            "deterministic_gpu_ops_enforced": False,
        }
        (run / "profile.json").write_text(json.dumps(report))
        (run / "manifest.json").write_text(json.dumps(manifest))
    return directory / "ader", directory / "dropout"


def test_comparison_percentage_points_and_split_disclosure(comparison, tmp_path):
    ader, dropout = make_runs(comparison, tmp_path)
    result = comparison.compare(ader, dropout)
    assert result["ader_minus_dropout_percentage_points"]["Recall@20"] == pytest.approx(5.0)
    assert result["difference_from_paper_percentage_points"]["ader"]["Recall@20"] == pytest.approx(
        -25.21
    )
    assert result["validation_splits_identical_by_test_period"] == {
        str(i): i == 1 for i in range(1, 17)
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("windows", 15),
        ("input_audit", {"changed": True}),
        ("packages", {"tensorflow": "different"}),
    ],
)
def test_comparison_rejects_incompatible_manifest(comparison, tmp_path, field, value):
    ader, dropout = make_runs(comparison, tmp_path)
    path = dropout / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        comparison.compare(ader, dropout)


def test_comparison_rejects_missing_period(comparison, tmp_path):
    ader, dropout = make_runs(comparison, tmp_path)
    path = ader / "profile.json"
    report = json.loads(path.read_text())
    report["windows"].pop(5)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Incomplete"):
        comparison.compare(ader, dropout)
