"""Offline checks for the window profile's input and measurement boundaries."""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def profiler(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "profile_ader_window", SCRIPTS / "profile_ader_window.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_inputs(profiler, directory, *periods):
    profiler.DATA_HASHES = {}
    for period, body in enumerate(periods):
        name = f"period_{period}.txt"
        (directory / name).write_bytes(body)
        profiler.DATA_HASHES[name] = hashlib.sha1(
            b"blob " + str(len(body)).encode() + b"\0" + body
        ).hexdigest()


def test_audit_tracks_filtered_contexts_and_rejects_tampering(profiler, tmp_path):
    write_inputs(profiler, tmp_path, b"1 1\n1 2\n2 3\n", b"3 1\n3 4\n3 2\n4 4\n5 3\n")
    counts = profiler.audit_data(tmp_path)["counts"]
    assert counts["train_actions"] == 3
    assert counts["train_singletons"] == 1
    assert counts["train_prefixes_before_validation"] == 1
    assert counts["upstream_sampler_rows_before_validation"] == 2
    assert counts["active_items"] == 3
    assert counts["test_raw_actions"] == 5
    assert counts["test_filtered_actions"] == 2
    assert counts["test_filtered_sessions"] == counts["test_filtered_prefixes"] == 1
    with (tmp_path / "period_1.txt").open("ab") as handle:
        handle.write(b"6 1\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        profiler.audit_data(tmp_path)


def test_audit_refuses_noncontiguous_candidates(profiler, tmp_path):
    write_inputs(profiler, tmp_path, b"1 1\n1 3\n", b"2 1\n2 3\n")
    with pytest.raises(ValueError, match="must cover"):
        profiler.audit_data(tmp_path)


def test_gpu_measurements_ignore_other_processes_and_preserve_unknown(profiler):
    assert profiler.gpu_sample("12, GPU-A, 123\n13, GPU-B, 9999\n12, GPU-C, [N/A]\n", 12) == [
        {"gpu_uuid": "GPU-A", "used_mib": 123.0},
        {"gpu_uuid": "GPU-C", "used_mib": None},
    ]


@pytest.mark.parametrize("exit_code", [0, 3])
def test_monitor_retains_process_failure_and_unknown_memory(
    profiler, tmp_path, monkeypatch, exit_code
):
    monkeypatch.setattr(profiler, "COMMAND", ["-c", f"import sys; sys.exit({exit_code})"])
    monkeypatch.setattr(profiler.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    report = profiler.run_training(tmp_path, tmp_path)
    assert report["returncode"] == exit_code
    assert report["sampled_peak_training_gpu_mib"] is None
    assert report["seconds"] > 0


def test_existing_profile_output_is_preserved(tmp_path):
    marker = tmp_path / "manifest.json"
    marker.write_text("existing evidence", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "profile_ader_window.py"),
            "--source",
            "absent",
            "--data",
            "absent",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "Use a new output directory" in result.stderr
    assert marker.read_text(encoding="utf-8") == "existing evidence"


def test_two_window_audit_grows_candidates_without_test_lookahead(profiler, tmp_path):
    write_inputs(profiler, tmp_path, b"1 1\n1 2\n", b"2 2\n2 3\n", b"3 3\n3 4\n3 1\n")
    first, second = profiler.audit_data(tmp_path, 2)["counts"]
    assert first["active_items"] == 2 and first["test_filtered_prefixes"] == 0
    assert second["active_items"] == 3 and second["test_filtered_prefixes"] == 1
    # A directory can contain future files; one-window preparation must ignore them.
    assert len(profiler.audit_data(tmp_path)["files"]) == 2


def profile_fixture(directory, method="ader", windows=2):
    log = []
    for period in range(1, windows + 1):
        log.extend(
            [
                f"Period {period}:",
                "epoch:1, test (MRR@20: 0.1, RECALL@20: 0.5, MRR@10: 0.1, RECALL@10: 0.4)",
            ]
        )
        folder = directory / f"profile-period{period}"
        folder.mkdir()
        values = {
            "continuation": {
                "old_items": 0 if period == 1 else period + 1,
                "active_items": period + 2,
                "training_rows": 10,
                "batches_per_epoch": 2,
                "input_replay_size": 0 if period == 1 else 4,
                "replay_batch_size": 0 if period == 1 else 2,
                "lambda": 0
                if period == 1
                else 0.8 * math.sqrt((period + 1) / (period + 2) * 4 / 10),
                "restored_global_step": (period - 1) * 2,
            },
            "updates": {
                "global_step_after_training": (period - 1) * 2 + 4,
                "distillation_updates": 0 if period == 1 else 4,
            },
            "selection": {"selected_epoch": 1, "selected_global_step": period * 2},
            "herding": {"actual_replay_size": 4, "active_items": period + 2, "seconds": 1},
        }
        values["metrics"] = {
            "MRR@20": 0.1000123,
            "Recall@20": 0.5000123,
            "MRR@10": 0.1,
            "Recall@10": 0.4,
        }
        values["split"] = {
            "candidate_ids": list(range(1, period + 3)),
            "validation_prefixes": [[1, 2]],
        }
        if method == "dropout":
            values["continuation"].update(
                input_replay_size=0, replay_batch_size=0, **{"lambda": 0.0}
            )
            values["updates"]["distillation_updates"] = 0
            values["herding"]["actual_replay_size"] = 0
        for name, value in values.items():
            (folder / f"{name}.json").write_text(json.dumps(value))
        (folder / "epochs.jsonl").write_text(
            "\n".join(
                json.dumps({"epoch": e, "seconds": 1, "validation_recall20": v})
                for e, v in [(1, 0.5), (2, 0.4)]
            )
        )
    (directory / "Training_logs.txt").write_text("\n".join(log) + "\nDone.")


def test_two_window_report_checks_selected_restore_and_completed_kd(profiler, tmp_path):
    profile_fixture(tmp_path)
    reports = profiler.collect_results(tmp_path, 2)
    assert [r["test_period"] for r in reports] == [1, 2]
    assert reports[1]["updates"]["distillation_updates"] == 4
    assert reports[1]["continuation"]["restored_global_step"] == 2


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("continuation", "restored_global_step", 4),
        ("continuation", "input_replay_size", 3),
        ("continuation", "lambda", 0.0),
        ("updates", "distillation_updates", 0),
    ],
)
def test_two_window_report_rejects_broken_continuation(profiler, tmp_path, file, key, value):
    profile_fixture(tmp_path)
    path = tmp_path / "profile-period2" / f"{file}.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        profiler.collect_results(tmp_path, 2)


def test_report_rejects_incomplete_second_window(profiler, tmp_path):
    profile_fixture(tmp_path)
    path = tmp_path / "Training_logs.txt"
    path.write_text(path.read_text().removesuffix("Done."))
    with pytest.raises(RuntimeError, match="Incomplete"):
        profiler.collect_results(tmp_path, 2)


@pytest.mark.parametrize("method", ["ader", "dropout"])
def test_all_windows_report_exact_metrics_and_unweighted_mean(profiler, tmp_path, method):
    profile_fixture(tmp_path, method, 16)
    reports = profiler.collect_results(tmp_path, 16, method)
    assert len(reports) == 16
    assert reports[-1]["selection"]["selected_global_step"] == 32
    assert reports[0]["test_metrics"]["Recall@20"] == 0.5000123
    # Unequal query counts must not change the paper's window-weighted aggregation.
    reports[0]["test_metrics"]["Recall@20"] = 0.0
    assert profiler.mean_metrics(reports)["Recall@20"] == pytest.approx(15 * 0.5000123 / 16)


def test_dropout_launch_only_changes_method_flag_and_output_label(profiler):
    command = profiler.build_command("dropout")
    assert command[-2:] == ["--dropout", "True"]
    command = command[:-2]
    command[command.index("--save_dir") + 1] = "ADER"
    assert command == profiler.build_command("ader") == profiler.COMMAND


def test_dropout_report_rejects_hidden_replay(profiler, tmp_path):
    profile_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="must not use replay"):
        profiler.collect_results(tmp_path, 2, "dropout")
