"""Profile up to 16 DIGINETICA windows with pinned ADER or Dropout code."""

import argparse
import csv
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from verify_ader import COMMIT, SOURCE_HASHES, prepare_main, verify_source

DATA_HASHES = {
    "period_0.txt": "5f2f70ec5544f82ee74b8df78b77334f637e7883",
    "period_1.txt": "c02fc6de4fe15a41f4d686c78cb6d5b43259abb6",
    "period_2.txt": "3e1fe1dde546f4bc98df2d60a3dee1065fbf0020",
    "period_3.txt": "358f973e8f2bc978184c501f7b430a2bb95830c7",
    "period_4.txt": "1a6411b93123d36ce9fe40fda17f0caff999a022",
    "period_5.txt": "7c15b9188740a4c6b9c05bc60fb77779e34de2d9",
    "period_6.txt": "1fdbe4c191e93639faf1dcb8f4483a29ebbd6048",
    "period_7.txt": "38d822a97e3602e5ff84f4bb2b0ec94ab8e5732f",
    "period_8.txt": "02985b7ef93aa20b9ed4b7f0f4fc4bfce213e226",
    "period_9.txt": "d6cca2cdf19ad5bcd492d6be8670eb73fa60d998",
    "period_10.txt": "f986aaf370a6af41d6b8e871b33d621c8c5106be",
    "period_11.txt": "6cede818ce23391e637248d26e7c45039d837d22",
    "period_12.txt": "9fdeca38d8f31080c2d748c3357ce9a711e02e52",
    "period_13.txt": "9866fcd9db5e898cfa771858d902b8527f1abdaa",
    "period_14.txt": "d47c32712acce2b303b1246292ef37e6b31a9c99",
    "period_15.txt": "316d66b05b88e140e58f0c86e044e6c19727f540",
    "period_16.txt": "3d7662088982a4ac428902e28ce7123ab51c20ab",
}
COMMAND = [
    "main.py",
    "--dataset",
    "DIGINETICA",
    "--save_dir",
    "ADER",
    "--num_epochs",
    "100",
    "--stop",
    "5",
    "--batch_size",
    "256",
    "--test_batch",
    "64",
    "--exemplar_size",
    "30000",
    "--lambda_",
    "0.8",
    "--random_seed",
    "0",
    "--hidden_units",
    "150",
    "--num_blocks",
    "2",
    "--num_heads",
    "1",
    "--maxlen",
    "50",
    "--dropout_rate",
    "0.3",
    "--lr",
    "0.0005",
]


def build_command(method):
    command = COMMAND.copy()
    if method == "dropout":
        command[command.index("--save_dir") + 1] = "Dropout"
        command.extend(["--dropout", "True"])
    elif method != "ader":
        raise ValueError("Unknown method: " + method)
    return command


def mean_metrics(windows):
    return {
        key: sum(w["test_metrics"][key] for w in windows) / len(windows)
        for key in ("MRR@20", "Recall@20", "MRR@10", "Recall@10")
    }


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def audit_data(data, windows=1):
    periods, hashes = [], {}
    for period in range(windows + 1):
        name = f"period_{period}.txt"
        expected = DATA_HASHES[name]
        body = (data / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if blob != expected:
            raise ValueError("Data differs from pinned upstream artifact: " + name)
        hashes[name] = {"git_blob": blob, "sha256": hashlib.sha256(body).hexdigest()}
        sessions = defaultdict(list)
        for row in body.decode("utf-8").splitlines():
            session, item = map(int, row.split(" "))
            if session < 0 or not 1 <= item <= 43136:
                raise ValueError("Invalid session/item ID in " + name)
            sessions[session].append(item)
        periods.append(list(sessions.values()))
    counts, seen = [], set()
    for train, test in pairwise(periods):
        seen.update(item for session in train for item in session)
        counts.append(audit_window(train, test, seen))
    return {"files": hashes, "counts": counts[0] if windows == 1 else counts}


def audit_window(train, test, seen):
    if not seen or seen != set(range(1, max(seen) + 1)):
        raise ValueError("Training item IDs must cover 1..max_item")
    filtered = [[item for item in session if item in seen] for session in test]
    filtered = [session for session in filtered if len(session) >= 2]
    stats = {
        "train_actions": sum(map(len, train)),
        "train_sessions": len(train),
        "train_singletons": sum(len(s) == 1 for s in train),
        "train_prefixes_before_validation": sum(max(0, len(s) - 1) for s in train),
        "upstream_sampler_rows_before_validation": sum(max(1, len(s) - 1) for s in train),
        "active_items": len(seen),
        "test_raw_actions": sum(map(len, test)),
        "test_raw_sessions": len(test),
        "test_filtered_actions": sum(map(len, filtered)),
        "test_filtered_sessions": len(filtered),
        "test_filtered_prefixes": sum(len(s) - 1 for s in filtered),
    }
    return stats


def instrument_main(original):
    edits = [
        ("import time\n", "import time\nimport json\n"),
        (
            "best_performance, performance = 0, 0",
            "best_performance, performance = -1, 0\n        stop_counter = 0",
        ),
        (
            "        max_item = dataloader.max_item()",
            """        max_item = dataloader.max_item()
        profile_dir = 'profile-period%d' % period
        os.mkdir(profile_dir)
        with open(profile_dir + '/split.json', 'x') as profile:
            json.dump({'training_prefixes': train_subseq, 'validation_prefixes': valid_subseq,
                       'candidate_ids': list(range(1, max_item + 1))}, profile)""",
        ),
        (
            "for epoch in range(1, args.num_epochs + 1):",
            "for epoch in range(1, args.num_epochs + 1):\n                epoch_started = time.monotonic()",
        ),
        (
            "                # early stop",
            """                with open(profile_dir + '/epochs.jsonl', 'a') as profile:
                    profile.write(json.dumps({'epoch': epoch, 'seconds': time.monotonic() - epoch_started,
                                              'validation_recall20': performance}) + '\\n')
                # early stop""",
        ),
        (
            "            # Select exemplars",
            "            herding_started = time.monotonic()\n            # Select exemplars",
        ),
        (
            "            # Save current item number for next cycle",
            """            with open(profile_dir + '/herding.json', 'x') as profile:
                json.dump({'seconds': 0.0 if args.dropout else time.monotonic() - herding_started, 'actual_replay_size': 0 if args.dropout else saved_num,
                           'active_items': max_item}, profile)
            # Save current item number for next cycle""",
        ),
        (
            "        with tf.Session(config=config) as sess:",
            """        with tf.Session(config=config) as sess:
            devices = [{'name': d.name, 'type': d.device_type} for d in sess.list_devices()]
            if not any(d['type'] == 'GPU' for d in devices):
                raise RuntimeError('Real-window GPU profile requires a visible GPU')
            with open(profile_dir + '/runtime.json', 'x') as profile:
                json.dump({'devices': devices, 'tensorflow': tf.__version__, 'numpy': np.__version__,
                           'build': tf.sysconfig.get_build_info(), 'args': vars(args)}, profile)""",
        ),
    ]
    edits.extend(
        [
            (
                "            exemplar_sampler = Sampler([], args.maxlen, exemplar_batch)",
                """            if exemplar_batch < 1:
                raise RuntimeError('Replay batch is empty; cannot verify distillation')
            exemplar_sampler = Sampler([], args.maxlen, exemplar_batch)""",
            ),
            (
                "            # Train\n",
                """            step_before_training = int(sess.run(model.global_step))
            distillation_updates = 0
            with open(profile_dir + '/continuation.json', 'x') as profile:
                json.dump({'old_items': item_num_prev, 'active_items': max_item,
                           'training_rows': train_sampler.data_size(), 'batches_per_epoch': batch_num,
                           'input_replay_size': exemplar_size if period > 1 and not args.dropout else 0,
                           'replay_batch_size': exemplar_batch if period > 1 and not args.dropout else 0,
                           'lambda': lambda_ if period > 1 and not args.dropout else 0.0,
                           'restored_global_step': step_before_training}, profile)
            # Train
""",
            ),
            (
                (
                    "                                                      model.exemplar_logits: logits,\n"
                    "                                                      model.dropout_rate: args.dropout_rate,\n"
                    "                                                      model.lr: args.lr})"
                ),
                (
                    "                                                      model.exemplar_logits: logits,\n"
                    "                                                      model.dropout_rate: args.dropout_rate,\n"
                    "                                                      model.lr: args.lr})\n"
                    "                            distillation_updates += 1"
                ),
            ),
            (
                "            # Test performance",
                """            with open(profile_dir + '/updates.json', 'x') as profile:
                json.dump({'global_step_after_training': int(sess.run(model.global_step)),
                           'distillation_updates': distillation_updates}, profile)
            # Test performance""",
            ),
            (
                "            test_evaluator = Evaluator(test_sess, False, args.maxlen, args.test_batch,",
                """            with open(profile_dir + '/selection.json', 'x') as profile:
                json.dump({'selected_epoch': best_epoch,
                           'selected_global_step': int(sess.run(model.global_step))}, profile)
            test_evaluator = Evaluator(test_sess, False, args.maxlen, args.test_batch,""",
            ),
        ]
    )
    edits.append(
        (
            "            MRR_20.append(test_evaluator.results()[0])",
            """            with open(profile_dir + '/metrics.json', 'x') as profile:
                json.dump(dict(zip(('MRR@20', 'Recall@20', 'MRR@10', 'Recall@10'),
                                   map(float, test_evaluator.results()))), profile, allow_nan=False)
            MRR_20.append(test_evaluator.results()[0])""",
        )
    )
    for old, new in edits:
        if original.count(old) != 1:
            raise ValueError("Unexpected upstream instrumentation target: " + old)
        original = original.replace(old, new)
    return original


def gpu_sample(text, pid):
    """Return this training process's memory per GPU; unavailable measurements stay null."""
    result = []
    for row in csv.reader(text.splitlines()):
        if len(row) == 3 and row[0].strip() == str(pid):
            memory = row[2].strip()
            result.append(
                {
                    "gpu_uuid": row[1].strip(),
                    "used_mib": float(memory) if memory.replace(".", "", 1).isdigit() else None,
                }
            )
    return result


def run_training(work, output, command=None):
    started = time.monotonic()
    peak = None
    monitor_errors = 0
    environment = dict(os.environ, TF_DETERMINISTIC_OPS="0", PYTHONUNBUFFERED="1")
    with (
        (output / "console.log").open("x", encoding="utf-8") as console,
        subprocess.Popen(
            [sys.executable, "-u"] + (COMMAND if command is None else command),
            cwd=work,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
        ) as child,
    ):
        print(f"Training PID {child.pid}; follow {output / 'console.log'}", flush=True)
        try:
            with (output / "gpu-memory.jsonl").open("x", encoding="utf-8") as samples:
                while child.poll() is None:
                    try:
                        sample = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=5,
                        )
                        values = gpu_sample(sample.stdout, child.pid)
                        numeric = [v["used_mib"] for v in values if v["used_mib"] is not None]
                        if numeric:
                            peak = max(peak or 0, sum(numeric))
                        samples.write(
                            json.dumps({"seconds": time.monotonic() - started, "devices": values})
                            + "\n"
                        )
                        samples.flush()
                    except (OSError, subprocess.SubprocessError):
                        monitor_errors += 1
                    time.sleep(2)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
            raise
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "returncode": child.returncode,
        "seconds": time.monotonic() - started,
        "peak_child_host_rss_mib": usage.ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "sampled_peak_training_gpu_mib": peak,
        "gpu_monitor_errors": monitor_errors,
        "gpu_sampling_seconds": 2,
    }


def collect_results(results, windows, method="ader"):
    log = (results / "Training_logs.txt").read_text(encoding="utf-8")
    matches = re.findall(
        r"epoch:(\d+), test \(MRR@20: ([\d.]+), RECALL@20: ([\d.]+), MRR@10: ([\d.]+), RECALL@10: ([\d.]+)\)",
        log,
    )
    periods = re.findall(r"^Period (\d+):$", log, re.MULTILINE)
    if (
        len(matches) != windows
        or periods != list(map(str, range(1, windows + 1)))
        or not log.endswith("Done.")
    ):
        raise RuntimeError("Incomplete or unexpected training log")
    reports = []
    for period, match in enumerate(matches, 1):
        directory = results / f"profile-period{period}"

        def read(name, directory=directory):
            return json.loads((directory / f"{name}.json").read_text())

        continuation, updates, selection = read("continuation"), read("updates"), read("selection")
        epochs = [json.loads(row) for row in (directory / "epochs.jsonl").read_text().splitlines()]
        if not epochs or [e["epoch"] for e in epochs] != list(range(1, len(epochs) + 1)):
            raise RuntimeError("Missing epoch history")
        if any(not math.isfinite(e["validation_recall20"]) for e in epochs):
            raise RuntimeError("Non-finite validation score")
        selected = int(match[0])
        batch_count = continuation["batches_per_epoch"]
        steps = batch_count * len(epochs)
        start_step = continuation["restored_global_step"]
        if batch_count < 1 or updates["global_step_after_training"] != start_step + steps:
            raise RuntimeError("Optimizer update count differs from completed epochs")
        if selection != {
            "selected_epoch": selected,
            "selected_global_step": start_step + batch_count * selected,
        }:
            raise RuntimeError("Selected checkpoint step differs from selected epoch")
        if not 1 <= selected <= len(epochs) or epochs[selected - 1]["validation_recall20"] != max(
            e["validation_recall20"] for e in epochs
        ):
            raise RuntimeError("Selected epoch is not a validation maximum")
        if period > 1 and start_step != reports[-1]["selection"]["selected_global_step"]:
            raise RuntimeError("Continuation did not restore the selected checkpoint step")
        if method == "ader" and period > 1:
            previous = reports[-1]
            expected_lambda = 0.8 * math.sqrt(
                previous["replay"]["active_items"]
                / continuation["active_items"]
                * previous["replay"]["actual_replay_size"]
                / continuation["training_rows"]
            )
            if (
                start_step != previous["selection"]["selected_global_step"]
                or continuation["input_replay_size"] != previous["replay"]["actual_replay_size"]
                or continuation["old_items"] != previous["replay"]["active_items"]
                or continuation["replay_batch_size"]
                != continuation["input_replay_size"] // batch_count
                or continuation["replay_batch_size"] < 1
                or not math.isclose(continuation["lambda"], expected_lambda, rel_tol=1e-12)
                or updates["distillation_updates"] != steps
            ):
                raise RuntimeError(
                    "Continuation replay, lambda or optimizer evidence is inconsistent"
                )
        elif (
            (period == 1 and start_step != 0)
            or updates["distillation_updates"] != 0
            or any(
                continuation[k] != 0 for k in ("input_replay_size", "replay_batch_size", "lambda")
            )
            or (method == "dropout" and read("herding")["actual_replay_size"] != 0)
        ):
            raise RuntimeError("Initial window and Dropout must not use replay/distillation")
        rounded = dict(zip(("MRR@20", "Recall@20", "MRR@10", "Recall@10"), map(float, match[1:])))
        metrics = read("metrics")
        if metrics.keys() != rounded.keys() or any(
            abs(metrics[k] - rounded[k]) > 0.000051 for k in rounded
        ):
            raise RuntimeError("Saved metrics disagree with upstream log")
        if not all(0 <= value <= 1 for value in metrics.values()):
            raise RuntimeError("Invalid test metrics")
        split = read("split")
        if split["candidate_ids"] != list(range(1, continuation["active_items"] + 1)):
            raise RuntimeError("Candidate IDs differ from active catalogue")
        validation_hash = hashlib.sha256(
            json.dumps(split["validation_prefixes"]).encode()
        ).hexdigest()
        reports.append(
            {
                "train_period": period - 1,
                "test_period": period,
                "selected_epoch": selected,
                "validation_sha256": validation_hash,
                "test_metrics": metrics,
                "epochs": epochs,
                "replay": read("herding"),
                "continuation": continuation,
                "updates": updates,
                "selection": selection,
            }
        )
    return reports


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; source preparation uses assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--windows", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--method", choices=("ader", "dropout"), default="ader")
    args = parser.parse_args()
    command = build_command(args.method)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Use a new output directory: " + str(args.output))
    source, data, output = (
        args.source.resolve(strict=True),
        args.data.resolve(strict=True),
        args.output.resolve(),
    )
    verify_source(source)
    audit = audit_data(data, args.windows)
    output.mkdir(parents=True, exist_ok=False)
    work = prepare_main(source, output, "gpu")
    data_copy = work / "data" / "DIGINETICA"
    data_copy.mkdir(parents=True)
    for name in audit["files"]:
        shutil.copyfile(data / name, data_copy / name)
    if audit_data(data_copy, args.windows) != audit:
        raise ValueError("Data changed during preparation")
    main_path = work / "main.py"
    patched = instrument_main(main_path.read_text(encoding="utf-8"))
    compile(patched, "main.py", "exec")
    main_path.write_text(patched, encoding="utf-8")
    original = (source / "main.py").read_text(encoding="utf-8")
    (output / "profile-main.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="a/main.py",
                tofile="b/main.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "purpose": "Pinned author-code reproduction with recorded compatibility changes",
        "method": args.method,
        "dataset": "DIGINETICA",
        "windows": args.windows,
        "train_periods": list(range(args.windows)),
        "test_periods": list(range(1, args.windows + 1)),
        "source_commit": COMMIT,
        "source_hashes": SOURCE_HASHES,
        "input_audit": audit,
        "command": command,
        "deterministic_gpu_ops_enforced": False,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
        "script_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("profile_ader_window.py", "verify_ader.py")
        },
        "limits": [
            "ADER builds replay after each window; Dropout has no replay or distillation.",
            "Upstream validation overlap/reuse and known-item test filtering retained.",
            "GPU memory is sampled per training PID and may miss short peaks or be unavailable on MIG.",
            "Host RSS is the maximum across child processes, not total Slurm job memory.",
            "Upstream checkpoints omit replay/sampler/RNG state; this pilot is not resumable.",
            "GPU run-to-run reproducibility is not established.",
        ],
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps(audit["counts"], indent=2), flush=True)
    if args.prepare_only:
        print("Prepared only; no training started: " + str(output))
        return
    report = {"status": "failed"}
    try:
        report.update(run_training(work, output, command))
        if report["returncode"]:
            raise RuntimeError("Training failed; inspect " + str(output / "console.log"))
        results = (
            work
            / "results"
            / ("DIGINETICA-ADER" if args.method == "ader" else "DIGINETICA-Dropout")
        )
        windows = collect_results(results, args.windows, args.method)
        report.update(
            status="passed",
            method=args.method,
            metrics_precision="Full-precision upstream evaluator results",
            mean_test_metrics=mean_metrics(windows),
        )
        if args.windows == 1:
            report.update(windows[0])
        else:
            report["windows"] = windows
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save_json(output / "profile.json", report)
    print(f"Completed {args.windows} real window(s). Report: {output / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
