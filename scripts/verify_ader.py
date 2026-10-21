"""Verify pinned upstream ADER on synthetic data; Python 3.7+ and TensorFlow required.

This is a reference-code check, not a cp4285 model or a research-data runner.
Run without -O: assertions are the verification gates.
"""

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

COMMIT = "348fc2128bceb6bb3f3d8dd25ec8bbbae831af94"
SOURCE_HASHES = {
    "ADER.py": "afd79020442a4aa8b0ad803094ef183bb859f77b",
    "EWC.py": "8df48f32c505e7b21e5cbabb82cdd908fb21e0e2",
    "main.py": "9e645d67ea1c83d53fbc77d4a0eb81db42df5e2a",
    "modules.py": "025edaaca15de8ce64601112b1f792270835e47b",
    "util.py": "c1111dcf44d17de8e8a44c2d5f9d888e6f1705e6",
    "data/preprocessing.py": "ada28dabf28a088df5ef1cc3ea2870d2262e0e8f",
    "data/util.py": "66ba9013b47f7455360e54675f0abee9cd3c1cd9",
    "LICENSE": "caa2ea77cc1ad42e950b17939602b8580a8f3611",
}


def verify_source(source):
    for name, expected in SOURCE_HASHES.items():
        body = (source / name).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if digest != expected:
            raise ValueError("Source differs from pinned commit: " + name)


def prepare_main(source, output, device):
    """Copy upstream code and record compatibility/device edits without running it."""
    work = output / "upstream-main"
    work.mkdir()
    for name in ("main.py", "ADER.py", "EWC.py", "modules.py", "util.py", "LICENSE"):
        shutil.copyfile(source / name, work / name)
    original = (work / "main.py").read_text(encoding="utf-8")
    old = "np.array(exemplar_data_logits)[:, 0].tolist()"
    assert original.count(old) == 1
    patched = original.replace(old, "[record[0] for record in exemplar_data_logits]")
    device_assignment = "os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)"
    assert patched.count(device_assignment) == 1
    patched = patched.replace(
        device_assignment,
        "# Preserve the caller's CUDA_VISIBLE_DEVICES (including Slurm assignments).",
    )
    assert device_assignment not in patched
    if device == "cpu":
        assert patched.count("config = tf.ConfigProto()") == 1
        patched = patched.replace(
            "config = tf.ConfigProto()", "config = tf.ConfigProto(device_count={'GPU': 0})"
        )
    else:
        # TF 2.15 cannot enforce deterministic GPU scatter updates on legacy ref variables.
        deterministic = "os.environ['TF_DETERMINISTIC_OPS'] = '1'"
        assert patched.count(deterministic) == 1
        patched = patched.replace(deterministic, "os.environ['TF_DETERMINISTIC_OPS'] = '0'")
        print(
            "NOTE: GPU entry-point smoke disables deterministic-op enforcement; "
            "this does not verify run-to-run reproducibility.",
            flush=True,
        )
    (work / "main.py").write_text(patched, encoding="utf-8")
    (output / "main-numpy-compat.patch").write_text(
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
    return work


def check_main(source, output, device):
    """Run two synthetic windows in a disposable copy of the authors' entry point."""
    work = prepare_main(source, output, device)
    data = work / "data" / "DIGINETICA"
    data.mkdir(parents=True)
    for period in range(3):
        with (data / f"period_{period}.txt").open("x", encoding="utf-8") as handle:
            for session in range(12):
                for offset in range(4):
                    handle.write(
                        f"{session + 100 * period} {(session + offset) % (4 + period) + 1}\n"
                    )
    command = [
        sys.executable,
        "main.py",
        "--num_epochs",
        "2",
        "--batch_size",
        "64",
        "--test_batch",
        "64",
        "--exemplar_size",
        "8",
    ]
    environment = dict(os.environ, TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
    if device == "gpu":
        environment["TF_DETERMINISTIC_OPS"] = "0"
    with (work / "console.log").open("x", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=work,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    results = work / "results" / "DIGINETICA-ADER"
    log = (results / "Training_logs.txt").read_text(encoding="utf-8")
    assert "Period 1:" in log and "Period 2:" in log and log.endswith("Done.")
    assert log.count("test (MRR@20:") == 2
    assert len(list((results / "model").glob("period*/epoch=1.ckpt.index"))) == 2
    return {
        "command": command[1:],
        "windows": 2,
        "epochs_per_window": 2,
        "patch": "main-numpy-compat.patch",
        "console": "upstream-main/console.log",
        "deterministic_ops_enforced": device == "cpu",
        "limits": "Tiny synthetic inputs; no full-data, baseline, or EWC coverage. "
        "GPU smoke does not guarantee run-to-run reproducibility.",
    }


def check_scoring_device(partition_graphs, node_name, device):
    placed = [
        node.device for graph in partition_graphs for node in graph.node if node.name == node_name
    ]
    if not placed or not all(f"/device:{device.upper()}:" in name for name in placed):
        raise RuntimeError(f"Scoring operation {node_name} did not run on {device}: {placed}")
    return placed


def run_checks(source, output, device):
    import numpy as np
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    sys.path.insert(0, str(source))
    from ADER import Ader
    from util import DataLoader, Evaluator, ExemplarGenerator, Sampler

    # Keep the paper's width, depth and context length; only the catalogue/data are tiny.
    args = SimpleNamespace(
        maxlen=50,
        hidden_units=150,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.3,
        random_seed=0,
        l2_emb=0.0,
        disable_distillation=False,
    )
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    checks = []

    def passed(name):
        checks.append(name)
        print("PASS: " + name, flush=True)

    def log_softmax(logits):
        shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    data = output / "synthetic"
    data.mkdir()
    (data / "period_0.txt").write_text("1 1\n1 2\n1 3\n2 4\n2 3\n", encoding="utf-8")
    (data / "period_1.txt").write_text("3 1\n3 5\n3 2\n4 5\n", encoding="utf-8")
    loader = DataLoader("DIGINETICA")
    loader.path = str(data)
    sessions, _ = loader.train_loader(0)
    assert sessions == [[1, 2, 3], [4, 3]] and loader.max_item() == 4
    tested, _ = loader.evaluate_loader(1)
    assert tested == [[1, 2]] and loader.item_set == {1, 2, 3, 4}
    sampler = Sampler(sessions, args.maxlen, 8)
    assert sampler.prepared_data == [[1, 2, 3], [1, 2], [4, 3]]
    padded, target = sampler.label_generator([1, 2, 3])
    assert target == 3 and padded[-2:].tolist() == [1, 2] and not padded[:-2].any()
    long, target = sampler.label_generator(list(range(1, 61)))
    assert target == 60 and long.tolist() == list(range(10, 60))
    passed("upstream period loading, unknown-item filtering, prefix expansion and truncation")

    split_sampler = Sampler([list(range(1, 13))], args.maxlen, 8)
    np.random.seed(0)
    valid, train = split_sampler.split_data(0.1, return_train=True)
    assert any(v[-1] in t[:-1] for v in valid for t in train)
    passed("known upstream validation overlap reproduced (not an isolation pass)")

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.ranks = [0, 9, 10, 19, 20]
    np.testing.assert_allclose(
        evaluator.results(),
        [(1 + 1 / 10 + 1 / 11 + 1 / 20) / 5, 0.8, 1.1 / 5, 0.4],
    )
    passed("upstream Recall/MRR boundaries at ranks 1, 10, 11, 20 and 21")

    config = tf.ConfigProto(
        # Legacy int32 variables (global_step) require CPU, even in a GPU model.
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    if device == "cpu":
        config.device_count["GPU"] = 0
    with tf.Session(config=config) as session:
        devices = [{"name": d.name, "type": d.device_type} for d in session.list_devices()]
    if device == "gpu" and not any(d["type"] == "GPU" for d in devices):
        raise RuntimeError("GPU requested but TensorFlow found no GPU; refusing CPU fallback")

    with tf.device("/cpu:0" if device == "cpu" else "/gpu:0"):
        model = Ader(6, args)
        model.set_vanilla_loss()
    old = [[1, 2], [2, 3], [3, 4], [4, 1]]
    inputs, targets = zip(*(sampler.label_generator(row) for row in old))
    initial_feed = {
        model.input_seq: inputs,
        model.pos: targets,
        model.max_item: 4,
        model.is_training: False,
        model.dropout_rate: 0.0,
        model.lr: 0.0005,
    }
    initial_saver = tf.train.Saver()
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        placement = tf.RunMetadata()
        initial_logits, initial_loss = session.run(
            [model.logits, model.loss],
            initial_feed,
            options=tf.RunOptions(output_partition_graphs=True),
            run_metadata=placement,
        )
        scoring_devices = check_scoring_device(
            placement.partition_graphs, model.logits.op.name, device
        )
        passed("executed scoring operation is on the requested device")
        ce = -log_softmax(initial_logits)[np.arange(4), np.array(targets) - 1].mean()
        np.testing.assert_allclose(initial_loss, ce, rtol=1e-5, atol=1e-6)
        for _ in range(8):
            session.run(model.train_op, initial_feed)
        trained_loss = float(session.run(model.loss, initial_feed))
        assert trained_loss < initial_loss and session.run(model.global_step) == 8
        passed("full-softmax CE matches NumPy and eight synthetic updates reduce loss")

        before_eval = session.run(tf.global_variables())
        scores_a = session.run(model.logits, initial_feed)
        scores_b = session.run(model.logits, initial_feed)
        np.testing.assert_array_equal(scores_a, scores_b)
        ranks = model.predict(session, inputs, [1, 2, 3, 4])
        assert np.all(np.diff(np.sort(scores_a, axis=1), axis=1) != 0)
        np.testing.assert_array_equal(ranks, np.argsort(np.argsort(-scores_a, axis=1), axis=1))
        reordered = model.predict(session, inputs, [4, 2, 1, 3])
        np.testing.assert_array_equal(reordered, ranks[:, [3, 1, 0, 2]])
        for before, after in zip(before_eval, session.run(tf.global_variables())):
            np.testing.assert_array_equal(before, after)
        passed("evaluation is deterministic, candidate aligned and does not mutate variables")

        memory = ExemplarGenerator(old * 2, 4, False, 4, args.maxlen, 0.3, 4)
        memory_count = memory.herding_selection(session, model)
        records = [record for group in memory.exemplars.values() for record in group]
        assert memory_count == len(records) and 0 < memory_count <= 4
        replay_inputs, _ = zip(*(sampler.label_generator(row) for row, _ in records))
        teachers = np.asarray([logits for _, logits in records], dtype=np.float32)
        assert teachers.shape == (memory_count, 4) and np.isfinite(teachers).all()
        fresh_teacher = session.run(
            model.logits,
            {
                model.input_seq: replay_inputs,
                model.max_item: 4,
                model.is_training: False,
                model.dropout_rate: 0.3,
            },
        )
        np.testing.assert_allclose(teachers, fresh_teacher, rtol=1e-5, atol=1e-6)
        passed("herding selects real examples and stores current dropout-free teacher logits")
        initial_state = session.run(tf.global_variables())
        initial_names = [v.name for v in tf.global_variables()]
        initial_saver.save(session, str(output / "initial.ckpt"))

    current = [[1, 5], [2, 6]]
    current_inputs, current_targets = zip(*(sampler.label_generator(row) for row in current))
    weight = 0.8 * math.sqrt((4 / 6) * (memory_count / len(current)))
    with tf.device("/cpu:0" if device == "cpu" else "/gpu:0"):
        model.update_loss(weight)
        current_ce = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=model.labels[: len(current)],
                logits=model.logits[: len(current)],
            )
        )
        kd_gradient = tf.gradients(model.exemp_loss - current_ce, model.logits)[0]
    assert [v.name for v in tf.global_variables()] == initial_names
    continuation_feed = {
        model.input_seq: tuple(current_inputs) + tuple(replay_inputs),
        model.pos: current_targets,
        model.max_item: 6,
        model.exemplar_logits: teachers,
        model.is_training: False,
        model.dropout_rate: 0.0,
        model.lr: 0.0005,
    }
    saver = tf.train.Saver()
    with tf.Session(config=config) as session:
        saver.restore(session, str(output / "initial.ckpt"))
        for before, restored in zip(initial_state, session.run(tf.global_variables())):
            np.testing.assert_array_equal(before, restored)
        assert any("Adam" in name for name in initial_names)
        passed("next-window graph restores every weight, Adam slot and global step")

        logits, total = session.run([model.logits, model.exemp_loss], continuation_feed)
        ce = -log_softmax(logits[:2])[np.arange(2), np.array(current_targets) - 1].mean()
        teacher_probability = np.exp(log_softmax(teachers))
        kd = -(teacher_probability * log_softmax(logits[2:, :4])).sum(axis=1).mean()
        np.testing.assert_allclose(total, ce + weight * kd, rtol=1e-5, atol=1e-6)
        changed_teacher = teachers.copy()
        changed_teacher[:, 0] += 2
        gradient_feed = dict(continuation_feed)
        gradient_feed[model.exemplar_logits] = changed_teacher
        gradient = session.run(kd_gradient, gradient_feed)
        np.testing.assert_allclose(gradient[:2], 0, atol=1e-7)
        np.testing.assert_array_equal(gradient[2:, 4:], 0)
        assert np.any(np.abs(gradient[2:, :4]) > 1e-7)
        passed("ADER loss matches separate CE/KD means; KD excludes new-item logits")

        state_before = session.run(tf.trainable_variables())
        session.run(model.train_op, continuation_feed)
        state_after = session.run(tf.trainable_variables())
        assert any(not np.array_equal(a, b) for a, b in zip(state_before, state_after))
        assert session.run(model.global_step) == 9
        np.testing.assert_array_equal(teachers, continuation_feed[model.exemplar_logits])
        passed("distillation continuation updates parameters with frozen teacher inputs")

        # Saver does not save Python sampler state or TensorFlow dropout RNG progress.
        saver.save(session, str(output / "continuation.ckpt"))
        session.run(model.train_op, continuation_feed)
        expected_state = session.run(tf.global_variables())
    with tf.Session(config=config) as session:
        saver.restore(session, str(output / "continuation.ckpt"))
        session.run(model.train_op, continuation_feed)
        resumed_state = session.run(tf.global_variables())
        for expected, actual in zip(expected_state, resumed_state):
            np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-6)
        stochastic_feed = dict(continuation_feed)
        stochastic_feed[model.is_training] = True
        stochastic_feed[model.dropout_rate] = 0.3
        stochastic_a = session.run(model.logits, stochastic_feed)
        stochastic_b = session.run(model.logits, stochastic_feed)
        assert not np.array_equal(stochastic_a, stochastic_b)
        session.run(model.train_op, stochastic_feed)
        assert np.isfinite(session.run(model.exemp_loss, continuation_feed))
    passed("restored next optimizer update matches uninterrupted execution with dropout disabled")
    passed("training dropout is active and a dropout-enabled continuation step remains finite")

    return {
        "checks": checks,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": devices,
        "scoring_devices": scoring_devices,
        "model_config": vars(args),
        "initial_loss": float(initial_loss),
        "trained_loss": trained_loss,
        "actual_replay_size": memory_count,
        "lambda": weight,
        "limits": [
            "Synthetic six-item catalogue, not real-data performance or resource profiling.",
            "Not cross-version TensorFlow parity; rerun in the selected cluster environment.",
            "Deterministic optimizer restart checked with dropout disabled only.",
            "Full process/RNG restart is not verified by these model checks.",
        ],
    }


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; verification requires assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    options = parser.parse_args()
    if options.output.exists() or options.output.is_symlink():
        raise FileExistsError("Use a new output directory: " + str(options.output))
    source = options.source.resolve(strict=True)
    verify_source(source)
    options.output.mkdir(parents=True, exist_ok=False)
    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    started = time.monotonic()
    report = {
        "status": "running",
        "source_commit": COMMIT,
        "source_hashes": SOURCE_HASHES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": options.device,
        "data": "synthetic only",
    }
    try:
        report.update(run_checks(source, options.output, options.device))
        report["main_smoke"] = check_main(source, options.output, options.device)
        report["checks"].append("two-window upstream main.py smoke with recorded NumPy fix")
        print("PASS: two-window upstream main.py smoke with recorded NumPy fix", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        with (options.output / "verification.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    print("Report: " + str(options.output / "verification.json"))


if __name__ == "__main__":
    main()
