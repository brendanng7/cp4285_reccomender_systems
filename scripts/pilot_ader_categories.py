"""Run a two-period, training-only ADER category pilot on a prepared period file."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verify_ader import COMMIT, SOURCE_HASHES, verify_source

MAXLEN = 50
PILOT_ROWS = 50_000
BATCH_SIZE = 256
REPLAY_LIMIT = 1_000
SEED = 0


def pad_history(history, mapping):
    values = [mapping[item] for item in history[-MAXLEN:]]
    result = np.zeros(MAXLEN, dtype=np.int32)
    if values:
        result[-len(values) :] = values
    return result


def select_examples(con, period_file, events_file, period, mapping, count):
    rows = con.execute(
        """
        with chosen as (
            select row_number() over (order by timestamp,user_id,parent_asin) qid,
                   user_id,timestamp,category_id,history_start,history_end
            from read_parquet(?) where period=?
            order by md5(user_id || '|' || cast(timestamp as varchar) || '|' || parent_asin),
                     timestamp,user_id,parent_asin limit ?
        )
        select q.qid,q.timestamp,q.category_id,
               list(e.category_id order by e.position) history
        from chosen q join read_parquet(?) e
          on e.user_id=q.user_id and e.position between q.history_start and q.history_end
        group by q.qid,q.timestamp,q.category_id order by q.qid
        """,
        [str(period_file), period, count, str(events_file)],
    ).fetchall()
    if len(rows) != count or any(not row[3] for row in rows):
        raise ValueError(f"Period {period} did not yield {count} complete histories")
    inputs = np.stack([pad_history(row[3], mapping) for row in rows])
    targets = np.asarray([mapping[row[2]] for row in rows], dtype=np.int32)
    if np.any(targets <= 0) or np.any(targets > len(mapping)):
        raise ValueError("A target is outside the active category prefix")
    return inputs, targets


def train_order(n, seed):
    return np.random.RandomState(seed).permutation(n)


def assert_self_checks():
    old = [4, 8, 12]
    new = [2, 9]
    mapping = {category: i + 1 for i, category in enumerate(old + new)}
    assert [mapping[x] for x in old] == [1, 2, 3]
    assert np.array_equal(pad_history([4, 8, 12], mapping)[-3:], [1, 2, 3])
    assert np.array_equal(pad_history([], mapping), np.zeros(MAXLEN, dtype=np.int32))
    assert np.array_equal(train_order(8, SEED), train_order(8, SEED))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(args):
    import duckdb
    import tensorflow.compat.v1 as tf

    assert_self_checks()
    verify_source(args.source)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    period_file = args.periods / "train_periods.parquet"
    report_path = args.periods / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.periods / "manifest.json").read_text(encoding="utf-8"))
    period_info = report["artifact"]
    if period_file.stat().st_size != period_info["bytes"]:
        raise ValueError("Prepared period artifact size differs from its report")
    digest = sha256_file(period_file)
    if digest != period_info["sha256"]:
        raise ValueError("Prepared period artifact checksum differs from its report")
    event_info = manifest["input_artifacts"]["train_events.parquet"]
    if args.events.stat().st_size != event_info["bytes"]:
        raise ValueError("Training event artifact size differs from its manifest")
    if sha256_file(args.events) != event_info["sha256"]:
        raise ValueError("Training event artifact checksum differs from its manifest")
    period_meta = {item["period"]: item for item in report["periods"]}
    if not {1, 2} <= period_meta.keys():
        raise ValueError("The prepared artifact must contain periods 1 and 2")
    con = duckdb.connect()
    active = {}
    old_ids = []
    for period in (1, 2):
        cutoff = period_meta[period]["last_target_ms"]
        ids = [row[0] for row in con.execute(
            """
            select category_id from (
                select distinct category_id from read_parquet(?) where period<=?
                union
                select distinct category_id from read_parquet(?) where timestamp<=?
            ) order by category_id
            """,
            [str(period_file), period, str(args.events), cutoff],
        ).fetchall()]
        if not set(old_ids) <= set(ids):
            raise ValueError("Active category sets must be cumulative")
        ordered = old_ids + [item for item in ids if item not in set(old_ids)]
        active[period] = ordered
        old_ids = ordered
        if len(ordered) != period_meta[period]["cumulative_seen_categories"]:
            raise ValueError("Active category count differs from the prepared period report")
    if not active[1] or len(active[2]) <= len(active[1]):
        raise ValueError("Expected new categories to arrive in period 2")

    examples = {}
    for period in (1, 2):
        examples[period] = select_examples(
            con,
            period_file,
            args.events,
            period,
            {category: i + 1 for i, category in enumerate(active[period])},
            PILOT_ROWS,
        )
    con.close()

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("This pilot requires the allocated GPU")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    sys.path.insert(0, str(args.source.resolve()))
    from ADER import Ader

    model_args = SimpleNamespace(
        maxlen=MAXLEN,
        hidden_units=150,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.3,
        random_seed=SEED,
        l2_emb=0.0,
        disable_distillation=False,
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    def new_model(replay_weight=None):
        tf.reset_default_graph()
        tf.set_random_seed(SEED)
        with tf.device("/gpu:0"):
            model = Ader(len(active[2]), model_args)
            if replay_weight is None:
                model.set_vanilla_loss()
            else:
                model.update_loss(replay_weight)
        if "/device:GPU:0" not in model.logits.op.device:
            raise RuntimeError("Category scoring graph is not assigned to the allocated GPU")
        return model, tf.train.Saver(max_to_keep=1)

    start = time.monotonic()
    p1_inputs, p1_targets = examples[1]
    order1 = train_order(len(p1_targets), SEED)
    model, saver = new_model()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for start_ix in range(0, len(order1), BATCH_SIZE):
            ix = order1[start_ix : start_ix + BATCH_SIZE]
            sess.run(
                model.train_op,
                {
                    model.input_seq: p1_inputs[ix], model.pos: p1_targets[ix],
                    model.max_item: len(active[1]), model.is_training: True,
                    model.dropout_rate: model_args.dropout_rate, model.lr: 0.0005,
                },
            )
        p1_steps = int(sess.run(model.global_step))
        if p1_steps != math.ceil(PILOT_ROWS / BATCH_SIZE):
            raise RuntimeError("Period 1 did not train every selected example once")

        quotas = np.random.RandomState(SEED).multinomial(
            REPLAY_LIMIT, np.bincount(p1_targets, minlength=len(active[1]) + 1)[1:] / len(p1_targets)
        )
        memory_inputs, memory_logits = [], []
        for category in range(1, len(active[1]) + 1):
            candidates = np.flatnonzero(p1_targets == category)
            quota = min(int(quotas[category - 1]), len(candidates))
            if not quota:
                continue
            chunks = []
            for ix in np.array_split(candidates, math.ceil(len(candidates) / 512)):
                if len(ix):
                    rep, logits = sess.run(
                        [model.rep, model.logits],
                        {
                            model.input_seq: p1_inputs[ix], model.max_item: len(active[1]),
                            model.is_training: False, model.dropout_rate: model_args.dropout_rate,
                        },
                    )
                    chunks.append((rep, logits))
            reps = np.concatenate([x[0] for x in chunks])
            logits = np.concatenate([x[1] for x in chunks])
            norms = np.linalg.norm(reps, axis=1)
            if np.any(norms == 0) or not np.isfinite(reps).all():
                raise ValueError("Invalid representation in replay herding")
            normalized = reps / norms[:, None]
            mean = normalized.mean(axis=0)
            residual = mean.copy()
            selected = []
            for _ in range(math.ceil(1.1 * quota)):
                index = int(np.argmax(residual @ normalized.T))
                residual += mean - normalized[index]
                if index not in selected:
                    selected.append(index)
                    if len(selected) == quota:
                        break
            chosen = candidates[np.asarray(selected)]
            memory_inputs.extend(p1_inputs[chosen])
            memory_logits.extend(logits[np.asarray(selected)])
        memory_inputs = np.asarray(memory_inputs, dtype=np.int32)
        memory_logits = np.asarray(memory_logits, dtype=np.float32)
        if not 0 < len(memory_inputs) <= REPLAY_LIMIT or memory_logits.shape != (len(memory_inputs), len(active[1])):
            raise RuntimeError("Replay memory did not contain aligned teacher logits")
        p1_checkpoint = saver.save(sess, str(output / "period1.ckpt"))

    p2_inputs, p2_targets = examples[2]
    order2 = train_order(len(p2_targets), SEED + 1)
    batch_count = math.ceil(len(order2) / BATCH_SIZE)
    replay_chunks = np.array_split(np.arange(len(memory_inputs)), batch_count)
    replay_weight = 0.8 * math.sqrt(
        (len(active[1]) / len(active[2])) * (len(memory_inputs) / PILOT_ROWS)
    )
    methods = {}
    for method in ("dropout", "ader"):
        model, saver = new_model(replay_weight if method == "ader" else None)
        replay_used = 0
        with tf.Session(config=config) as sess:
            saver.restore(sess, p1_checkpoint)
            restored_step = int(sess.run(model.global_step))
            if restored_step != p1_steps:
                raise RuntimeError("Period 2 did not restore the shared period 1 optimizer state")
            for batch_ix, start_ix in enumerate(range(0, len(order2), BATCH_SIZE)):
                ix = order2[start_ix : start_ix + BATCH_SIZE]
                feed = {
                    model.input_seq: p2_inputs[ix], model.pos: p2_targets[ix],
                    model.max_item: len(active[2]), model.is_training: True,
                    model.dropout_rate: model_args.dropout_rate, model.lr: 0.0005,
                }
                if method == "ader":
                    replay_ix = replay_chunks[batch_ix]
                    replay_used += len(replay_ix)
                    feed[model.input_seq] = np.concatenate(
                        [p2_inputs[ix], memory_inputs[replay_ix]], axis=0
                    )
                    feed[model.exemplar_logits] = memory_logits[replay_ix]
                sess.run(model.train_op, feed)
            final_step = int(sess.run(model.global_step))
            expected_step = p1_steps + batch_count
            if final_step != expected_step:
                raise RuntimeError(f"Unexpected {method} optimizer step count")
            if method == "ader" and replay_used != len(memory_inputs):
                raise RuntimeError("Not every selected replay example was consumed")
            methods[method] = {
                "restored_global_step": restored_step,
                "final_global_step": final_step,
                "period2_updates": batch_count,
                "replay_examples_selected": len(memory_inputs) if method == "ader" else 0,
                "replay_examples_consumed": replay_used,
                "lambda": replay_weight if method == "ader" else 0.0,
            }
    gpu_file = os.environ.get("ADER_GPU_MONITOR_FILE")
    gpu_samples = []
    if gpu_file and Path(gpu_file).exists():
        for line in Path(gpu_file).read_text(encoding="utf-8").splitlines():
            try:
                gpu_samples.append(int(line.strip()))
            except ValueError:
                continue
    result = {
        "status": "passed",
        "source_commit": COMMIT,
        "source_hashes": SOURCE_HASHES,
        "period_file": str(period_file),
        "period_file_sha256": digest,
        "validation_or_test_read": False,
        "data": "prepared training rows only; deterministic hash sample of 50000 queries from each of periods 1 and 2",
        "active_categories": {str(k): len(v) for k, v in active.items()},
        "training_targets": {str(k): len(examples[k][1]) for k in examples},
        "model": {"maxlen": MAXLEN, "hidden_units": 150, "blocks": 2, "heads": 1, "dropout": 0.3},
        "batch_size": BATCH_SIZE,
        "epochs_per_period": 1,
        "period1_optimizer_steps": p1_steps,
        "replay_limit": REPLAY_LIMIT,
        "actual_replay_size": len(memory_inputs),
        "replay_teacher_shape": list(memory_logits.shape),
        "methods": methods,
        "seconds": time.monotonic() - start,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "sampled_peak_gpu_memory_mib": max(gpu_samples) if gpu_samples else None,
        "gpu_memory_samples": len(gpu_samples),
        "limits": [
            "Training-only pilot; no validation selection or test scoring.",
            "One epoch and 50000 targets per period; not a full-data result.",
            "The period-1 checkpoint is shared, then period 2 is branched by method.",
            "GPU dropout execution is stochastic; no run-to-run parity claim.",
        ],
    }
    save_json(output / "pilot.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
