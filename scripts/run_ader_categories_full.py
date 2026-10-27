"""Train one full-data, eight-period ADER/Dropout category seed; never reads validation/test."""

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
BATCH_SIZE = 256
MEMORY_SIZE = 30_000


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def write_progress(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def category_stages(con, period_file, event_file, report):
    active, previous = {}, []
    metadata = {row["period"]: row for row in report["periods"]}
    for period in range(1, 9):
        cutoff = metadata[period]["last_target_ms"]
        ids = [row[0] for row in con.execute(
            """
            select category_id from (
                select distinct category_id from read_parquet(?) where period<=?
                union
                select distinct category_id from read_parquet(?) where timestamp<=?
            ) order by category_id
            """,
            [str(period_file), period, str(event_file), cutoff],
        ).fetchall()]
        if not ids or not set(previous) <= set(ids):
            raise ValueError("Training category sets must grow cumulatively")
        ordered = previous + [category for category in ids if category not in set(previous)]
        if len(ordered) != metadata[period]["cumulative_seen_categories"]:
            raise ValueError("Category stage count differs from the prepared report")
        active[period] = ordered
        previous = ordered
    return active


def load_period(con, period_file, event_file, period, expected_rows, id_map, output_dir):
    """Build one fixed-width period in bounded fetch batches; histories stay right aligned."""
    output = output_dir / f"period-{period}-inputs.npy"
    targets = np.empty(expected_rows, dtype=np.int32)
    inputs = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.int32, shape=(expected_rows, MAXLEN)
    )
    token_columns = ",".join(f"list_extract(tokens,{i})" for i in range(1, MAXLEN + 1))
    sql = f"""
        with q as (
            select row_number() over(order by timestamp,user_id,parent_asin) qid,
                   user_id,category_id target,label_source,history_start,history_end
            from read_parquet(?) where period=?
        ), h as (
            select q.qid,q.target,q.label_source,list(e.category_id order by e.position) history
            from q join read_parquet(?) e
              on e.user_id=q.user_id and e.position between q.history_start and q.history_end
            group by q.qid,q.target,q.label_source
        ), p as (
            select qid,target,
                   case label_source when 'observed' then 1 when 'predicted' then 2 else 0 end source,
                   list_concat(list_resize([]::integer[],{MAXLEN}-len(history),0),history) tokens
            from h
        )
        select target,source,{token_columns} from p order by qid
    """
    cursor = con.execute(sql, [str(period_file), period, str(event_file)])
    offset = 0
    while rows := cursor.fetchmany(32_768):
        block = np.asarray(rows, dtype=np.int32)
        end = offset + len(block)
        targets[offset:end] = id_map[block[:, 0]]
        inputs[offset:end] = id_map[block[:, 2:]]
        if (
            np.any(block[:, 1] == 0)
            or np.any(inputs[offset:end, -1] == 0)
            or np.any((block[:, 2:] > 0) & (inputs[offset:end] == 0))
        ):
            raise ValueError("Training query has an unknown label or empty history")
        offset = end
    if offset != expected_rows or np.any(targets <= 0):
        raise ValueError(f"Period {period} row count or target mapping is invalid")
    inputs.flush()
    return inputs, targets


def build_model(tf, model_args, item_count, weight):
    tf.reset_default_graph()
    tf.set_random_seed(model_args.random_seed)
    with tf.device("/gpu:0"):
        from ADER import Ader

        model = Ader(item_count, model_args)
        model.set_vanilla_loss()
        vanilla_op = model.train_op
        model.update_loss(weight)
        ader_op = model.train_op
    if "/device:GPU:0" not in model.logits.op.device:
        raise RuntimeError("Category scoring graph is not assigned to the allocated GPU")
    return model, vanilla_op, ader_op, tf.train.Saver(max_to_keep=20)


def feed_for(model, x, y, active_count, model_args):
    return {
        model.input_seq: x,
        model.pos: y,
        model.max_item: active_count,
        model.is_training: True,
        model.dropout_rate: model_args.dropout_rate,
        model.lr: 0.0005,
    }


def train_branch(tf, config, model, saver, operation, checkpoint, output, method,
                 inputs, targets, active_count, model_args, period, seed,
                 replay_inputs=None, replay_logits=None):
    order = np.random.RandomState(seed * 101 + period).permutation(len(targets))
    batches = math.ceil(len(order) / BATCH_SIZE)
    replay_parts = []
    if method == "ader":
        replay_order = np.random.RandomState(seed * 101 + period + 1).permutation(len(replay_inputs))
        replay_parts = np.array_split(replay_order, batches)
    replay_used = 0
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)
        restored_step = int(sess.run(model.global_step))
        for batch_ix, start in enumerate(range(0, len(order), BATCH_SIZE)):
            ix = order[start : start + BATCH_SIZE]
            feed = feed_for(model, inputs[ix], targets[ix], active_count, model_args)
            if method == "ader":
                mem_ix = replay_parts[batch_ix]
                replay_used += len(mem_ix)
                feed[model.input_seq] = np.concatenate(
                    [inputs[ix], replay_inputs[mem_ix]], axis=0
                )
                feed[model.exemplar_logits] = replay_logits[mem_ix]
            sess.run(operation, feed)
        final_step = int(sess.run(model.global_step))
        if final_step - restored_step != batches:
            raise RuntimeError(f"Unexpected {method} optimizer step count in period {period}")
        if method == "ader" and replay_used != len(replay_inputs):
            raise RuntimeError("ADER did not consume every saved replay example")
        saved = saver.save(sess, str(output / f"{method}-period-{period}.ckpt"))
        return saved, {
            "restored_step": restored_step,
            "final_step": final_step,
            "updates": batches,
            "replay_consumed": replay_used,
        }


def select_memory(sess, model, inputs, targets, previous_inputs, previous_targets,
                  active_count, model_args, memory_size, seed):
    candidate_inputs = inputs if not len(previous_inputs) else np.concatenate(
        [inputs, previous_inputs], axis=0
    )
    candidate_targets = targets if not len(previous_targets) else np.concatenate(
        [targets, previous_targets]
    )
    counts = np.bincount(candidate_targets, minlength=active_count + 1)[1:]
    quotas = np.random.RandomState(seed).multinomial(memory_size, counts / counts.sum())
    saved_inputs, saved_targets, saved_logits = [], [], []
    for category, quota in enumerate(quotas, 1):
        indices = np.flatnonzero(candidate_targets == category)
        quota = min(int(quota), len(indices))
        if not quota:
            continue
        reps, logits = [], []
        for start in range(0, len(indices), 512):
            ix = indices[start : start + 512]
            rep, score = sess.run(
                [model.rep, model.logits],
                {
                    model.input_seq: candidate_inputs[ix],
                    model.max_item: active_count,
                    model.is_training: False,
                    model.dropout_rate: model_args.dropout_rate,
                },
            )
            reps.append(rep)
            logits.append(score)
        reps = np.concatenate(reps)
        logits = np.concatenate(logits)
        norms = np.linalg.norm(reps, axis=1)
        if np.any(norms == 0) or not np.isfinite(reps).all() or not np.isfinite(logits).all():
            raise ValueError("Non-finite teacher or representation in replay herding")
        directions = reps / norms[:, None]
        mean = directions.mean(axis=0)
        residual = mean.copy()
        selected, seen = [], set()
        for _ in range(math.ceil(1.1 * quota)):
            index = int(np.argmax(residual @ directions.T))
            residual += mean - directions[index]
            if index not in seen:
                seen.add(index)
                selected.append(index)
                if len(selected) == quota:
                    break
        selected = np.asarray(selected, dtype=np.int64)
        saved_inputs.append(candidate_inputs[indices[selected]])
        saved_targets.append(candidate_targets[indices[selected]])
        saved_logits.append(logits[selected])
    if not saved_inputs:
        raise ValueError("Herding selected no replay examples")
    return (
        np.concatenate(saved_inputs),
        np.concatenate(saved_targets),
        np.concatenate(saved_logits).astype(np.float32, copy=False),
    )


def run(args):
    import duckdb
    import tensorflow.compat.v1 as tf

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Choose a fresh seed output directory")
    verify_source(args.source)
    report = json.loads((args.periods / "report.json").read_text(encoding="utf-8"))
    period_manifest = json.loads((args.periods / "manifest.json").read_text(encoding="utf-8"))
    period_file = args.periods / "train_periods.parquet"
    events_file = args.events
    period_info = report["artifact"]
    event_info = period_manifest["input_artifacts"]["train_events.parquet"]
    if period_file.stat().st_size != period_info["bytes"] or sha256(period_file) != period_info["sha256"]:
        raise ValueError("Period Parquet differs from its recorded checksum")
    if events_file.stat().st_size != event_info["bytes"] or sha256(events_file) != event_info["sha256"]:
        raise ValueError("Training event Parquet differs from its recorded checksum")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "progress.jsonl").open("x").close()
    con = duckdb.connect(config={
        "threads": 4, "memory_limit": "12GB", "temp_directory": str(args.output / "spill")
    })
    active = category_stages(con, period_file, events_file, report)
    total_targets = sum(row["targets"] for row in report["periods"])
    save_manifest = {
        "source_commit": COMMIT,
        "source_hashes": SOURCE_HASHES,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "period_file_sha256": period_info["sha256"],
        "events_file_sha256": event_info["sha256"],
        "period_counts": [row["targets"] for row in report["periods"]],
        "active_categories": {str(p): len(active[p]) for p in active},
        "seed": args.seed,
        "model": {"maxlen": 50, "hidden_units": 150, "blocks": 2, "heads": 1,
                  "dropout": 0.3, "learning_rate": 0.0005, "batch_size": BATCH_SIZE},
        "memory_limit": MEMORY_SIZE,
        "validation_or_test_read": False,
        "duckdb": duckdb.__version__,
    }
    write_json(args.output / "manifest.json", save_manifest)

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Full run requires its allocated GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    sys.path.insert(0, str(args.source.resolve()))
    model_args = SimpleNamespace(
        maxlen=MAXLEN, hidden_units=150, num_blocks=2, num_heads=1,
        dropout_rate=0.3, random_seed=args.seed, l2_emb=0.0,
        disable_distillation=False,
    )
    config = tf.ConfigProto(
        allow_soft_placement=True, intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    started = time.monotonic()
    previous_checkpoint = None
    previous_ader_checkpoint = None
    previous_dropout_checkpoint = None
    memory_inputs = np.empty((0, MAXLEN), dtype=np.int32)
    memory_targets = np.empty(0, dtype=np.int32)
    memory_logits = np.empty((0, len(active[1])), dtype=np.float32)
    period_results = []
    id_map = np.zeros(max(active[8]) + 1, dtype=np.int32)

    for period in range(1, 9):
        stage = active[period]
        for stage_id, category in enumerate(stage, 1):
            id_map[category] = stage_id
        n = report["periods"][period - 1]["targets"]
        inputs, targets = load_period(
            con, period_file, events_file, period, n, id_map, args.output
        )
        old_count = len(active[period - 1]) if period > 1 else 0
        weight = 0.8 * math.sqrt(
            (old_count / len(stage)) * (len(memory_inputs) / n)
        ) if period > 1 else 0.0
        model, vanilla_op, ader_op, saver = build_model(
            tf, model_args, len(active[8]), weight
        )
        if period == 1:
            order = np.random.RandomState(args.seed * 101 + period).permutation(n)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                for start in range(0, n, BATCH_SIZE):
                    ix = order[start : start + BATCH_SIZE]
                    sess.run(vanilla_op, feed_for(model, inputs[ix], targets[ix], len(stage), model_args))
                saved_step = int(sess.run(model.global_step))
                if saved_step != math.ceil(n / BATCH_SIZE):
                    raise RuntimeError("Period 1 optimizer step count is incorrect")
                previous_checkpoint = saver.save(sess, str(args.output / "shared-period-1.ckpt"))
                memory_inputs, memory_targets, memory_logits = select_memory(
                    sess, model, inputs, targets, memory_inputs, memory_targets,
                    len(stage), model_args, MEMORY_SIZE, args.seed + period,
                )
            previous_ader_checkpoint = previous_checkpoint
            previous_dropout_checkpoint = previous_checkpoint
            method_results = {"shared_initial_period": saved_step}
        else:
            previous_dropout_checkpoint, dropout_report = train_branch(
                tf, config, model, saver, vanilla_op, previous_dropout_checkpoint,
                args.output, "dropout", inputs, targets, len(stage), model_args,
                period, args.seed,
            )
            previous_ader_checkpoint, ader_report = train_branch(
                tf, config, model, saver, ader_op, previous_ader_checkpoint,
                args.output, "ader", inputs, targets, len(stage), model_args,
                period, args.seed, memory_inputs, memory_logits,
            )
            method_results = {"dropout": dropout_report, "ader": ader_report}
            if period < 8:
                with tf.Session(config=config) as sess:
                    saver.restore(sess, previous_ader_checkpoint)
                    memory_inputs, memory_targets, memory_logits = select_memory(
                        sess, model, inputs, targets, memory_inputs, memory_targets,
                        len(stage), model_args, MEMORY_SIZE, args.seed + period,
                    )
        period_results.append({
            "period": period,
            "targets": n,
            "active_categories": len(stage),
            "replay_saved_for_next_period": len(memory_inputs) if period < 8 else None,
            "lambda": weight,
            "methods": method_results,
        })
        write_progress(args.output / "progress.jsonl", period_results[-1])
        print(json.dumps(period_results[-1]), flush=True)
        del inputs, targets
        (args.output / f"period-{period}-inputs.npy").unlink()

    con.close()
    gpu_samples = []
    monitor = os.environ.get("ADER_GPU_MONITOR_FILE")
    if monitor and Path(monitor).exists():
        for line in Path(monitor).read_text(encoding="utf-8").splitlines():
            try:
                gpu_samples.append(int(line.strip()))
            except ValueError:
                continue
    final = {
        "status": "passed_training_only",
        "manifest": "manifest.json",
        "training_targets": total_targets,
        "periods": period_results,
        "seconds": time.monotonic() - started,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "sampled_peak_gpu_memory_mib": max(gpu_samples) if gpu_samples else None,
        "validation_or_test_read": False,
        "metrics": None,
        "limits": [
            "One full-data seed run only; no validation selection or scores.",
            "Final-test data was not read.",
            "GPU dropout is stochastic; repeatability is not established.",
        ],
    }
    write_json(args.output / "profile.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())
