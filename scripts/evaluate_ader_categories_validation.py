"""Evaluate completed eight-period runs on the frozen validation cohort only."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_ader_categories_full import (
    COMMIT,
    MAXLEN,
    SOURCE_HASHES,
    build_model,
    category_stages,
    sha256,
)

BATCH_SIZE = 256


def add_rank_counts(counter, ranks, candidate_count):
    histogram = np.bincount(ranks, minlength=candidate_count + 1)
    counter.update({rank: int(count) for rank, count in enumerate(histogram) if count})


def assert_metric_self_check():
    sample = Counter()
    add_rank_counts(sample, np.asarray([1, 3, 3], dtype=np.int32), 3)
    assert sample == Counter({1: 1, 3: 2})


def validation_arrays(con, queries, events, id_map, expected, output):
    x = np.empty((expected, MAXLEN), dtype=np.int32)
    y = np.empty(expected, dtype=np.int32)
    sources = np.empty(expected, dtype=np.int8)
    token_columns = ",".join(f"list_extract(tokens,{i})" for i in range(1, MAXLEN + 1))
    sql = f"""
        with q as (
            select row_number() over(order by timestamp,user_id,parent_asin) qid,
                   user_id,category_id target,label_source,history_start,history_end
            from read_parquet(?) where split='validation'
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
    cursor = con.execute(sql, [str(queries), str(events)])
    offset = 0
    while rows := cursor.fetchmany(32_768):
        block = np.asarray(rows, dtype=np.int32)
        end = offset + len(block)
        y[offset:end] = id_map[block[:, 0]]
        sources[offset:end] = block[:, 1]
        x[offset:end] = id_map[block[:, 2:]]
        if (
            np.any(sources[offset:end] == 0)
            or np.any(y[offset:end] <= 0)
            or np.any((block[:, 2:] > 0) & (x[offset:end] == 0))
            or np.any(x[offset:end, -1] == 0)
        ):
            raise ValueError("Validation target/source or history mapping is invalid")
        offset = end
    if offset != expected:
        raise ValueError("Validation history join changed the frozen query count")
    return x, y, sources


def score_model(tf, config, source, model_args, checkpoint, x, y, sources, global_ids):
    model, _, _, saver = build_model(tf, model_args, len(global_ids), 0.0)
    rank_counts = {name: Counter() for name in ("all", "observed", "predicted")}
    with tf.Session(config=config) as sess:
        saver.restore(sess, str(checkpoint))
        for start in range(0, len(y), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(y))
            scores = sess.run(
                model.logits,
                {
                    model.input_seq: x[start:end],
                    model.pos: y[start:end],
                    model.max_item: len(global_ids),
                    model.is_training: False,
                    model.dropout_rate: model_args.dropout_rate,
                },
            )
            row = np.arange(end - start)
            target_score = scores[row, y[start:end] - 1]
            target_global_id = global_ids[y[start:end] - 1]
            ties_before = (scores == target_score[:, None]) & (
                global_ids[None, :] < target_global_id[:, None]
            )
            ranks = 1 + (scores > target_score[:, None]).sum(axis=1) + ties_before.sum(axis=1)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite validation score")
            for name, mask in (
                ("all", np.ones(len(row), dtype=bool)),
                ("observed", sources[start:end] == 1),
                ("predicted", sources[start:end] == 2),
            ):
                add_rank_counts(rank_counts[name], ranks[mask], len(global_ids))
    result = {}
    for cohort, counts in rank_counts.items():
        total = sum(counts.values())
        if not total:
            raise ValueError(f"Validation cohort is empty: {cohort}")
        result[cohort] = {
            "examples": total,
            "Recall@1": counts[1] / total,
            "Recall@3": sum(counts[k] for k in range(1, 4)) / total,
            "MRR@3": sum(counts[k] / k for k in range(1, 4)) / total,
        }
    return result


def run(args):
    import duckdb
    import tensorflow.compat.v1 as tf

    assert_metric_self_check()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Choose a fresh validation output directory")
    sequence_dir, period_dir = args.sequences, args.periods
    source_report = json.loads((sequence_dir / "report.json").read_text(encoding="utf-8"))
    period_report = json.loads((period_dir / "report.json").read_text(encoding="utf-8"))
    period_manifest = json.loads((period_dir / "manifest.json").read_text(encoding="utf-8"))
    query_file = sequence_dir / "queries.parquet"
    event_file = sequence_dir / "train_events.parquet"
    period_file = period_dir / "train_periods.parquet"
    for path, info in (
        (query_file, source_report["artifacts"]["queries.parquet"]),
        (event_file, source_report["artifacts"]["train_events.parquet"]),
        (period_file, period_report["artifact"]),
    ):
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError("Validation input differs from its recorded artifact")
    if period_manifest["input_artifacts"]["train_events.parquet"]["sha256"] != sha256(event_file):
        raise ValueError("Period and sequence runs refer to different training histories")
    runs = [Path(path) for path in args.run_dir]
    if len(runs) != 3:
        raise ValueError("Provide exactly three completed seed run directories")
    manifests = [json.loads((path / "manifest.json").read_text()) for path in runs]
    profiles = [json.loads((path / "profile.json").read_text()) for path in runs]
    seeds = [item["seed"] for item in manifests]
    if sorted(seeds) != [0, 1, 2] or any(item["status"] != "passed_training_only" for item in profiles):
        raise ValueError("All three predeclared full-data training seeds must pass first")
    for item in manifests:
        if item["period_file_sha256"] != period_report["artifact"]["sha256"]:
            raise ValueError("Seed runs do not share the same prepared period input")
        if item["events_file_sha256"] != source_report["artifacts"]["train_events.parquet"]["sha256"]:
            raise ValueError("Seed runs do not share the same training histories")

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hashes = {}
    for run_dir, manifest in zip(runs, manifests, strict=True):
        for method in ("dropout", "ader"):
            prefix = run_dir / f"{method}-period-8.ckpt"
            files = [prefix.with_suffix(prefix.suffix + ".index"), *sorted(run_dir.glob(prefix.name + ".data-*"))]
            if len(files) < 2 or any(not file.is_file() for file in files):
                raise FileNotFoundError(f"Missing final {method} checkpoint for seed {manifest['seed']}")
            checkpoint_hashes[f"seed{manifest['seed']}_{method}"] = {
                file.name: sha256(file) for file in files
            }
    write_json(args.output / "manifest.json", {
        "source_commit": COMMIT,
        "source_hashes": SOURCE_HASHES,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "query_sha256": source_report["artifacts"]["queries.parquet"]["sha256"],
        "events_sha256": source_report["artifacts"]["train_events.parquet"]["sha256"],
        "period_sha256": period_report["artifact"]["sha256"],
        "seeds": {str(item["seed"]): str(path) for item, path in zip(manifests, runs, strict=True)},
        "checkpoint_hashes": checkpoint_hashes,
        "evaluation": "validation only, after all three full training seeds; no test rows read",
    })
    con = duckdb.connect(config={
        "threads": 4,
        "memory_limit": "12GB",
        "temp_directory": str(args.output / "spill"),
    })
    active = category_stages(con, period_file, event_file, period_report)
    global_ids = np.asarray(active[8], dtype=np.int32)
    id_map = np.zeros(int(global_ids.max()) + 1, dtype=np.int32)
    for stage_id, category in enumerate(global_ids, 1):
        id_map[category] = stage_id
    expected = source_report["splits"]["validation"]["examples"]
    unique_users = con.execute(
        "select count(distinct user_id) from read_parquet(?) where split='validation'",
        [str(query_file)],
    ).fetchone()[0]
    if unique_users != expected:
        raise ValueError("Expected one frozen validation query per person")
    x, y, sources = validation_arrays(
        con, query_file, event_file, id_map, expected, args.output
    )
    con.close()

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Validation scoring requires its allocated GPU")
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    # TF 2.15 Grappler crashes on this legacy graph during GPU inference.
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    methods = {}
    for run_dir, manifest in zip(runs, manifests, strict=True):
        seed = manifest["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        sys.path.insert(0, str(args.source.resolve()))
        model_args = SimpleNamespace(
            maxlen=MAXLEN, hidden_units=150, num_blocks=2, num_heads=1,
            dropout_rate=0.3, random_seed=seed, l2_emb=0.0,
            disable_distillation=False,
        )
        for method in ("dropout", "ader"):
            checkpoint = run_dir / f"{method}-period-8.ckpt"
            methods[f"seed{seed}_{method}"] = score_model(
                tf, config, args.source, model_args, checkpoint, x, y, sources, global_ids
            )
            print(json.dumps({f"seed{seed}_{method}": methods[f"seed{seed}_{method}"]}), flush=True)
    aggregate = {}
    for method in ("dropout", "ader"):
        aggregate[method] = {}
        for metric in ("Recall@1", "Recall@3", "MRR@3"):
            values = [methods[f"seed{seed}_{method}"]["all"][metric] for seed in sorted(seeds)]
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
    report = {
        "status": "passed_validation_only",
        "seeds": sorted(seeds),
        "periods": 8,
        "validation_examples": expected,
        "validation_users": unique_users,
        "candidate_categories": len(global_ids),
        "candidate_tie_break": "original frozen category ID order",
        "methods": methods,
        "aggregate_all_validation_users": aggregate,
        "test_read": False,
        "limits": [
            "Validation was opened only after all three predeclared seed runs completed.",
            "No test rows or labels were read.",
            "GPU dropout repeatability is not established.",
        ],
    }
    write_json(args.output / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--periods", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
