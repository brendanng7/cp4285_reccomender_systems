"""Prepare eight balanced chronological blocks using training target times only."""

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from .artifacts import save_json
from .category_audit import sha256_file

PERIOD_COUNT = 8


def prepare_periods(con):
    """Read queries and materialize training_periods without altering history references."""
    total, minimum, maximum, cuts = con.execute("""
        select count(*),min(timestamp),max(timestamp),
        quantile_disc(timestamp,[0.125,0.25,0.375,0.5,0.625,0.75,0.875])
        from queries where split='train'
    """).fetchone()
    if not total or len({minimum, *cuts, maximum}) != PERIOD_COUNT + 1:
        raise ValueError("Not enough distinct training timestamps for eight periods")
    if minimum < 0:
        raise ValueError("Training timestamps must be nonnegative")
    branches = " ".join(f"when timestamp <= ? then {i}" for i in range(1, PERIOD_COUNT))
    con.execute(
        f"""
        create table training_periods as select *,
        case {branches} else {PERIOD_COUNT} end::integer period
        from queries where split='train'
    """,
        cuts,
    )
    periods = []
    for i in range(1, PERIOD_COUNT + 1):
        n, users, lo, hi, labels = con.execute(
            """
            select count(*),count(distinct user_id),min(timestamp),max(timestamp),
            count(distinct subcategory) from training_periods where period=?
        """,
            [i],
        ).fetchone()
        if not n:
            raise ValueError("An empty period was created")
        cumulative_labels = con.execute(
            """
            select count(distinct subcategory) from training_periods
            where period<=? and subcategory is not null
        """,
            [i],
        ).fetchone()[0]
        cumulative_input_labels = con.execute(
            "select count(distinct subcategory) from train_events where timestamp<=?", [hi]
        ).fetchone()[0]
        cumulative_seen_labels = con.execute(
            """
            select count(distinct subcategory) from (
                select subcategory from training_periods where period<=?
                union
                select subcategory from train_events where timestamp<=?
            )
            """,
            [i, hi],
        ).fetchone()[0]
        periods.append(
            {
                "period": i,
                "targets": n,
                "users": users,
                "target_categories": labels,
                "cumulative_target_categories": cumulative_labels,
                "cumulative_history_categories": cumulative_input_labels,
                "cumulative_seen_categories": cumulative_seen_labels,
                "start_exclusive_ms": cuts[i - 2] if i > 1 else None,
                "end_inclusive_ms": cuts[i - 1] if i < PERIOD_COUNT else maximum,
                "first_target_ms": lo,
                "last_target_ms": hi,
                "first_target_utc": datetime.fromtimestamp(lo / 1000, UTC).isoformat(),
                "last_target_utc": datetime.fromtimestamp(hi / 1000, UTC).isoformat(),
            }
        )
    if sum(p["targets"] for p in periods) != total:
        raise ValueError("Period assignment changed training target count")
    if any(a["last_target_ms"] >= b["first_target_ms"] for a, b in pairwise(periods)):
        raise ValueError("Periods overlap or split an equal-timestamp group")
    all_targets = con.execute(
        "select count(distinct subcategory) from training_periods"
    ).fetchone()[0]
    all_contexts = con.execute("select count(distinct subcategory) from train_events").fetchone()[0]
    candidates_without_targets = con.execute("""
        select count(*) from vocabulary v
        where not exists (select 1 from training_periods p where p.subcategory=v.subcategory)
    """).fetchone()[0]
    return {
        "training_targets": total,
        "period_count": PERIOD_COUNT,
        "ever_supervised_target_categories": all_targets,
        "training_context_categories": all_contexts,
        "candidate_categories_without_training_targets": candidates_without_targets,
        "periods": periods,
    }


def run_preparation(sequence_dir, output):
    import duckdb

    sequence_dir, output = Path(sequence_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a fresh period output directory")
    source_report = json.loads((sequence_dir / "report.json").read_text())
    source_manifest = json.loads((sequence_dir / "manifest.json").read_text())
    for name in ("queries.parquet", "train_events.parquet", "vocabulary.parquet"):
        info = source_report["artifacts"][name]
        path = sequence_dir / name
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError("Prepared artifacts differ from recorded checksums")
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "manifest.json",
        {
            "source_preparation": source_manifest,
            "source_report_sha256": sha256_file(sequence_dir / "report.json"),
            "input_artifacts": source_report["artifacts"],
            "code_sha256": sha256_file(__file__),
            "duckdb": duckdb.__version__,
            "assignment": "training-target timestamp octiles, equal timestamps stay together",
            "history": "original same-user training ranges unchanged; older periods may supply context",
            "validation": "not assigned to periods or used to choose boundaries",
            "test": "not read",
        },
    )
    with duckdb.connect(
        config={"threads": 4, "memory_limit": "6GB", "temp_directory": str(output / "spill")}
    ) as con:
        con.read_parquet(str(sequence_dir / "queries.parquet")).create_view("queries")
        con.read_parquet(str(sequence_dir / "train_events.parquet")).create_view("train_events")
        con.read_parquet(str(sequence_dir / "vocabulary.parquet")).create_view("vocabulary")
        report = prepare_periods(con)
        if report["training_targets"] != source_report["splits"]["train"]["examples"]:
            raise ValueError("Training count differs from source preparation")
        con.execute(
            "copy (select * from training_periods order by period,timestamp,user_id,parent_asin) to ? (format parquet, compression zstd)",
            [str(output / "train_periods.parquet")],
        )
    path = output / "train_periods.parquet"
    report["artifact"] = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
