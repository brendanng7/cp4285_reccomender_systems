"""Bounded Electronics category-frequency audit; no ADER training or test scoring."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import save_json
from .evaluation import popularity_baseline


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(connection):
    """Analyze typed reviews/labels tables. Only timestamps select the 80/10/10 cuts."""
    con = connection
    if con.execute(
        "select count(*) from reviews where user_id is null or user_id='' or parent_asin is null or parent_asin='' or timestamp is null or timestamp<0 or rating is null or not isfinite(rating) or rating<0 or rating>5"
    ).fetchone()[0]:
        raise ValueError("Invalid review identity, timestamp or rating")
    if con.execute("select count(*)-count(distinct parent_asin) from labels").fetchone()[0]:
        raise ValueError("Category mapping contains duplicate/null product IDs")
    if con.execute(
        "select count(*) from labels where label_source is null or label_source not in ('observed','predicted','unknown') or (label_source in ('observed','predicted') and (subcategory is null or trim(subcategory)='')) or (label_source='unknown' and subcategory is not null)"
    ).fetchone()[0]:
        raise ValueError("Invalid category label/source")
    n, minimum, maximum, cuts = con.execute(
        "select count(*), min(timestamp), max(timestamp), quantile_disc(timestamp, [0.8, 0.9]) from reviews"
    ).fetchone()
    if not n or not minimum < cuts[0] < cuts[1] < maximum:
        raise ValueError("Not enough distinct timestamps for chronological audit")
    train_end, valid_end = cuts
    splits = {
        "train": ("timestamp <= ?", [train_end]),
        "validation": ("timestamp > ? and timestamp <= ?", [train_end, valid_end]),
    }
    # Validation/test data never enters training counts or history eligibility.
    con.execute(
        "create temp table training_users as select user_id from reviews where timestamp <= ? group by user_id",
        [train_end],
    )
    con.execute(
        "create temp view joined as select r.*, l.subcategory, l.label_source, l.parent_asin is not null as has_metadata from reviews r left join labels l using(parent_asin)"
    )
    report = {
        "rows": n,
        "zero_rating_rows_retained": con.execute(
            "select count(*) from reviews where rating=0"
        ).fetchone()[0],
        "timestamp_range": [minimum, maximum],
        "train_end_ms": train_end,
        "validation_end_ms": valid_end,
        "splits": {},
        "baseline": {},
        "reserved_test_rows": con.execute(
            "select count(*) from reviews where timestamp > ?", [valid_end]
        ).fetchone()[0],
    }
    counts = {}
    for name, (condition, parameters) in splits.items():
        groups = con.execute(
            f"select subcategory, label_source, count(*) from joined where {condition} group by subcategory, label_source order by subcategory, label_source",
            parameters,
        ).fetchall()
        observed = {s: count for s, source, count in groups if source == "observed"}
        known = dict(observed)
        for s, source, count in groups:
            if source == "predicted":
                known[s] = known.get(s, 0) + count
        counts[name] = {"observed_only": observed, "observed_and_predicted": known}
        source_counts = con.execute(
            f"select case when not has_metadata then 'unmatched' else label_source end, count(*) from joined where {condition} group by 1 order by 1",
            parameters,
        ).fetchall()
        total = sum(count for _, count in source_counts)
        report["splits"][name] = {
            "rows": total,
            "label_source_counts": dict(source_counts),
            "category_event_counts": [
                {"subcategory": s, "label_source": source, "reviews": count}
                for s, source, count in groups
            ],
            "known_categories": len(known),
            "top20_known_event_share": sum(sorted(known.values(), reverse=True)[:20])
            / sum(known.values())
            if known
            else None,
        }
    if sum(s["rows"] for s in report["splits"].values()) + report["reserved_test_rows"] != n:
        raise ValueError("Join/split changed the number of review events")
    for policy in ("observed_only", "observed_and_predicted"):
        training = counts["train"][policy]
        condition = (
            "label_source='observed'"
            if policy == "observed_only"
            else "label_source in ('observed','predicted')"
        )
        warm = dict(
            con.execute(
                f"select subcategory, count(*) from joined j where timestamp > ? and timestamp <= ? and {condition} and exists(select 1 from training_users t where t.user_id=j.user_id) group by subcategory",
                [train_end, valid_end],
            ).fetchall()
        )
        rank = sorted(training, key=lambda s: (-training[s], s))
        report["baseline"][policy] = {
            "training_top20": [{"subcategory": s, "reviews": training[s]} for s in rank[:20]],
            "all_known_validation_events": {
                str(k): popularity_baseline(training, counts["validation"][policy], k=k)
                for k in (1, 3, 5, 20)
            },
            "validation_reviewers_with_training_history": {
                str(k): popularity_baseline(training, warm, k=k) for k in (1, 3, 5, 20)
            },
        }
    report["training_history_lengths"] = con.execute(
        "select case when n=1 then 1 when n=2 then 2 when n<=5 then 3 else 6 end as bucket, count(*) from (select user_id,count(*) n from reviews where timestamp<=? group by user_id) group by 1 order by 1",
        [train_end],
    ).fetchall()
    report["training_history_bucket_labels"] = {
        "1": "one review",
        "2": "two reviews",
        "3": "three to five reviews",
        "6": "six or more reviews",
    }
    report["limits"] = [
        "Preliminary temporal audit split, not a finalized ADER sequence protocol.",
        "Reviews of all ratings retained, including source zero ratings; ratings are not model targets. No taxonomy merging or rare-label filtering.",
        "Test labels/frequencies/scores not inspected; only raw timestamps and test row count used.",
        "Both baselines rank labels using training review frequencies only; unseen validation labels are misses.",
        "All-event baseline includes first-time reviewers; second cohort requires an earlier training review.",
        "No individual sequence model trained; metadata and predicted labels are a retrospective snapshot.",
        "Official source is advertised as deduplicated; repeated user/product pairs not independently audited.",
    ]
    return report


def run_audit(reviews, label_root, output):
    import duckdb

    reviews, label_root, output = Path(reviews), Path(label_root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh audit output directory")
    label_file = label_root / "item_categories/domain=Electronics/data_0.parquet"
    review_manifest = json.loads((reviews.parent / "DOWNLOAD_VERIFIED.json").read_text())
    label_manifest = json.loads((label_root / "DOWNLOAD_VERIFIED.json").read_text())
    expected = next(
        f["sha256"]
        for f in label_manifest["files"]
        if f["path"] == str(label_file.relative_to(label_root))
    )
    if sha256_file(reviews) != review_manifest["sha256"] or sha256_file(label_file) != expected:
        raise ValueError("Downloaded inputs differ from recorded checksums")
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "manifest.json",
        {
            "review_download": review_manifest,
            "label_revision": label_manifest["revision"],
            "label_sha256": expected,
            "duckdb": duckdb.__version__,
            "audit_sha256": sha256_file(__file__),
            "evaluation_sha256": sha256_file(Path(__file__).with_name("evaluation.py")),
            "provisional_primary_k": 3,
            "split": "timestamp-only 80/10/10 quantile boundaries; ties stay together",
        },
    )
    with duckdb.connect(
        str(output / "audit.duckdb"),
        config={"threads": 4, "memory_limit": "6GB", "temp_directory": str(output / "spill")},
    ) as con:
        print("Reading official review CSV into the audit database", flush=True)
        con.execute(
            "create table reviews as select * from read_csv(?, header=true, columns={'user_id':'VARCHAR','parent_asin':'VARCHAR','rating':'DOUBLE','timestamp':'BIGINT'})",
            [str(reviews)],
        )
        con.execute(
            "create table labels as select parent_asin, subcategory, label_source from read_parquet(?)",
            [str(label_file)],
        )
        print("Counting labels and evaluating the training-only popularity ranking", flush=True)
        report = analyze(con)
    if report["timestamp_range"][1] < 10**12 or report["timestamp_range"][1] > int(
        datetime.now(UTC).timestamp() * 1000
    ):
        raise ValueError("Review timestamp units/range do not match expected Unix milliseconds")
    report["cutoffs_utc"] = {
        k: datetime.fromtimestamp(report[k] / 1000, UTC).isoformat()
        for k in ("train_end_ms", "validation_end_ms")
    }
    save_json(output / "audit.json", report)
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "cutoffs_utc": report["cutoffs_utc"],
                "baseline": report["baseline"],
            },
            indent=2,
        ),
        flush=True,
    )
