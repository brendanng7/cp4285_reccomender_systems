"""Prepare compact next-category histories from the verified review audit."""

import json
from pathlib import Path

from .artifacts import save_json
from .category_audit import sha256_file

MAX_HISTORY = 50


def prepare_sequences(con, train_end, validation_end):
    """Read reviews/labels; create train_events, vocabulary and queries tables.

    Histories are inclusive position ranges in train_events for the same user.
    Unknown labels and timestamp groups with multiple events break the history.
    """
    if not 0 < train_end < validation_end:
        raise ValueError("Expected increasing positive cutoff timestamps")
    if con.execute("select count(*)-count(distinct parent_asin) from labels").fetchone()[0]:
        raise ValueError("Duplicate/null category mapping IDs")
    if con.execute("""
        select count(*) from reviews where user_id is null or user_id='' or
        parent_asin is null or parent_asin='' or timestamp is null or timestamp<0
    """).fetchone()[0]:
        raise ValueError("Invalid review identity or timestamp")
    if con.execute("""
        select count(*) from labels where label_source is null or
        label_source not in ('observed','predicted','unknown') or
        (label_source in ('observed','predicted') and
         (subcategory is null or trim(subcategory)='')) or
        (label_source='unknown' and subcategory is not null)
    """).fetchone()[0]:
        raise ValueError("Invalid label/source")
    con.execute(
        """
        create table ordered_train as
        with grouped as (
            select r.user_id, r.timestamp, count(*) event_count,
                   min(r.parent_asin) parent_asin, min(l.subcategory) subcategory,
                   min(l.label_source) label_source
            from reviews r left join labels l using(parent_asin)
            where timestamp<=? group by r.user_id,r.timestamp
        ), positioned as (
            select *, row_number() over(partition by user_id order by timestamp) position,
                   event_count=1 and subcategory is not null as usable
            from grouped
        )
        select *, max(case when not usable then position else 0 end)
            over(partition by user_id order by position rows unbounded preceding) barrier
        from positioned
    """,
        [train_end],
    )
    con.execute("""
        create table vocabulary as select subcategory,
        row_number() over(order by subcategory)::integer category_id
        from (select distinct subcategory from ordered_train where usable)
    """)
    con.execute("""
        create table train_events as
        select user_id,parent_asin,timestamp,position,subcategory,category_id,label_source
        from ordered_train join vocabulary using(subcategory) where usable
    """)
    con.execute(
        """
        create table queries as
        select 'train' as split, user_id,parent_asin,timestamp,subcategory,
               category_id,label_source,
               greatest(barrier+1,position-?) history_start,
               position-1 history_end
        from ordered_train join vocabulary using(subcategory)
        where usable and position>barrier+1
    """,
        [MAX_HISTORY],
    )
    # Select the first raw validation event before label/history filtering.
    con.execute(
        """
        create table validation_first as
        select r.user_id,r.timestamp,count(*) event_count,
               min(r.parent_asin) parent_asin,min(l.subcategory) subcategory,
               min(l.label_source) label_source
        from reviews r left join labels l using(parent_asin)
        join (select user_id,min(timestamp) timestamp from reviews
              where timestamp>? and timestamp<=? group by user_id) first
        using(user_id,timestamp)
        group by r.user_id,r.timestamp
    """,
        [train_end, validation_end],
    )
    con.execute(
        """
        create table validation_eligibility as
        select v.*,t.position history_end,
               greatest(t.barrier+1,t.position-?+1) history_start,
               case when v.event_count>1 then 'tied_first_timestamp'
                    when v.subcategory is null then 'unknown_first_label'
                    when t.user_id is null then 'no_training_history'
                    when not t.usable then 'history_ends_at_barrier'
                    else 'eligible' end reason
        from validation_first v left join (
            select * from ordered_train qualify
            row_number() over(partition by user_id order by position desc)=1
        ) t using(user_id)
    """,
        [MAX_HISTORY],
    )
    con.execute("""
        insert into queries
        select 'validation',user_id,parent_asin,timestamp,subcategory,category_id,
               label_source,history_start,history_end
        from validation_eligibility left join vocabulary using(subcategory)
        where reason='eligible'
    """)
    if con.execute(
        """
        select count(*) from queries q
        left join train_events a on q.user_id=a.user_id and q.history_start=a.position
        left join train_events b on q.user_id=b.user_id and q.history_end=b.position
        where a.position is null or b.position is null or
              q.history_end<q.history_start or q.history_end-q.history_start+1>? or
              a.timestamp>=q.timestamp or b.timestamp>=q.timestamp or b.timestamp>?
    """,
        [MAX_HISTORY, train_end],
    ).fetchone()[0]:
        raise ValueError("Invalid or noncausal history endpoints")
    report = {
        "train_end_ms": train_end,
        "validation_end_ms": validation_end,
        "max_history": MAX_HISTORY,
        "candidate_categories": con.execute("select count(*) from vocabulary").fetchone()[0],
        "training_input": dict(
            zip(
                (
                    "raw_events",
                    "timestamp_groups",
                    "tied_groups",
                    "tied_events",
                    "unknown_singleton_events",
                ),
                con.execute("""
                select sum(event_count),count(*),count(*) filter(where event_count>1),
                       coalesce(sum(event_count) filter(where event_count>1),0),
                       count(*) filter(where event_count=1 and not usable)
                from ordered_train
            """).fetchone(),
                strict=True,
            )
        ),
        "validation_users_by_reason": dict(
            con.execute(
                "select reason,count(*) from validation_eligibility group by reason order by reason"
            ).fetchall()
        ),
        "splits": {},
    }
    for split in ("train", "validation"):
        row = con.execute(
            """
            select count(*),count(distinct user_id),
                   min(history_end-history_start+1),max(history_end-history_start+1),
                   quantile_disc(history_end-history_start+1,[0.5,0.9,0.99]),
                   count(*) filter(where history_end-history_start+1=50),
                   count(*) filter(where category_id is null)
            from queries where split=?
        """,
            [split],
        ).fetchone()
        report["splits"][split] = dict(
            zip(
                (
                    "examples",
                    "users",
                    "min_history",
                    "max_history",
                    "history_quantiles_50_90_99",
                    "histories_at_cap",
                    "targets_outside_candidates",
                ),
                row,
                strict=True,
            )
        )
        report["splits"][split]["target_sources"] = dict(
            con.execute(
                "select label_source,count(*) from queries where split=? group by label_source",
                [split],
            ).fetchall()
        )
    return report


def run_preparation(audit_dir, output):
    import duckdb

    audit_dir, output = Path(audit_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a fresh sequence output directory")
    audit = json.loads((audit_dir / "audit.json").read_text())
    provenance = json.loads((audit_dir / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "manifest.json",
        {
            "source_audit": provenance,
            "audit_report_sha256": sha256_file(audit_dir / "audit.json"),
            "audit_database_sha256": sha256_file(audit_dir / "audit.duckdb"),
            "preparation_sha256": sha256_file(__file__),
            "duckdb": duckdb.__version__,
            "rules": {
                "labels": "observed and predicted; source flags preserved",
                "history": "1..50 strictly earlier training events, same user",
                "barriers": "unknown label or multiple raw reviews at one user timestamp",
                "repeated_categories": "retained; no merging, rating or frequency threshold",
                "validation": "first raw event per user after training cutoff; no skipping ahead",
                "candidates": "sorted distinct usable training labels; 0 reserved for padding",
                "test": "no test examples or category statistics produced",
            },
        },
    )
    with duckdb.connect(
        str(output / "sequences.duckdb"),
        config={"threads": 4, "memory_limit": "6GB", "temp_directory": str(output / "spill")},
    ) as con:
        # ATTACH takes a SQL literal, not a bound parameter.
        source = str((audit_dir / "audit.duckdb").resolve()).replace("'", "''")
        con.execute(f"attach '{source}' as source (read_only)")
        con.execute("create temp view reviews as select * from source.reviews")
        con.execute("create temp view labels as select * from source.labels")
        if con.execute("select count(*) from reviews").fetchone()[0] != audit["rows"]:
            raise ValueError("Audit input row count changed")
        report = prepare_sequences(con, audit["train_end_ms"], audit["validation_end_ms"])
        for table in ("train_events", "vocabulary", "queries"):
            con.execute(
                f"copy {table} to ? (format parquet, compression zstd)",
                [str(output / f"{table}.parquet")],
            )
    report["artifacts"] = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in ("train_events.parquet", "vocabulary.parquet", "queries.parquet")
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
