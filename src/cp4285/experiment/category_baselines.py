"""Score simple category rankings on frozen prepared validation queries."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from cp4285.models.category_baselines import rank_categories

from .artifacts import save_json
from .category_audit import sha256_file
from .evaluation import evaluate_rank_counts


def evaluate_baselines(con):
    """Read queries/train_events/vocabulary tables; return only aggregate results."""
    candidates = tuple(
        r[0]
        for r in con.execute("select subcategory from vocabulary order by subcategory").fetchall()
    )
    training = dict(
        con.execute("""
        select subcategory,count(*) from queries where split='train' group by subcategory
    """).fetchall()
    )
    ranking = rank_categories(training, candidates)
    popular_ranks = {s: r for r, s in enumerate(ranking, 1)}
    expected, users = con.execute("""
        select count(*),count(distinct user_id) from queries where split='validation'
    """).fetchone()
    if not expected or users != expected:
        raise ValueError("Need one validation query per user")
    # Aggregate by last category and target: no 1.5M-by-283 score matrix is needed.
    groups = con.execute("""
        select e.subcategory last_category,q.subcategory target,q.label_source,
               case when q.history_end-q.history_start+1=1 then '1'
                    when q.history_end-q.history_start+1<=5 then '2-5' else '6-50' end,
               count(*)
        from queries q left join train_events e
             on q.user_id=e.user_id and q.history_end=e.position
        where q.split='validation'
        group by 1,2,3,4
    """).fetchall()
    if sum(n for *_, n in groups) != expected or any(last is None for last, *_ in groups):
        raise ValueError("Validation history endpoint join changed rows or lost context")
    histograms = defaultdict(lambda: defaultdict(Counter))
    promoted_ranks = {}
    for last, target, source, length, n in groups:
        if source not in ("observed", "predicted"):
            raise ValueError("Unexpected target label source")
        if last not in promoted_ranks:
            promoted_ranks[last] = {
                s: r
                for r, s in enumerate(rank_categories(training, candidates, last_category=last), 1)
            }
        ranks = {
            "popularity": popular_ranks.get(target, 0),
            "repeat_last_only": 1 if target == last else 0,
            "last_then_popularity": promoted_ranks[last].get(target, 0),
        }
        for cohort in ("all", f"target_source:{source}", f"history_length:{length}"):
            for method, rank in ranks.items():
                histograms[cohort][method][rank] += n
    results = {}
    for cohort, methods in sorted(histograms.items()):
        results[cohort] = {}
        for method, counts in methods.items():
            cutoffs = (1,) if method == "repeat_last_only" else (1, 3)
            results[cohort][method] = {str(k): evaluate_rank_counts(counts, k=k) for k in cutoffs}
    return {
        "training_targets": sum(training.values()),
        "validation_examples": expected,
        "candidate_categories": len(candidates),
        "popularity_ranking": [
            {"subcategory": s, "training_targets": training.get(s, 0)} for s in ranking
        ],
        "results": results,
        "limits": [
            "Validation only; same frozen queries and candidates for every method.",
            "Popularity counts prepared training targets, not all raw review events.",
            "Repeat-last-only returns one category; no artificial top-three ranking.",
            "Last-then-popularity promotes the last category and fills without duplicates.",
            "Target-source slices share the same combined-label training fit and histories; not an observed-only rebuild.",
            "No significance test, neural training, or final-test evaluation.",
        ],
    }


def run_baselines(sequence_dir, output):
    import duckdb

    sequence_dir, output = Path(sequence_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a fresh baseline output directory")
    source_report = json.loads((sequence_dir / "report.json").read_text())
    source_manifest = json.loads((sequence_dir / "manifest.json").read_text())
    for name in ("queries.parquet", "train_events.parquet", "vocabulary.parquet"):
        info = source_report["artifacts"][name]
        if (sequence_dir / name).stat().st_size != info["bytes"] or sha256_file(
            sequence_dir / name
        ) != info["sha256"]:
            raise ValueError("Prepared artifacts differ from recorded checksums")
    output.mkdir(parents=True, exist_ok=False)
    from cp4285.models import category_baselines

    from . import evaluation

    save_json(
        output / "manifest.json",
        {
            "source_preparation": source_manifest,
            "source_report_sha256": sha256_file(sequence_dir / "report.json"),
            "input_artifacts": source_report["artifacts"],
            "code_sha256": {
                name: sha256_file(path)
                for name, path in {
                    "experiment": __file__,
                    "models": category_baselines.__file__,
                    "evaluation": evaluation.__file__,
                }.items()
            },
            "duckdb": duckdb.__version__,
        },
    )
    with duckdb.connect(
        config={"threads": 4, "memory_limit": "6GB", "temp_directory": str(output / "spill")}
    ) as con:
        for table in ("queries", "train_events", "vocabulary"):
            con.read_parquet(str(sequence_dir / f"{table}.parquet")).create_view(table)
        report = evaluate_baselines(con)
    if report["validation_examples"] != source_report["splits"]["validation"]["examples"]:
        raise ValueError("Validation count differs from preparation")
    if report["training_targets"] != source_report["splits"]["train"]["examples"]:
        raise ValueError("Training count differs from preparation")
    save_json(output / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "popularity_ranking"}, indent=2),
        flush=True,
    )
