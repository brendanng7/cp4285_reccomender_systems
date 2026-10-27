"""Baseline ranking semantics and exact-query aggregation."""

import json

import pytest

from cp4285.experiment.category_audit import sha256_file
from cp4285.experiment.category_baselines import evaluate_baselines, run_baselines
from cp4285.experiment.evaluation import evaluate_rank_counts
from cp4285.models.category_baselines import rank_categories


def test_rankings_promote_last_without_duplicates_and_keep_zero_count_candidates():
    counts = {"B": 2, "A": 2}
    assert rank_categories(counts, ("C", "B", "A")) == ("A", "B", "C")
    assert rank_categories(counts, ("C", "B", "A"), last_category="C") == ("C", "A", "B")
    assert rank_categories(counts, ("C", "B", "A"), last_category="A") == ("A", "B", "C")
    with pytest.raises(ValueError):
        rank_categories(counts, ("A", "B"), last_category="C")


def test_rank_metric_misses_denominator_and_invalid_input():
    r = evaluate_rank_counts({0: 2, 1: 1, 3: 3, 4: 4}, k=3)
    assert r["recall"] == 0.4 and r["mrr"] == 0.2
    assert r["examples"] == 10 and r["unranked_targets"] == 2
    assert evaluate_rank_counts({}, k=1)["recall"] is None
    with pytest.raises(ValueError):
        evaluate_rank_counts({True: 1}, k=3)
    with pytest.raises(ValueError):
        evaluate_rank_counts({1: 1}, k=0)


def fixture():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("create table vocabulary(subcategory varchar)")
    con.executemany("insert into vocabulary values (?)", [("A",), ("B",), ("C",), ("D",)])
    con.execute("create table train_events(user_id varchar,position integer,subcategory varchar)")
    con.executemany(
        "insert into train_events values (?,?,?)",
        [("u1", 1, "B"), ("u2", 2, "C"), ("u3", 1, "A"), ("u4", 1, "D"), ("u5", 1, "A")],
    )
    con.execute(
        "create table queries(split varchar,user_id varchar,subcategory varchar,label_source varchar,history_start integer,history_end integer)"
    )
    con.executemany(
        "insert into queries values (?,?,?,?,?,?)",
        [("train", "t", s, "observed", 1, 1) for s in ("A", "A", "A", "B", "C")]
        + [
            ("validation", "u1", "B", "observed", 1, 1),
            ("validation", "u2", "A", "predicted", 1, 2),
            ("validation", "u3", "C", "observed", 1, 1),
            ("validation", "u4", "D", "observed", 1, 1),
            ("validation", "u5", "X", "observed", 1, 1),
        ],
    )
    return con


def test_sql_aggregation_and_validation_cannot_fit_rankings():
    with fixture() as con:
        report = evaluate_baselines(con)
        r = report["results"]["all"]
        assert r["popularity"]["1"]["recall"] == 0.2
        assert r["popularity"]["3"]["recall"] == 0.6
        assert r["popularity"]["3"]["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 5)
        assert r["repeat_last_only"]["1"]["recall"] == 0.4
        assert r["last_then_popularity"]["3"]["recall"] == 0.8
        assert r["last_then_popularity"]["3"]["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1) / 5)
        assert r["last_then_popularity"]["3"]["unranked_targets"] == 1
        assert (
            sum(
                report["results"]["target_source:" + s]["popularity"]["1"]["examples"]
                for s in ("observed", "predicted")
            )
            == 5
        )
        con.execute("update queries set subcategory='D' where split='validation'")
        assert evaluate_baselines(con)["popularity_ranking"] == report["popularity_ranking"]
        con.execute("delete from train_events where user_id='u1'")
        with pytest.raises(ValueError, match="endpoint"):
            evaluate_baselines(con)


def test_file_verification_full_run_and_no_overwrite(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    artifacts = {}
    with fixture() as con:
        for table in ("queries", "vocabulary", "train_events"):
            path = source / f"{table}.parquet"
            con.execute(f"copy {table} to ? (format parquet)", [str(path)])
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (source / "report.json").write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "splits": {"train": {"examples": 5}, "validation": {"examples": 5}},
            }
        )
    )
    (source / "manifest.json").write_text('{"data":"synthetic"}')
    output = tmp_path / "results"
    run_baselines(source, output)
    assert json.loads((output / "report.json").read_text())["validation_examples"] == 5
    with pytest.raises(FileExistsError):
        run_baselines(source, output)
    with (source / "queries.parquet").open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError, match="checksums"):
        run_baselines(source, tmp_path / "bad-input")
