"""Popularity audit boundaries without changing the sample evaluator contract."""

import pytest

from cp4285.experiment.evaluation import popularity_baseline


def test_popularity_uses_training_only_and_counts_unseen_targets():
    training = {"A": 12, "B": 4}
    targets = {"B": 1, "C": 1}
    report = popularity_baseline(training, targets, k=3)
    assert report["recall"] == 0.5
    assert report["mrr"] == 0.25
    assert report["unseen_target_events"] == 1
    assert report["macro_label_recall"] == 0.5
    assert popularity_baseline(training, targets, k=1)["recall"] == 0
    assert training == {"A": 12, "B": 4}


def test_popularity_ties_and_event_weighting():
    report = popularity_baseline({"B": 2, "A": 2}, {"A": 9, "B": 1}, k=1)
    assert report["recall"] == 0.9 and report["macro_label_recall"] == 0.5
    assert popularity_baseline({"A": 1}, {}, k=3)["recall"] is None


@pytest.mark.parametrize(
    "training,targets,k",
    [
        ({}, {"A": 1}, 3),
        ({"A": -1}, {"A": 1}, 3),
        ({"A": 1}, {"A": True}, 3),
        ({"A": 1}, {"A": 1}, 0),
    ],
)
def test_popularity_rejects_invalid_counts(training, targets, k):
    with pytest.raises(ValueError):
        popularity_baseline(training, targets, k=k)


def test_sql_audit_split_isolation_and_input_checks():
    duckdb = pytest.importorskip("duckdb")
    from cp4285.experiment.category_audit import analyze

    with duckdb.connect() as con:
        con.execute(
            "create table reviews(user_id varchar, parent_asin varchar, rating double, timestamp bigint)"
        )
        con.execute(
            "create table labels(parent_asin varchar, subcategory varchar, label_source varchar)"
        )
        con.executemany(
            "insert into labels values (?,?,?)",
            [("a", "A", "observed"), ("b", "B", "observed"), ("c", "C", "observed")],
        )
        rows = [("u1", "a" if i <= 12 else "b", 5.0, i * 1000) for i in range(1, 17)]
        rows += [
            ("u1", "b", 0.0, 17000),
            ("new", "c", 5.0, 18000),
            ("u1", "c", 5.0, 19000),
            ("new", "c", 5.0, 20000),
        ]
        con.executemany("insert into reviews values (?,?,?,?)", rows)
        report = analyze(con)
        assert report["train_end_ms"] == 16000 and report["validation_end_ms"] == 18000
        assert report["reserved_test_rows"] == 2
        assert report["zero_rating_rows_retained"] == 1
        baseline = report["baseline"]["observed_only"]
        assert baseline["all_known_validation_events"]["3"]["recall"] == 0.5
        assert baseline["validation_reviewers_with_training_history"]["3"]["recall"] == 1.0
        con.execute("update reviews set parent_asin='a' where timestamp>18000")
        con.execute("drop table training_users")
        con.execute("drop view joined")
        assert analyze(con) == report
        con.execute("drop table training_users")
        con.execute("drop view joined")
        con.execute("insert into labels values ('a','B','observed')")
        with pytest.raises(ValueError, match="duplicate"):
            analyze(con)
