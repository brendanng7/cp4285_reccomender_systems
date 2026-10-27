"""Sequence rules checked with tiny synthetic data, never private review identities."""

import pytest

from cp4285.experiment.category_sequences import prepare_sequences, run_preparation


def build(rows, database=":memory:"):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(database)
    con.execute(
        "create table reviews(user_id varchar,parent_asin varchar,rating double,timestamp bigint)"
    )
    con.execute("create table labels(parent_asin varchar,subcategory varchar,label_source varchar)")
    con.executemany(
        "insert into labels values (?,?,?)",
        [
            ("a", "A", "observed"),
            ("b", "B", "predicted"),
            ("c", "C", "observed"),
            ("d", "D", "observed"),
            ("x", None, "unknown"),
        ],
    )
    con.executemany("insert into reviews values (?,?,?,?)", [(u, p, 5, t) for u, p, t in rows])
    return con


def fixture_rows():
    rows = [
        ("u", "a", 1),
        ("u", "a", 2),
        ("u", "a", 3),
        ("u", "b", 3),
        ("u", "c", 4),
        ("u", "b", 5),
        ("u", "x", 6),
        ("u", "a", 7),
        ("u", "b", 8),
        ("u", "a", 201),
        ("u", "c", 202),
        ("u", "d", 400),
        ("cold", "a", 201),
        ("tail", "x", 9),
        ("tail", "a", 201),
        ("bad", "a", 1),
        ("bad", "x", 201),
        ("bad", "a", 202),
        ("tie", "a", 1),
        ("tie", "b", 201),
        ("tie", "c", 201),
        ("tie", "a", 202),
        ("novel", "a", 1),
        ("novel", "d", 201),
    ]
    rows += [("long", "a", i) for i in range(100, 160)] + [("long", "b", 201)]
    return rows


def test_histories_barriers_cap_first_event_and_unseen_label():
    with build(fixture_rows()) as con:
        report = prepare_sequences(con, 200, 300)
        queries = con.execute(
            "select timestamp,history_start,history_end from queries where user_id='u' order by timestamp"
        ).fetchall()
        assert queries == [(2, 1, 1), (5, 4, 4), (8, 7, 7), (201, 7, 8)]
        assert con.execute(
            "select subcategory from train_events where user_id='u' order by position"
        ).fetchall() == [("A",), ("A",), ("C",), ("B",), ("A",), ("B",)]
        assert con.execute(
            "select history_start,history_end from queries where user_id='long' and split='validation'"
        ).fetchone() == (11, 60)
        assert report["validation_users_by_reason"] == {
            "eligible": 3,
            "history_ends_at_barrier": 1,
            "no_training_history": 1,
            "tied_first_timestamp": 1,
            "unknown_first_label": 1,
        }
        assert report["splits"]["validation"]["targets_outside_candidates"] == 1
        assert report["candidate_categories"] == 3
        assert report["splits"]["train"]["examples"] == 62
        assert report["training_input"]["tied_events"] == 2
        assert report["training_input"]["unknown_singleton_events"] == 2
        # Expand every compact range and independently check ownership, order and size.
        for split, user, ts, start, end in con.execute(
            "select split,user_id,timestamp,history_start,history_end from queries"
        ).fetchall():
            history = con.execute(
                "select timestamp from train_events where user_id=? and position between ? and ? order by position",
                [user, start, end],
            ).fetchall()
            times = [t for (t,) in history]
            assert 1 <= len(times) == end - start + 1 <= 50
            assert times == sorted(set(times)) and max(times) < ts
            assert max(times) <= 200


def test_future_labels_cannot_change_training_or_first_validation():
    snapshots = []
    for changed in (False, True):
        with build(fixture_rows()) as con:
            if changed:
                con.execute("update reviews set parent_asin='d' where timestamp>201")
            report = prepare_sequences(con, 200, 300)
            snapshots.append(
                (
                    report,
                    con.execute("select * from train_events order by user_id,position").fetchall(),
                    con.execute(
                        "select * from queries order by split,user_id,timestamp"
                    ).fetchall(),
                )
            )
    assert snapshots[0] == snapshots[1]


def test_duplicate_mapping_rejected_and_existing_output_preserved(tmp_path):
    with build(fixture_rows()) as con:
        con.execute("insert into labels values ('a','other','observed')")
        with pytest.raises(ValueError, match="Duplicate"):
            prepare_sequences(con, 200, 300)
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_preparation(tmp_path / "missing", output)


def test_cli_preparation_exports_private_artifacts(tmp_path):
    import json

    audit = tmp_path / "audit"
    audit.mkdir()
    rows = fixture_rows()
    with build(rows, str(audit / "audit.duckdb")):
        pass
    (audit / "audit.json").write_text(
        json.dumps({"rows": len(rows), "train_end_ms": 200, "validation_end_ms": 300})
    )
    (audit / "manifest.json").write_text('{"data":"synthetic"}')
    output = tmp_path / "prepared"
    run_preparation(audit, output)
    report = json.loads((output / "report.json").read_text())
    assert report["splits"]["train"]["examples"] == 62
    assert set(report["artifacts"]) == {
        "train_events.parquet",
        "queries.parquet",
        "vocabulary.parquet",
    }
    for name, info in report["artifacts"].items():
        assert (output / name).stat().st_size == info["bytes"]
    with pytest.raises(FileExistsError):
        run_preparation(audit, output)
