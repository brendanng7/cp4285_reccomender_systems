"""Period assignment must preserve targets, ties and frozen history references."""

import pytest

from cp4285.experiment.category_periods import prepare_periods, run_preparation


def test_periods_preserve_targets_ties_and_ignore_validation():
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect() as con:
        con.execute(
            "create table queries(split varchar,user_id varchar,parent_asin varchar,timestamp bigint,subcategory varchar,history_start int,history_end int)"
        )
        rows = [("train", f"u{i}", f"p{i}", i // 2 * 1000, "A", 1, 2) for i in range(40)]
        con.execute(
            "create table train_events(user_id varchar,timestamp bigint,subcategory varchar)"
        )
        con.execute("insert into train_events values ('context-only',1000,'B')")
        con.execute("create table vocabulary(subcategory varchar)")
        con.executemany("insert into vocabulary values (?)", [("A",), ("B",)])
        rows += [("validation", "v", "pv", 100000, "FUTURE", 3, 4)]
        con.executemany("insert into queries values (?,?,?,?,?,?,?)", rows)
        first = prepare_periods(con)
        assert first["training_targets"] == 40 and len(first["periods"]) == 8
        assert [p["cumulative_target_categories"] for p in first["periods"]] == [1] * 8
        assert [p["cumulative_seen_categories"] for p in first["periods"]] == [2] * 8
        assert first["candidate_categories_without_training_targets"] == 1
        assert (
            con.execute(
                "select count(*) from (select timestamp,count(distinct period) n from training_periods group by timestamp) where n>1"
            ).fetchone()[0]
            == 0
        )
        assert (
            con.execute(
                "select count(*) from training_periods where split!='train' or history_start!=1 or history_end!=2"
            ).fetchone()[0]
            == 0
        )
        assert (
            con.execute(
                "select * exclude(period) from training_periods except all select * from queries where split='train'"
            ).fetchall()
            == []
        )
        assert (
            con.execute(
                "select * from queries where split='train' except all select * exclude(period) from training_periods"
            ).fetchall()
            == []
        )
        con.execute("update queries set timestamp=1,subcategory='CHANGED' where split='validation'")
        con.execute("drop table training_periods")
        assert prepare_periods(con) == first


def test_periods_reject_insufficient_distinct_times_and_existing_output(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect() as con:
        con.execute("create table queries as select 'train' split,1::bigint timestamp")
        with pytest.raises(ValueError, match="distinct"):
            prepare_periods(con)
    with pytest.raises(FileExistsError):
        run_preparation(tmp_path / "absent", tmp_path)


def test_period_export_and_checksum_verification(tmp_path):
    import json

    from cp4285.experiment.category_audit import sha256_file

    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {}
    with duckdb.connect() as con:
        con.execute(
            "create table queries as select 'train' split, 'user'||i user_id, 'item'||i parent_asin, (1000*i)::bigint timestamp, 'A' subcategory, 1 history_start, 2 history_end from range(40) r(i)"
        )
        con.execute(
            "create table train_events as select user_id,timestamp,subcategory from queries"
        )
        con.execute("create table vocabulary as select distinct subcategory from queries")
        statements = {
            "queries": "select * from queries",
            "train_events": "select user_id,timestamp,subcategory from train_events",
            "vocabulary": "select distinct subcategory from queries",
        }
        for name, query in statements.items():
            path = source / f"{name}.parquet"
            con.execute(f"copy ({query}) to ? (format parquet)", [str(path)])
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (source / "report.json").write_text(
        json.dumps({"artifacts": artifacts, "splits": {"train": {"examples": 40}}})
    )
    (source / "manifest.json").write_text('{"data":"synthetic"}')
    output = tmp_path / "periods"
    run_preparation(source, output)
    report = json.loads((output / "report.json").read_text())
    assert report["training_targets"] == 40
    assert report["artifact"]["sha256"] == sha256_file(output / "train_periods.parquet")
    with duckdb.connect() as con:
        assert (
            con.execute(
                "select count(*) from read_parquet(?)", [str(output / "train_periods.parquet")]
            ).fetchone()[0]
            == 40
        )
    with (source / "queries.parquet").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksums"):
        run_preparation(source, tmp_path / "changed")
