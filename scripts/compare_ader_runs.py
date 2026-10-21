"""Compare completed 16-window DIGINETICA ADER and Dropout runs; never select a run."""

import argparse
import json
from pathlib import Path

from profile_ader_window import build_command, mean_metrics, save_json

# Table 2, p. 6 of arXiv:2007.12000v1, converted from percentages.
PAPER = {
    "ader": {"Recall@20": 0.5021, "Recall@10": 0.3752, "MRR@20": 0.1732, "MRR@10": 0.1645},
    "dropout": {"Recall@20": 0.4907, "Recall@10": 0.3653, "MRR@20": 0.1686, "MRR@10": 0.1600},
}


def compare(ader, dropout):
    reports, manifests, means = {}, {}, {}
    for method, directory in (("ader", ader), ("dropout", dropout)):
        report = json.loads((directory / "profile.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            report.get("status") != "passed"
            or report.get("returncode") != 0
            or report.get("method") != method
            or manifest.get("method") != method
            or manifest.get("windows") != 16
            or manifest.get("dataset") != "DIGINETICA"
            or manifest.get("command") != build_command(method)
        ):
            raise ValueError("Expected a completed default-config 16-window " + method + " run")
        windows = report.get("windows", [])
        if [(w["train_period"], w["test_period"]) for w in windows] != [
            (i, i + 1) for i in range(16)
        ]:
            raise ValueError("Incomplete or unordered windows: " + method)
        if any(not 0 <= value <= 1 for w in windows for value in w["test_metrics"].values()):
            raise ValueError("Invalid metrics: " + method)
        means[method] = mean_metrics(windows)
        if means[method] != report.get("mean_test_metrics"):
            raise ValueError("Stored mean differs from unweighted per-window mean")
        reports[method], manifests[method] = report, manifest
    for key in (
        "source_commit",
        "source_hashes",
        "input_audit",
        "script_sha256",
        "packages",
        "deterministic_gpu_ops_enforced",
    ):
        if key not in manifests["ader"] or manifests["ader"][key] != manifests["dropout"].get(key):
            raise ValueError("Comparison provenance differs: " + key)
    pairs = list(zip(reports["ader"]["windows"], reports["dropout"]["windows"]))
    if any(
        a["continuation"]["active_items"] != d["continuation"]["active_items"] for a, d in pairs
    ):
        raise ValueError("Candidate catalogues differ")
    return {
        "ader_run": str(ader.resolve()),
        "dropout_run": str(dropout.resolve()),
        "aggregation": "Unweighted mean of all 16 test windows; seed 0",
        "means": means,
        "ader_minus_dropout_percentage_points": {
            k: 100 * (means["ader"][k] - means["dropout"][k]) for k in means["ader"]
        },
        "difference_from_paper_percentage_points": {
            method: {k: 100 * (means[method][k] - PAPER[method][k]) for k in means[method]}
            for method in means
        },
        "validation_splits_identical_by_test_period": {
            str(a["test_period"]): a["validation_sha256"] == d["validation_sha256"]
            for a, d in pairs
        },
        "per_window_difference_percentage_points": [
            {
                "test_period": a["test_period"],
                **{
                    k: 100 * (a["test_metrics"][k] - d["test_metrics"][k])
                    for k in a["test_metrics"]
                },
            }
            for a, d in pairs
        ],
        "limits": [
            "Single seed; GPU repeatability and historical-runtime parity unverified.",
            "Upstream validation overlap and method-dependent RNG/splits retained.",
            "No claimed statistical significance or automatic replication tolerance.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ader", type=Path, required=True)
    parser.add_argument("--dropout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.ader, args.dropout)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
