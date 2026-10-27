# Run the samples

From the repository root, with Python 3.12 and uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

**Development examples live in [Packages and building rules](PACKAGES-AND-BUILDING.md):** a complete model, shared evaluator, contract check and notebook recipe. The tests execute those examples and the notebook's Python cells. No dataset, GPU or numerical library is needed.

For the interactive notebook:

```sh
uv run --with jupyterlab jupyter lab notebooks/01_shared_experiment.ipynb
```

The optional launcher may be installed through uv. Restart the kernel and Run All. This checks synthetic plumbing, not a real recommender experiment or browser certification.

For the separately isolated authors' ADER implementation, see [local verification and cluster handoff](ADER.md#local-verification-and-cluster-handoff). `scripts/verify_ader.py` checks synthetic TensorFlow behavior and a two-window upstream smoke run; it is not a shared-package model or a real-data experiment runner.

The separately authorized ADER study passed initial and continuation resource pilots in user-provided reports. The [full 16-window comparison](ADER.md#full-16-window-comparison) uses `scripts/profile_ader_window.py --windows 16 --method ader|dropout`, `scripts/run_ader_full_slurm.sh` and `scripts/compare_ader_runs.py`. It retains the released evaluation choices and records compatibility differences, exact metrics and split fingerprints. Run it in the isolated cluster environment; it is outside the fixed A/B sample API. The user supplied a completed seed-0 comparison: see [results and limits](ADER.md#completed-seed-0-results). Full remote artifacts have not been retrieved by the agent.

## Electronics category audit

The user authorized an initial **next-subcategory** data audit before adapting ADER. `scripts/audit_amazon_categories.py` calls `experiment/category_audit.py`; popularity Recall/MRR arithmetic lives in `experiment/evaluation.py`. This is a frequency baseline, not a sequence model or the fixed A/B runner.

On a CPU allocation, with the downloaded files and their acquisition manifests available:

```sh
PYTHONPATH=src uv run --no-project --python 3.11 --with duckdb==2.0.0 \
  python scripts/audit_amazon_categories.py \
  --reviews data/amazon23-reviews/electronics-0core-20261024/Electronics.csv.gz \
  --label-root data/amazon23-item-categories/822d4745b11748ae5e24f6619e4c6c32a501f36b \
  --output runs/electronics-category-audit-new
```

Use a fresh output directory; inputs are checksum-verified against acquisition manifests. DuckDB remains an isolated optional dependency. The audit uses four threads, a 6 GB database memory limit and disk spill; the measured cluster run requests 12 GB host RAM. It produces a private on-disk database, `manifest.json` and aggregate `audit.json`. Keep databases, raw identities and datasets ignored.

The preliminary split uses global timestamp quantiles: earliest 80% training, next 10% validation, final 10% reserved. Equal timestamps stay together. Training-only category counts fix the popularity ranking; validation evaluates K=1,3,5,20. There are observed-only and observed-plus-predicted variants, each evaluated on all known-label validation events and on reviewers with a training history. Unknown/unmatched labels are counted separately, unseen known targets count as misses, and ratings are not targets (including two source zero ratings). Final-test category frequencies and scores are not queried. This split and cohort are not yet the ADER sequence protocol.

The small SQL check is optional in the standard-library environment and runs explicitly with:

```sh
uv run --with duckdb==2.0.0 pytest -q tests/test_category_audit.py
```

It checks temporal boundaries, test-label isolation, cold-start misses, history eligibility, zero-rating retention and duplicate-join rejection. Standard tests always check the count-based metric arithmetic. See [source and results](SOURCES.md#electronics-review-event-audit) for the real-data findings.

## Electronics chronological histories — 25 October 2026

The user authorized constructing histories and defining which examples qualify. The first development dataset uses the existing global training/validation cutoffs. Its task is **predict the category of a person's next Electronics review**, not their next purchase or next shopping session. The measured median training gap is 75.84 days; no session timeout is imposed.

Frozen preparation rules for this version:

1. Keep original user/product IDs, category strings and timestamps. Use both observed and predicted labels, retaining their source flags. Keep all ratings, repeated categories and rare labels; no five-review minimum or category-change-only filtering.
2. Group each user's reviews by timestamp. A group containing multiple reviews has no known internal order. That group, or a review without a known category, is a **history barrier**: it cannot be a target or input, and histories restart afterward. This avoids fabricating a transition by skipping an unobserved event.
3. A training target qualifies when it has at least one strictly earlier usable training review after the last barrier. Keep at most the latest **50** such reviews. A one-review user supplies no training example; a two-review user can supply one. Qualification uses past events only, not eventual lifetime history length.
4. For validation, first locate each person's **first raw review after the training cutoff**. Only then apply label, timestamp and history checks. It qualifies if its timestamp is unambiguous, its label is known and the person's training history ends in a usable event. Do not skip an invalid first event to choose a later target. There is at most one validation query per user; all its input events come from training.
5. Freeze candidate IDs from usable training-event labels, sorted by original category string; IDs start at one, with zero reserved for padding. Preserve a known validation label absent from these candidates with a null encoded target; future evaluation must count it as a miss, not drop it or add it to the vocabulary.
6. No final-test queries or category statistics are produced. The final evaluation cohort and continual-training window schedule remain a separate step. Validation labels must not later enter training or replay.

These rules define this development dataset; they are an adaptation of ADER to long-term category histories. They do not implement the authors' click-session preprocessing or override the fixed A/B sample contract. An observed-only sensitivity dataset would need its own explicit rebuild, because removing predicted events also changes usable histories.

```sh
PYTHONPATH=src uv run --no-project --python 3.11 --with duckdb==2.0.0 \
  python scripts/prepare_category_sequences.py \
  --audit-dir runs/electronics-category-audit-ID \
  --output runs/electronics-category-sequences-new
uv run --with duckdb==2.0.0 pytest -q tests/test_category_sequences.py
```

`category_sequences.py` reads the successful audit database read-only, records its hash and the source/audit/code provenance, and requires a fresh output directory. Run on a CPU allocation with four cores and 12 GB RAM; DuckDB has a 6 GB limit plus disk spill. No download, GPU, neural training or core dependency change occurs.

Outputs are `train_events.parquet`, `queries.parquet`, `vocabulary.parquet`, `manifest.json`, `report.json` and a preparation database. `train_events` contains chronological usable events with original identities and per-user positions. Each query stores its target and inclusive `history_start`/`history_end` positions. Reconstruct its category input by selecting **the same user's** training events in that range, ordered by position. No repeated materialization of every prefix is needed. Parquet physical row order is not a sequence order; always use the explicit positions. All event/query/database files contain private identifiers and remain ignored. Aggregate reports record exclusions, history lengths and file hashes.

The synthetic checks exercise repeated categories, unknown/tied barriers, strictly causal histories, the 50-event cap, first-raw-event selection, unseen candidate targets, duplicate mapping rejection, future-label isolation, full Parquet export and no-overwrite behavior. The earlier popularity audit evaluated all known validation events; those scores must be recomputed on this new one-query-per-person cohort before comparison with ADER.

## Category baselines on prepared queries — 25 October 2026

`models/category_baselines.py` owns ranking rules; `experiment/category_baselines.py` loads/joins frozen prepared files and `experiment/evaluation.py` computes Recall/MRR from rank counts. These lightweight category rankings are separate from the in-memory A/B runner and its model ABC.

- **Popularity:** count category labels among the 19,478,807 prepared training targets. Rank the complete frozen candidate set by descending count, breaking ties by original category string. Zero-count candidates stay at the end. No validation labels fit the counts.
- **Repeat last only:** output exactly the last training-history category. Report Recall@1/MRR@1 only; a single prediction is not padded into three recommendations.
- **Last + popularity:** move that last category to rank one, then retain popularity order for every other candidate, without duplicates. Report Recall@1/3 and MRR@1/3 alongside popularity.

All methods use the same prepared validation queries. Known targets outside the candidate set count as misses. Source-label and history-length slices reuse the same fitted rankings; observed-target slices are **not** an observed-only training/history sensitivity experiment. The rank-count evaluator's rank zero means an unranked target, which includes ordinary non-repeats for the one-prediction baseline.

```sh
PYTHONPATH=src uv run --no-project --python 3.11 --with duckdb==2.0.0 \
  python scripts/evaluate_category_baselines.py \
  --sequence-dir runs/category-sequences-ID \
  --output runs/category-baselines-new
uv run --with duckdb==2.0.0 pytest -q tests/test_category_baselines.py
```

The job requires a fresh output directory and verifies all three Parquet files against preparation hashes and sizes. It checks that validation contains one query per person and that joining the last history event preserves the count and context. Original query/candidate files are read-only. The output contains only an aggregate `report.json` and a `manifest.json` linking inputs, preparation provenance and executed model/evaluator code hashes. Use four CPU cores and 12 GB RAM; no GPU or neural fitting is involved. The complete ranking and per-slice denominators are recorded for review.

## Chronological category periods and comparison protocol — 25 October 2026

The user authorized preparing training periods and writing the next-review ADER comparison protocol. The timestamp-only 80/10/10 boundary stays fixed: training ends **29 June 2021**, validation ends **22 June 2022**, and the final 10% remains reserved. Eight periods divide only the 19,478,807 eligible training targets into nearly equal contiguous timestamp ranges; tied timestamps stay together. Validation determines no boundary and was not read to assign a period.

| Period | Eligible targets | Calendar range (UTC) | Categories in period targets | Cumulative categories seen in training targets or histories |
| --- | ---: | --- | ---: | ---: |
| 1 | 2,434,851 | 1999-11-17 – 2014-08-18 | 205 | 216 |
| 2 | 2,434,851 | 2014-08-18 – 2015-12-11 | 212 | 246 |
| 3 | 2,434,852 | 2015-12-11 – 2017-01-20 | 207 | 259 |
| 4 | 2,434,850 | 2017-01-20 – 2018-02-26 | 199 | 268 |
| 5 | 2,434,851 | 2018-02-26 – 2019-03-20 | 172 | 275 |
| 6 | 2,434,851 | 2019-03-20 – 2020-01-11 | 148 | 278 |
| 7 | 2,434,851 | 2020-01-11 – 2020-10-23 | 137 | 279 |
| 8 | 2,434,850 | 2020-10-23 – 2021-06-29 | 119 | 283 |

The uneven date spans reflect uneven review volume. “Categories in period targets” is that period alone; the cumulative column includes supervised targets or usable categories seen in training-history inputs by that boundary. The fixed vocabulary/candidate set has 283 labels: 271 eventually appear as supervised targets, while 12 occur only in context. All 283 are available in the prepared closed-catalogue task; category names are known in advance, and no test labels add categories. The staged classifier/output mask should expose the cumulative set at each boundary; context-only labels remain candidates but receive no positive target loss.

**Two-period resource and execution pilot.** Deterministically hash-sample **50,000 targets from each of periods 1 and 2**; this spreads the small sample across each period instead of selecting its earliest rows. This slice is for plumbing and resource measurement; do not report it as a representative score. Both methods use identical sampled examples and history ranges. Train period 1 once, then branch both period-2 methods from that same checkpoint, including its Adam state. Each method makes one pass through its 50,000 fresh targets in each period. In period 2, ADER-category adds up to **1,000** herded period-1 replay examples. Check teacher logits, old-category alignment, optimizer restoration, causal histories, and that validation is neither trained nor replayed. The control uses the same causal-attention architecture, fresh targets, minibatch order and dropout, without replay/distillation. The pilot records runtime and memory; it does not select hyperparameters.

**Full validation-selected experiment (proposed v1).** If the pilot passes its gates, run all eight periods. Use a category version of the ADER SASRec: causal attention, maximum history 50, 150 hidden units, two blocks, one head, dropout 0.3, Adam at 0.0005, batch size 256. Train one pass over each period in chronological order. Use identical fresh-target order for ADER and Dropout within each seed. Dropout trains on the fresh period only. ADER adds up to 30,000 herded examples from earlier training periods, stores the period-boundary teacher logits, and distills over categories already available before the current period. Its cross-entropy for fresh targets uses categories seen through the current period. Record actual replay counts and the adaptive coefficient `0.8 × sqrt((old seen-category count / current seen-category count) × (actual replay count / fresh-target count))`; “seen” is the cumulative union of eligible training input labels and target labels at that period boundary. Do not silently top up an underfilled buffer. Keep the full 283-category candidate mapping fixed and apply the stage's seen-category mask to model loss/distillation; all 283 categories are candidates at final evaluation. This fixed label catalogue and next-review task are adaptations, not the authors' original session protocol.

Run three predeclared seeds (**0, 1, 2**) for each method. Do not select intermediate checkpoints with the fixed validation set: its users' histories may include targets from later training periods. After period 8, evaluate once on that same untouched validation cohort, reporting event/user Recall@1, Recall@3 and MRR@3. Compare each seed with the popularity and last+popularity results already recorded; recompute them only if the candidate/query protocol changes. Validation may select among predeclared settings, but any setting change needs a new run identifier and recorded decision. Freeze the settings before final testing.

**Reserved final evaluation.** For each user, locate the first raw review after the 22 June 2022 validation cutoff. Retain a test query only when it is timestamp-unique, labelled, and the user had no raw review during validation and has a usable training history. Use training-only history; never use validation or another test event as context. Do not skip an invalid first event. Keep unknown targets outside the frozen candidates as misses. Report the eligible count and excluded reasons; this is a later-returning-reviewer cohort, not all Amazon users. Open this partition only after validation choices and the period-8 model are frozen. Test once per seed; do not refit on validation/test data. Report all three seeds, mean and standard deviation, plus the per-seed baseline comparisons. No test outcome chooses the model.

**Limits to the claim.** This is repeated category prediction over long-term review histories (median review gap 75.84 days), not next-click session recommendation. The validation/test reviewer cohorts condition on when a person returns, so later-period results are not a same-user old/new retention test. Report target-source slices: predicted metadata may be wrong, and these slices use histories that still contain predicted categories. A result can show performance on this protocol; it cannot by itself prove catastrophic forgetting or an ADER advantage on sessions.

The prepared period artifact contains every training target exactly once with the existing history endpoints. Its aggregate report lists exact cutoffs, UTC ranges, per-period and cumulative user/category counts, context-only candidate count and artifact hash. The final preparation job used four CPUs/12 GB and completed in 50 seconds with 7,696,580 KiB batch MaxRSS. Execute on the cluster with:

```sh
PYTHONPATH=src uv run --no-project --python 3.11 --with duckdb==2.0.0 \
  python scripts/prepare_category_periods.py \
  --sequence-dir runs/category-sequences-1010635 \
  --output runs/category-periods-new
uv run --with duckdb==2.0.0 pytest -q tests/test_category_periods.py
```

### Completed three-seed run and validation — 25 October 2026

The bounded pilot was followed by the complete proposed eight-period comparison. `scripts/run_ader_categories_full.py` trained seeds 0, 1 and 2; each seed trained period 1 once, then continued both methods from its shared period-1 model and Adam state. Dropout used fresh targets only. ADER used the herded replay selected at each period boundary, stored the current dropout-free teacher scores, and distillation over the previous category prefix. Both methods processed every target in all eight periods once (19,478,807 fresh targets and 76,096 updates per method/seed). The exact per-period outputs/checkpoints remain in ignored SoC `runs/amazon-ader-categories-seed0-1013143`, `...seed1-1013599` and `...seed2-1013600`; training jobs 1013143, 1013599 and 1013600 all passed.

Only after all six runs passed, `scripts/evaluate_ader_categories_validation.py` opened the frozen validation queries. Job **1014004** scored 1,547,555 users against the same 283 candidates for all six checkpoints, with original category ID as tie-break. The evaluator checked source hashes, query/history mappings and checkpoint provenance. It reports Recall@1, Recall@3 and MRR@3, overall and separately for observed/predicted targets. The full report and checkpoint hashes are in ignored SoC `runs/amazon-ader-validation-1014004/`; local review copies are `local/amazon-ader-validation-1014004.json` and `local/amazon-ader-validation-1014004-manifest.json`; no final-test rows or labels were read.

| Method | Recall@1 | Recall@3 | MRR@3 |
|---|---:|---:|---:|
| Dropout | 41.8348% (SD 0.2098 pp) | 66.3715% (SD 0.1627 pp) | 0.524956 (SD 0.001617) |
| ADER | 41.8506% (SD 0.0980 pp) | 66.4103% (SD 0.2123 pp) | 0.525209 (SD 0.001281) |

ADER's mean difference is +0.0158 percentage points at Recall@1, +0.0389 points at Recall@3 and +0.000253 MRR@3. Those gaps are small relative to seed variation; this run does not establish an ADER advantage. The baseline on these same queries was popularity Recall@3 60.9731% / MRR@3 0.498288, and last + popularity Recall@3 62.0441% / MRR@3 0.439964. The neural models improve top-three coverage over these baselines, while the marginal difference between ADER and dropout remains negligible.

This establishes completed full-data training and validation for this proposed category task. It is not an exact replication of ADER's click-session experiment, does not establish catastrophic forgetting or a causal replay benefit, and says nothing about held-out test performance. GPU dropout repeatability is not established. The next protocol step is to freeze decisions and construct the reserved test cohort without using validation/test events as context; do not access test until that decision is recorded.

### Two-period ADER category pilot — 25 October 2026

The user authorized the bounded training-only pilot above. `scripts/pilot_ader_categories.py` reuses the pinned ADER model/loss from `local/sources/ADER-348fc21` while reading only the prepared training-period and training-event Parquet files. It verifies their recorded hashes, samples 50,000 examples per period by a deterministic hash, and remaps the cumulative category IDs into a stable prefix so the 216 period-1 teacher logits align with the first 216 period-2 logits. It does not read validation or test rows or calculate recommendation metrics.

Cluster job **1012806** passed on one A100-40 GPU. It trained one pass per period with 256-row batches and the paper's 150 hidden units, two attention blocks, one head, 50-step history and 0.3 dropout. Both period-2 branches restored optimizer step 196 and ended at step 392. ADER herding selected and consumed **808** examples (the configured cap was 1,000); per-category quotas with no eligible sampled examples are left unfilled. The stored teacher shape was 808 × 216; the distillation coefficient was 0.09529. Dropout used no replay. The full job took 38 seconds in Slurm; the measured training script section took 24.14 seconds, process peak RSS was 1,597 MiB and sampled GPU memory peaked at 3,015 MiB. Source and data provenance are recorded in the cluster output `runs/ader-category-pilot-1012806/pilot.json`; an ignored local copy is `local/category-ader-pilot-1012806.json`.

This establishes that the bounded data/model/optimizer/replay path executes. It does **not** compare predictive performance: there was no validation selection or test scoring, it used one epoch and a 50,000-target sample per period, and GPU dropout is stochastic. It is not the full eight-period experiment or a paper replication. Submit the prepared launcher from the existing SoC project root with `sbatch scripts/pilot_ader_categories_slurm.sh`; it requests one A100-40, 64 GB host memory and a three-hour limit, and creates a fresh output directory keyed by the Slurm job ID.

## Outputs and failures

`run(...)` requires a **new** output directory. It writes `prepared.json` (supplied inputs), `manifest.json` (provenance/settings/source fingerprints) and `results.json` (validation trace and fixed-test records). `plot_results(...)` reads saved results; it never starts training. There is no durable model checkpoint or interrupted-run resume yet.

JSON publication refuses existing files/symlinks and racing writers using a same-filesystem hardlink. Hardlink support is required; atomic visibility is not power-loss durability or a multi-file transaction. A failed run may leave partial evidence: inspect it and use a new path, rather than deleting or overwriting it automatically.

The old trial CLI is retired. Its [historical runbook](https://github.com/Hong-yiii/cp4285_reccomender_systems/blob/8b9edd3/docs/IMPLEMENTATION.md) is not a current API. Before real data or expensive computation, agree the methods and verify the chosen backend's state/reload behavior; see [the guide's rules and limits](PACKAGES-AND-BUILDING.md#6-standing-rules-and-limits).
