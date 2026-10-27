# AI assistance log

## 25 October 2026 — Full Amazon category training and validation

**Request:** Continue the Amazon Electronics next-subcategory experiment on the SoC cluster and check the completed validation job.

**Assistance:** Ran the full eight-period comparison for seeds 0, 1 and 2, pairing ADER replay/distillation with the same dropout-only model. Kept validation unopened until all training jobs passed, then scored the fixed one-query-per-person validation set for all six checkpoints. TensorFlow 2.15 Grappler crashed during legacy GPU inference; reproduced it on a real validation history and restored checkpoint, then verified that disabling the meta optimizer lets batch sizes 1 and 256 score correctly. Applied that session setting to the evaluator. No test data was read. Updated the protocol/runbook, context and reviewer checkpoint with the completed result.

**Evidence:** Training jobs 1013143, 1013599 and 1013600 completed all 19,478,807 training examples in each method/seed (76,096 updates per method/seed). Validation job 1014004 completed in 5m11s and scored 1,547,555 queries against 283 candidates. Three-seed mean (sample SD): Dropout Recall@1 41.8348% (0.2098 pp), Recall@3 66.3715% (0.1627 pp), MRR@3 0.524956 (0.001617); ADER Recall@1 41.8506% (0.0980 pp), Recall@3 66.4103% (0.2123 pp), MRR@3 0.525209 (0.001281). The ADER gains were 0.0158 pp, 0.0389 pp and 0.000253 respectively. Outputs/checkpoints remain ignored on SoC; ignored local copies of the validation JSON and provenance manifest are under `local/`. The report records per-seed and observed/predicted-source metrics.

**Validation:** Ruff and Python compilation for the evaluator, shell syntax and `git diff --check` passed. The real-data evaluator verified hashes, label/history mappings and all checkpoint scoring; report status is `passed_validation_only`, with `test_read: false`. Direct GPU diagnostics passed on restored weights at batch sizes 1 and 256 with Grappler disabled.

**Limits:** Differences between methods are much smaller than seed variation and do not establish an ADER advantage. This is not ADER's original click-session task, does not establish catastrophic forgetting, and is not final-test performance. GPU dropout repeatability is unverified. The training profile template labels its general limitation as “Seed-0 only” for every seed; seed identity and actual counters remain in each manifest/profile, and the template wording is corrected in source without rewriting raw outputs. No test queries/labels were opened and no commit was made.

## 25 October 2026 — Two-period ADER category GPU pilot

**Request:** Run the bounded two-period Amazon next-subcategory pilot on the SoC cluster.

**Assistance:** Added a training-only runner and Slurm launcher that reuse the pinned ADER model/loss, validate prepared Parquet hashes, sample 50,000 queries per period, maintain a prefix-stable category mapping, and split period 2 into replay/KD and dropout-only branches from one period-1 model+Adam checkpoint. The pilot never opens validation/test data or calculates ranking metrics. Added local remapping/padding self-checks and sampled GPU-memory recording. Two startup errors were fixed before the successful run: the event hash's manifest location and TensorFlow's legacy int32 global-step placement. Both fixes are retained in source.

**Evidence:** SoC job 1012806 completed successfully on one A100-40 in 38 seconds; the training section took 24.14 seconds. It processed 50,000 targets per period in one epoch, with 196 optimizer updates per period. Both method branches restored global step 196 and ended at step 392. ADER selected and consumed 808 of the 1,000 replay cap; teacher logits were 808 × 216 and lambda was 0.09529. Process peak RSS was 1,597 MiB; sampled GPU memory peaked at 3,015 MiB. Report: ignored `local/category-ader-pilot-1012806.json`; job log: ignored `local/category-ader-pilot-1012806.log`.

**Validation:** Python compilation, pilot self-checks on a short cluster CPU allocation, Ruff, shell syntax and diff checks pass. The cluster report confirms prepared period hash, pinned ADER commit, cumulative active category counts (216/246), target counts (50,000 each), matching optimizer restoration and complete replay consumption. Validation/test were not read.

**Limits:** This is only an execution/resource check. There are no validation/test metrics, no full-data result, no eight-period run, and no run-to-run GPU repeatability claim. The per-category replay allocation left 192 of its 1,000 maximum unfilled; no top-up was applied. Failed pre-training attempts remain in the cluster Slurm logs for provenance. No commit/push was requested.

## 25 October 2026 — Chronological periods and Amazon ADER protocol

**Request:** Continue the plan by measuring the eligible training timeline, constructing chronological periods and specifying the comparison with a no-replay neural control.

**Assistance:** Profiled training-target counts by year and timestamp octile using the verified local training database only. Added a small period builder that validates exact target conservation, retains equal timestamps, preserves history references and exports a hash-checked Parquet artifact. Added cumulative category availability from training input histories and supervised targets to distinguish context-only classes. Wrote a bounded two-period resource pilot and a full ADER-category/dropout comparison protocol, including later-only validation selection and a first post-validation raw-review test cohort.

**Evidence:** Eight periods each contain 2,434,850–2,434,852 of 19,478,807 training targets. They run 1999-11-17 to 2021-06-29 and have very different calendar lengths (period 1 is about 15 years). Cumulative categories seen in history/target inputs are 216,246,259,268,275,278,279,283. 271 of the frozen 283 candidates ever have supervised training targets; 12 occur only as input context. The final aggregate period job 1012758 completed in 54 seconds, with 7,532,408 KiB batch MaxRSS. Validation/test rows never set boundaries; the test partition was not read.

**Validation:** All 58 repository tests pass with isolated DuckDB; Ruff, formatting and diff checks pass. Synthetic tests cover tie-preserving cuts, exact target conservation, unchanged histories, validation invariance, insufficient timestamp rejection, Parquet output and checksum verification. The executed v3 source hash matches its preserved cluster archive. No production comparison or neural run was launched.

**Limits:** The three-seed v1 settings are proposals, not team-approved choices. This study predicts long-term review categories, uses a closed candidate vocabulary, and does not reproduce session-based ADER. Validation runs only after the last period. The proposed test cohort excludes users with validation-period reviews, so results describe later returners and are not a same-user forgetting matrix. Predicted categories are not independently quality-checked. No final-test data was inspected/scored and no commit/push was requested.

## 25 October 2026 — Popularity and last-category baselines on frozen queries

**Request:** Recalculate popularity and “repeat the last category” using the exact prepared examples.

**Assistance:** Implemented simple category-ranking functions in the models package, a shared rank-count Recall/MRR evaluator, and a thin experiment/CLI workflow. Defined literal repeat-last as one prediction and separately disclosed a full-ranking last + popularity variant for top-three comparison. Popularity fits only prepared training targets; all methods share validation queries and candidates. Verified input Parquet sizes/hashes, preserved query files and recorded input/preparation/code provenance. Evaluated the authorized job directly on the SoC CPU cluster, then retrieved only aggregate reports.

**Results:** Across 1,547,555 validation queries and 283 candidates, popularity Recall@1/3 is 41.9594/60.9731%, MRR@3 0.498288. Repeat-last-only Recall@1 is 28.8119%. Last + popularity Recall@3 is 62.0441%, MRR@3 0.439964: slightly greater coverage but worse rank ordering. Saved source-label/history-length slices, their denominators and the full popularity ranking. These are the baselines for the prepared cohort; earlier all-event scores used different targets and fitting counts.

**Validation:** All 55 tests pass with isolated DuckDB; Ruff/format/diff checks pass. Tests cover lexical ties, zero-count candidates, last-category promotion without duplicates, missed/unseen targets, hand-calculated Recall/MRR, aggregate denominators, validation-independent rankings, missing history joins, full artifact execution, checksum rejection and overwrite protection. The cluster job completed in five seconds, exit zero; source code hashes match the executed manifest. Slurm reported 473,104 KiB batch MaxRSS, with short-run sampling limits.

**Limits:** No Amazon ADER model training, final-test scoring or significance test. The observed-target slice still uses the combined-label training fit and histories. The last + popularity method is explicitly an extension of literal repeat-last, not an unspecified top-three tie policy. Label quality and the continual-training schedule remain unresolved; private data remains ignored and no commit/push was requested.

## 25 October 2026 — Chronological category histories and eligibility

**Request:** Construct each person's chronological category history and finalize which histories qualify, following the review-frequency audit.

**Assistance:** Profiled training/development timestamp ties, repeated user/product pairs and history gaps on a CPU allocation. Implemented compact category-sequence preparation in the experiment package with a thin CLI, source/hash provenance and create-only exports. Defined this development version's eligibility: 1–50 earlier training events per target, repeated categories retained, unknown/tied groups break histories, and only the first raw validation event per returning reviewer may qualify. Kept both observed and predicted label-source flags. Did not alter the earlier audit artifacts or force the task into the fixed A/B runner.

**Evidence:** 775 tied training timestamp groups (1,619 events); no validation timestamp ties or repeated training user/product pairs. Median distinct-event training gap is 75.84 days, supporting long-term review-history terminology. The completed real preparation contains 19,478,807 training targets from 6,262,964 users, 1,547,555 validation targets/users and 283 training-derived candidate categories. Median context lengths are 3 and 2. Excluded first validation targets/users are recorded: 45 unknown labels, 1,585,323 without training history and 584 ending at a history barrier. No validation targets fall outside candidates. The CPU job completed in 3m51s, about 10.51 GiB batch MaxRSS; aggregate report and manifest were retrieved and the executed code hash matches local source.

**Validation:** All 51 tests pass with isolated DuckDB; standard tests pass 46 with five optional SQL checks skipped. Ruff, formatting and diff checks pass. Synthetic tests cover ordering, user isolation, repeated categories, tie/unknown barriers, one-prior-event eligibility, the 50-event cap, first-raw-target selection, unseen candidates, duplicate mappings, invariance to later held-out labels, artifact export and no-overwrite behavior.

**Limits:** This is a development preparation protocol for next-review category prediction, not shopping sessions or a finalized continual ADER experiment. The first-event validation cohort changes the metric denominator, so earlier all-review popularity scores cannot be reused. Final-test queries, continual windows, observed-only sequence sensitivity, label-quality checks and neural training remain unperformed. The metadata is retrospective. Private identities/data remain ignored; no commit/push was requested or performed.

## 24 October 2026 — Review frequencies and popularity baseline

**Request:** Start the next-subcategory plan with provisional K=3; measure categories among actual review events and a most-popular-categories baseline before committing the evaluation cutoff.

**Assistance:** Downloaded the official unsplit Electronics 0-core ratings CSV on a CPU allocation, checked size/gzip CRC and recorded SHA-256. Implemented a bounded DuckDB audit in the experiment package, with count-based Recall/MRR in the shared evaluator and a thin CLI. Verified identities and category joins; split by global timestamp quantiles before fitting popularity. Kept the latest 10% reserved; compared observed-only and observed-plus-predicted labels on validation, including a cohort with prior training reviews. Inputs/results remain ignored; dependencies are isolated from both the core package and ADER environment.

**Findings:** 43,365,426 events; 283 training labels, 126 validation labels, zero unmatched products in those splits. Combined-label popularity Recall@1/3/5/20 is 42.30/61.39/80.43/99.77%; observed-only Recall@3 is 66.07%. K=20 is almost saturated without history information. The two label policies have different top-three rankings, so their difference is not a label-quality estimate. The audit first rejected two source zero ratings; after inspecting aggregate invalid-field counts, retained and flagged them because ratings are not the prediction target. No review rows were silently discarded.

**Validation:** All 47 tests pass when the optional SQL dependency is included; Ruff and formatting pass. Checks cover count arithmetic, unseen targets, cold/warm cohorts, chronological boundaries, zero ratings, duplicate joins and invariance to reserved-test label changes. The successful real audit completed in 75 seconds on four CPU cores, about 4.10 GiB batch MaxRSS. Retrieved aggregate JSON and provenance; full source/download manifests remain on cluster. Updated context, source record, runbook and reviewer checkpoint.

**Limits:** No Amazon ADER training or final sequence dataset yet. The global audit split is provisional, and final test-category frequencies/scores were not inspected. Label accuracy, historical availability of metadata, equal-timestamp sequence ordering, repeated-category transitions and final history eligibility remain unresolved. The results are event-level popularity baselines, not personalized next-event results. No commit or push was requested/performed.

## 24 October 2026 — Meso-level target and Electronics label count

**Request:** The user clarified next-subcategory prediction as the intended task, requested an exact review-history download link, and asked for actual Electronics subcategory counts before deciding Recall@K.

**Assistance:** Verified the official unsplit 0-core ratings-only Electronics link from Amazon Reviews 2023 and its HTTP headers (969,766,940 compressed bytes). Read the already-downloaded Electronics Parquet in a small cluster CPU allocation using an isolated DuckDB dependency; no global or ADER package changes. Checked the file against its saved SHA-256, counted distinct products/labels by label source, and saved the complete aggregate label inventory under ignored `local/`.

**Findings:** 1,610,012 distinct products; 285 non-null labels; 1,481,570 observed, 128,432 predicted and 10 unknown products. The classifier's 20 classes are a separate count; 265 observed labels have fewer than 200 products. Top-20 labels cover 99.7632% of labelled products, which is not the same as review-event coverage. No label filtering, merging, Recall@K change or interaction download was performed.

**Limits:** Review histories have not been joined, so actual sequence coverage, label frequencies among review targets and metric usefulness are still unmeasured. No training or new Git commit was performed. Documentation now distinguishes the user-confirmed meso-level target from proposed input/splitting choices.

## 24 October 2026 — Amazon item-category dataset acquired on SoC

**Request:** The user announced newly available Codex SSH access and requested downloading `randomwish/amazon23-item-categories` onto the cluster, asking whether Hugging Face authentication was needed.

**Assistance:** Checked the public dataset API/card, existing SSH alias and available remote storage. Confirmed anonymous access to the ungated repository. Submitted a small CPU acquisition job that downloaded all 73 upstream files at revision `822d4745b11748ae5e24f6619e4c6c32a501f36b` into a fresh revision directory. No Hugging Face token, dependency installation or GPU was required. A one-off standard-library script and cluster connection notes remain ignored locally; no reusable downloader framework was added.

**Verification:** All 326,013,748 bytes matched upstream sizes; large-file contents matched the supplied LFS SHA-256 hashes and ordinary files matched Git blob hashes. The script also recorded SHA-256 for every downloaded file, checked completeness against the pinned repository inventory and wrote a success manifest. Transfer/verification took 90.15 seconds. The final manifest was retrieved into ignored local storage, and Slurm reported successful completion. The snapshot, source manifest and verification manifest remain on the cluster.

**Limits:** This is item/category metadata with observed, predicted and unknown labels, not review interactions. No Parquet row-level audit, label-quality validation, interaction join or training was performed. No dataset content or machine-specific account details were added to tracked files; authentication secrets were not accessed.

## 21 October 2026 — Completed seed-0 comparison and commit

**Request/evidence:** The user supplied the full comparison output and asked for a Git commit. Across 16 equally weighted windows, ADER Recall@20 was 0.4995867931 versus Dropout 0.4885693833; MRR@20 was 0.1725787836 versus 0.1677720335. All four metric differences favor ADER in each continuation window. Validation fingerprints differ after the initial window. This is pasted user-provided evidence, not an independently retrieved remote run.

**Assistance:** Recorded the result table, paper differences, provenance and limitations in the protocol, current context, runbook and dated reviewer checkpoint. Prepared a local commit of the accumulated cluster compatibility fixes, profiling/Slurm/comparison scripts, tests and documentation. Datasets, source caches, environments, transfer archives and run artifacts remain ignored.

**Validation and limits:** Repository tests, Ruff, shell syntax and Git whitespace checks were run for the commit. The earlier 16-window synthetic CPU checks passed for both methods. No additional real training or remote artifact retrieval was performed; multi-seed uncertainty, repeatability and session-isolated validation remain unchecked. No push was requested.

## 21 October 2026 — Full DIGINETICA comparison preparation

**Request/evidence:** After providing a passing real continuation profile (615.82 seconds, 44.85 GiB host RSS, 945 KD updates), the user authorized the full 16-window ADER-versus-Dropout milestone. The measured results are user-provided evidence, not remotely retrieved artifacts.

**Assistance:** Acquired the remaining 14 DIGINETICA period files through the connected GitHub tool, verified all 17 Git blob hashes and matched every paper action count (993,483 total). Extended the existing profiler to 1–16 windows and the released Dropout branch. Added exact evaluator metrics, unweighted means, split fingerprints and baseline no-replay checks, plus a provenance-checking comparison command. Preserved original hyperparameters, RNG behavior and known validation discrepancies. Estimated approximately 101 GiB later-window host RSS from the observed two-buffer footprint; prepared a 128 GiB/12-hour ADER job and documented 16 GiB/three-hour Dropout overrides. The public cluster guide was inaccessible through the web tool, so partition names/time limits use the user's supplied `sinfo`; current node feasibility/queue state remains unqueried.

**Validation:** 40 offline tests and Ruff pass. Each method completes all 16 tiny synthetic CPU windows with selected-checkpoint continuation and exact-metric collection. The ignored synthetic fixtures use a 32-item allocation and two epochs, explicitly distinct from the unchanged real settings. All real inputs pass preparation only. Shell syntax, source/input hashes and transfer archive integrity are checked for the handoff.

**Limits:** No real full training was run locally or submitted remotely by the agent. Memory is an extrapolation rather than a bound. Full-process replay/RNG resume, historical TF parity, repeatability and multi-seed significance remain unverified. Datasets, transfer archives and generated runs stay ignored; no shared-package holdout rules changed.

## 21 October 2026 — First real result and continuation preparation

**Request and evidence:** The user supplied a passing first-window profile and authorized a real continuation window. Reported elapsed time was 220.32 seconds, peak child host RSS 20.83 GiB and sampled GPU memory 1.43 GiB. Validation selected epoch 10; test Recall@20 was 0.4922 and MRR@20 0.1771. Replay retained 27,717 examples and took 121.09 seconds. This is user-provided one-window evidence, not a retrieved artifact or paper reproduction.

**Assistance:** Extended the existing profiler to one or two windows, retaining the one-window default and original hyperparameters. Added per-window artifact directories, replay/lambda/optimizer-step/KD-update records and report consistency checks. Acquired and hash-verified only the additional pinned period 2 file. Prepared a 64 GiB Slurm continuation job and ignored transfer archive. The two-window run starts from scratch because upstream checkpoints omit replay/RNG state; existing results remain untouched.

**Validation:** All 31 repository tests pass. A synthetic CPU execution of the instrumented upstream loop passes both windows, including selected-checkpoint restoration and completed KD updates. Real-data preparation and input hashes pass without training. Ruff, shell syntax and transfer-manifest checks are recorded with the handoff.

**Limits:** No real continuation training was run or submitted by the agent. Later-window peak memory, GPU repeatability and full-paper/baseline performance remain unverified. Data and generated artifacts remain ignored; shared-package holdout rules are unchanged.

## 21 October 2026 — Successful cluster verification and first real-window pilot

**Request:** After supplying a successful GPU verification report, authorize one real-data window with resource measurement before the full comparison.

**Evidence:** The user's report records all 13 gates passed in 34.14 seconds on TITAN V 12 GB, Python 3.11.17 / TensorFlow 2.15.1 / NumPy 1.26.4. GPU scoring placement and the two-window entry point passed; GPU deterministic-op enforcement was disabled and disclosed. This is user-provided evidence, not a remotely retrieved artifact or paper reproduction.

**Assistance:** Acquired only the two pinned DIGINETICA period files through GitHub, verified their Git blob hashes and audited counts. Prepared a one-window source-based runner and Slurm job with the original training settings, fresh run directories, source/data/environment manifests, exact patches, validation/split/checkpoint records, timings and process memory sampling. Reused the verifier's source-copy compatibility helper; added the documented first-zero-score checkpoint fix and profiling instrumentation. Prepared an ignored transfer bundle. No real-data training was run locally or remotely by the agent.

**Validation:** All 24 repository tests and Ruff checks pass. The 13-gate CPU verifier passes after helper extraction. A separate, explicitly synthetic CPU self-check exercises the instrumented entry point, subprocess monitoring and split/epoch/replay outputs. Both real inputs pass preparation without training. Shell syntax, source/data hashes, artifact hashes and no-overwrite boundaries were checked.

**Limits:** Full real-data runtime and peak memory remain unmeasured. The first window builds replay but has no distillation update. GPU memory sampling can miss peaks or be unavailable on MIG; host child RSS is not total job RAM. Upstream checkpoint files omit replay and RNG state, so the pilot is not resumable into the full experiment. Full 16-window performance, repeated GPU reproducibility and historical-runtime equivalence remain unverified. Data and generated outputs remain ignored.

## 21 October 2026 — Legacy GPU scatter determinism incompatibility

**Request:** Diagnose the upstream entry-point failure after the model/data gates passed on the cluster GPU.

**Assistance:** The supplied subprocess log identifies unsupported deterministic GPU scatter updates for legacy reference variables. The authors' entry point enables deterministic-op enforcement, whereas the already-passing model checks do not. The verifier now sets `TF_DETERMINISTIC_OPS=0` for the disposable GPU smoke copy and child environment, logs the deviation and records it in the patch and report. CPU smoke retains the authors' setting. No source-cache, model equation, dependency or GPU visibility change was made. This is an execution compatibility check, not an exact or repeatable GPU reproduction.

**Validation and limits:** All 18 repository tests and Ruff lint pass, including CPU/GPU launch tests checking the recorded patch, child environment, unchanged original source and preserved Slurm mapping. All 13 local CPU verification gates pass again; the updated GPU subprocess still requires a cluster rerun. Choosing a historical runtime or testing a resource-variable adaptation remains future replication work.

## 21 October 2026 — Correct GPU placement in the verifier

**Request:** Diagnose the user's first cluster verification failure at `global_step`.

**Assistance:** Identified our strict whole-graph GPU placement as the cause: the legacy int32 variable has only a CPU kernel. Enabled TensorFlow soft placement, matching the authors' entry point, and added an executed partition-graph check requiring the scoring operation on the requested device. Retained GPU presence checks, source hashes, Slurm visibility and no-overwrite behavior. Added regression cases rejecting CPU scoring and missing placement evidence in GPU mode. Updated context and replication notes; no dependencies or authors' source were changed.

**Validation and limits:** All 16 offline repository tests and Ruff lint pass. All 13 CPU verification gates pass, including executed scoring placement and two upstream training windows. No local NVIDIA GPU is available; the corrected cluster run is pending. The user's installation and failure logs are evidence of those stages, not a successful replication.

## 21 October 2026 — SoC GPU verification preparation

**Request:** Continue the cluster handoff after the user demonstrated Slurm access and a TITAN RTX allocation.

**Assistance:** Prepared a single-task Slurm smoke job and a small ignored transfer bundle containing the verifier and hash-checked authors' source, without datasets or environments. Removed the upstream entry point's GPU visibility override in its disposable copy so it preserves Slurm's assignment. Specified a separate Python 3.11 environment with TensorFlow 2.15.1's CUDA extra, checking official uv and TensorFlow package metadata. No remote connection or job submission was performed by the agent.

**Validation and limits:** All 12 local CPU synthetic checks passed again after the device handling change, along with 13 repository tests, Ruff lint, whitespace checks and Slurm script shell syntax. All 11 transferred files match the bundle manifest. These do not verify cluster GPU execution, Linux dependency installation or full-data resource use. Cluster diagnostic information comes from the user's terminal output.

## 21 October 2026 — Local TensorFlow verification and cluster handoff

**Request:** Verify code locally while the user investigates SoC cluster access, following their choice to begin replication with the authors' implementation.

**Assistance:** Created an isolated Python 3.11 / TensorFlow 2.15.1 / NumPy 1.26.4 compatibility environment. Added a portable source-hash-checked synthetic verifier, minimal dependency pins and three offline safety tests. Exercised the actual upstream model, independently checked losses/gradients and restored optimizer state. Confirmed that the unmodified entry point fails at second-window replay extraction on modern NumPy; recorded a list-comprehension fix in a disposable copy and completed two synthetic windows. CPU runs also record a device configuration edit. Updated the protocol/runbook, context, navigation and reviewer checkpoint. Applied the ponytail skill; no subagents or cluster actions were used.

**Validation:** All 12 TensorFlow checks passed in `runs/ader-verify-20261021-03/`, including two-window execution. Model dimensions match the paper while data and active vocabularies are tiny. All 13 repository tests, Ruff lint/format and whitespace checks passed. The shared environment was synchronized with the existing lockfile; TensorFlow dependencies remain separate. Source cache is unchanged; output contains the exact smoke-run patch, checkpoints, logs and environment versions.

**Limits:** No research dataset acquisition, published-result reproduction, full-data resource profile or GPU verification. The newer local runtime is not demonstrated equivalent to the authors' historical environment. Deterministic next-step restart is tested with dropout disabled; whole-process/sampler/RNG resume and other baselines remain unchecked. The upstream validation and early-stopping issues remain explicitly documented rather than silently changed.

## 21 October 2026 — ADER paper review and replication protocol

**Request:** Read the ADER paper, inspect the authors' code and define a replication protocol.

**Assistance:** Read the nine-page arXiv v1 paper and visually checked the algorithms, losses and result tables. Inspected all seven Python files, README, both environment descriptions and license in `doublemul/ADER` at commit `348fc2128bceb6bb3f3d8dd25ec8bbbae831af94`. Recorded the code/paper details, methodological differences and execution gaps in [ADER.md](ADER.md), with a proposed DIGINETICA ADER-versus-Dropout first milestone, full-table extensions, checks and artifact requirements. Updated context, source provenance, README navigation and the dated reviewer checkpoint. Used the ponytail and write-page skills, preserving repository documentation as the destination; no subagents were used.

**Validation:** All eleven cached upstream files match their Git blob hashes. Eight isolated synthetic checks passed for prefix validation overlap, unknown-item bridging, replay underfill, scalar-loss selection, metric cutoffs, missing preprocessing import, zero-score stopping and boolean parsing. These checks use extracted upstream definitions and existing local NumPy 2.5.4, without importing TensorFlow. Equations and table values were checked against rendered PDF pages.

**Limits:** No real dataset was downloaded, no dependency environment was installed, and no model training or full TensorFlow run occurred. The plan distinguishes an author-program reproduction from the changes required by the shared benchmark's holdout rules; it does not establish a team decision to change those rules or run expensive experiments. Performance targets are from the paper, and resource figures are calculated estimates. Source cache, PDF and audit harness remain ignored; the public documentation links to primary sources.

## 21 October 2026 — ADER exploration setup

**Request:** Create a new branch for exploring ADER and download the paper.

**Assistance:** Created `feat/ader-exploration` from the clean `feat/boiler-plate` checkout, checked the supplied arXiv reference and downloaded version 1 to ignored `local/sources/ader-2007.12000v1.pdf`. Recorded its public link and updated the context. Prior trials remain on the archive branch at `69a163a`.

**Limits:** No implementation, paper analysis, training or reproduction performed. The PDF stays local and is not intended for a public commit.

## 25 September 2026 — Project initialization

**Tool:** OpenAI Codex, including two internal agents for independent transcript synthesis and technical checking.

**User request:** Understand the CP4285 project from the supplied meeting notes and diagram, propose what to do first for user verification, and initialize a folder of Markdown notes.

**Inputs:** Supplied transcript and sketch; repository context; public course schedule, assignments, lecture slides and critique template; narrowly selected primary technical sources listed in SOURCES.md.

**Assistance provided:** Reconstructed the discussion and tentative roles; checked public assessment details; identified methodological ambiguities; proposed an experiment definition and critique preparation sequence; created this folder and updated the repository map.

**Human decisions pending:** Verify project interpretation, final scope, data/model choices, roles, deadline, and current team progress. None of the proposed experimental decisions is represented as team-approved.

**Artifacts and limits:** Markdown working notes and preserved source copies. No training, measured results, submitted work, external messages, or created slide deck.

This is a summary record, not a full prompt transcript. Retain the Codex conversation and add later prompts/outputs used by the team when preparing its collective AI-use documentation. This initialization does not establish how other teammates have used AI.

## 25 September 2026 — Neural workstream validation

**User clarification/request:** Hongyi owns the modern recommender contribution; validate the assumptions that modern models learn continually, that initial training is followed by learning during deployment, and that mixed-domain continued training is a valid experiment. Read and critique the supplied prior agent chat, then formalize the setup.

**Assistance provided:** Distinguished model architecture, changing inference inputs and actual parameter updates; checked SASRec, ADER, Monolith and CCTL primary sources; proposed a shared neural model and explicit training/evaluation controls. Preserved the supplied chat and added MODERN-RECOMMENDER.md. No new subagents were used for this follow-up.

**Status:** Role confirmed; architecture and implementation choices remain proposals. No experiments or results generated. Mathematical reasoning and proposed controls are agent analysis, not findings attributed to papers or the team.

## 27 September 2026 — SASRec paper walkthrough

**User request:** Explain the SASRec paper and how it fits the experiment.

**Assistance:** Read the original architecture, training and evaluation sections, visually inspected Figure 1, checked the authors' model/evaluation code and the official Week 05 SASRec material. Explained sequence representations versus persistent model weights, and separated the published model from the proposed continual-training protocol. Preserved the original paper locally (now `local/sources/sasrec-2018.pdf` after migration). No model selection was finalized and no experiment was run.

## 27 September 2026 — Continued-training specification

**User request:** Formalize adding continued training to the proposed SASRec experiment.

**Assistance:** Created CONTINUED-TRAINING.md with initial and continued phases, next-item objective, mixture accounting, optimizer-state policy, pseudocode, fixed test protocol and controls. Rechecked the original SASRec and ADER sources. The proposed defaults and equations are working design assistance; no dataset, architecture or numerical training budget has been finalized. No training code or empirical result was produced.

## 28 September 2026 — Scoped critique preparation

**User request:** Review the Group 3 PDF, break down requirements and scope preparation to Hongyi's modern-recommender work.

**Assistance:** Read all deck text and visually inspected relevant pages; compared with the preserved official template; expanded DEFENSE.md with page-specific gaps, neural preparation, controls, mixture definitions and rehearsal questions. Rechecked SASRec and ADER, including the draft's inaccurate RNN/YouTube reference.

**Limits:** PDF unchanged; no submission or experiment performed. No compute access, dataset approval or implementation progress inferred. Recommendations remain AI-assisted preparation for user/team verification.

## 28 September 2026 — Slide narrative and diagrams

**Request:** Provide diagrams and work out a logical slide flow for the neural contribution.

**Assistance:** Created SLIDE-FLOW.md with four editable Mermaid diagrams for SASRec inference, proposed shared-domain architecture, continued updates and comparison conditions. Added speaker notes and placement within the existing group sections. No slide deck was modified, no experiment was run and no synthetic performance results were drawn.

## 28 September 2026 — Dataset suitability review

**Request:** Evaluate Amazon Reviews 2023 Electronics/Movies & TV and DIGINETICA/YOOCHOOSE for the continued-training study, including a teammate's reference-paper interpretation.

**Assistance:** Checked official Amazon schema/processing statistics, ADER paper and repository, CCTL experiment setup and YOOCHOOSE authors' documentation. Created DATASET-ASSESSMENT.md with a provisional Amazon recommendation, limitations and feasibility criteria. Flagged the difference between ADER's separate temporal benchmarks and the proposed cross-dataset mixture, and the need to audit parameter sharing in the classical comparison.

**Limits:** Documentation assessment only. No full interaction files downloaded, overlap measured, dataset choice finalized, training run or performance claim made. The teammate's exact intended Amazon paper remains unidentified.

## 28 September 2026 — uv initialization and HTML update

**Request:** Initialize the existing repository for the proposed audit/target/design/A-only-pilot steps using uv, install dependencies and update the supplied HTML.

**Assistance:** Added pyproject/lockfile, a local Python 3.12 environment, configurable data audit/preparation commands, an official ID-file downloader, a SASRec-style A-only checkpoint/continuation pilot, meaningful tests and implementation documentation. Updated the existing HTML to six slides with Amazon dataset details, review-event semantics and explicit implementation gates. Preserved the existing Git repository.

**Validation:** Seven tests and lint/format checks passed. A synthetic end-to-end run reproduced evaluation after checkpoint reload with zero maximum score difference and verified a fixed frozen control. Browser review covers slide navigation, speaker notes and layout.

**Limits:** No Amazon interaction downloads or real-data metrics. Dataset selection, review target and time cutoffs remain provisional. A-only code does not yet implement B interfaces, mixed updates, classical models, exact SASRec paper replication or a final multi-seed study. The target defaults to all review events and can be changed through the rating filter.

## 28 September 2026 — Standalone repository and Pi handoff

**Tool:** OpenAI assistant in Pi, with independent read-only scout and reviewer agents.

**Request:** Make a standalone public `cp4285_reccomender_systems` repository, reorganize the existing project and context/slides, and redirect the course workspace to the new project home.

**Assistance:** Migrated code/tests and locked environment metadata; separated `configs/`, `docs/` and editable `slides/`; updated runtime/documentation paths; added README, CONTEXT and AGENTS entry points plus a layout regression check. Retained source evidence, course PDFs, draft team PDF and previous synthetic runs locally behind Git ignores. Removed private attachment paths and teammate identifiers from public summaries. Backed up the source project and redirected the course workspace without importing unrelated course Git history.

**Validation:** Locked environment installed; all 8 tests, Ruff lint/format and the synthetic CLI smoke check passed. A separate export containing only the 29 publishable files installed and passed all 8 tests without local/course sources; its synthetic demo passed on a direct retry after a combined validation task timed out. Both completed smoke checks reproduced checkpoint scores with zero maximum difference. Relative Markdown links, the old workspace redirect, source-copy integrity and Git ignores were checked. Gitleaks found no secrets in the public-only export; independent migration review reported no findings. Static diagnostics were not fully clean: stale environment/import findings and a pre-existing optional-checkpoint type warning remain, with details retained locally.

**Limits:** No new research decisions, data downloads, real-data results, final submission, license grant or deployment. Published notes remain AI-assisted working material, not team approval. Historical source files and runs retain their original contents; machine-specific migration details live only in `local/WORKSPACE.md`.

## 28 September 2026 — Implementation code review

**Tool:** Cursor Grok 4.7.

**Request:** Review the implemented audit, split, SASRec-style model and A-only pilot.

**Assistance:** Read the package, tests, pilot config and runbook. Compared the code with the stated holdout, checkpoint and evaluation contract. Did not change research decisions, download data, or treat synthetic checks as Amazon results.

**Findings to verify:** Right-padded histories give the prediction token a length-dependent position id. Validation and retention share one prefix, so checkpoint selection and the retention metric are two labels on one score vector. Out-of-vocabulary drops can put different users in those two splits. Download promotion checks the CSV header and does not read the gzip checksum. Prepare/audit outputs can be overwritten; DuckDB spill uses a working-directory-relative path.

**Validation:** `uv run pytest -q` passed, 8 tests. No code changes.

## 28 September 2026 — Original SASRec architecture

**Tool:** OpenAI assistant in Pi, with read-only upstream research and independent code review.

**User decision:** Use the original [kang205/SASRec](https://github.com/kang205/SASRec) immediately rather than the generic Transformer pilot.

**Assistance:** Inspected and pinned upstream commit `e3738967fddab206d6eeb4fda433e7a7034dd8b1`. Ported its architecture from Python 2 / TensorFlow 1.12 to the existing PyTorch runtime, preserving normalized residuals, raw K/V, masking, hidden-width feedforward, tied scoring and sampled loss. Adopted fixed left-padding, original architecture defaults, sequence-wise initial supervision with user sampling and Adam beta2=0.98. Continuation still masks historical targets to avoid implicit replay. Added source/license notices, checkpoint provenance and explicit protocol differences in SASREC.md. The earlier Cursor review entry above is preserved unchanged.

**Validation:** All 11 tests, Ruff lint and formatting passed. Tests include a separate NumPy translation of upstream forward/loss equations (one and two heads), causal isolation, finite gradients, training masks, holdout exclusion and checkpoint lifecycle. The full configured model completed a synthetic demo with zero reload score difference and unchanged frozen control. Independent source-fidelity review reported no findings. Source/wheel builds passed, and the wheel was checked for the Apache license and attribution notice.

**Limits:** This is a PyTorch port, not execution of the legacy TensorFlow code. No TensorFlow numerical comparison, original-paper benchmark reproduction or real Amazon experiment was performed. The model is selected; the two-domain design, final dataset and training budget are not thereby approved. Earlier generic-model checkpoints are incompatible. Existing data-audit/download concerns recorded in the Cursor review are outside this model change.

## 28 September 2026 — Classical comparator

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Explore non-deep-learning (matrix-factorisation / QR-based) recommenders for the project, build a baseline to iterate on, implement a classical method from the literature (SLIST), and integrate it into this repository.

**Assistance:** Downloaded the official 5-core Electronics and Movies & TV ID files (plus the unfiltered Movies & TV file) and measured A/B overlap. Implemented `src/cp4285/classical/`: DuckDB loader with shared original user IDs, leave-last-out split, disjoint and shared-user contamination designs, popularity, raw and QR-factorised Markov chains, user-item PureSVD with a QR-based incremental variant, and SLIST ported from the authors' code. Added `cp4285 classical` subcommands, eight tests, SciPy as a dependency and CLASSICAL.md with the first real-data sweeps.

**Validation:** 16 tests plus Ruff lint and format passed. CLI smoke test on real data. Full-data sweeps ran on a laptop (validation split, one seed).

**Limits:** Exploratory single-seed numbers with lightly chosen hyperparameters (SLIST α and age decay compared on validation). The protocol is leave-last-out, not the pilot's cutoff design, so neural and classical numbers are not yet comparable. No Diginetica run, no team approval of model choice implied.

## 28 September 2026 — SASRec PR and workstream integration

**Tool:** OpenAI assistant in Pi, with independent read-only review.

**Request:** PR the current work into main, then continue to the next checkpoint.

**Assistance:** Reviewed and published the original-SASRec changes in PR #2. Main advanced with the teammate's classical comparator (PR #1); merged that work without dropping its CLI, dependency, tests or AI log. Reconciled status notes to distinguish the synthetic neural cutoff pilot from contributor-reported real-data classical leave-last-out sweeps. Preserved the Cursor review and existing slide formatting.

**Validation:** The combined suite passed all 21 tests, Ruff lint/format, CLI doctor/classical help and source/wheel builds. Both SASRec-only and combined public exports passed Gitleaks. Integration review identified an unguarded classical config preparse and unqualified result provenance; fixed both, first reproducing missing/malformed-config failures with two regression cases. Follow-up review confirmed the working-tree fixes. The classical real-data sweeps were not rerun during integration.

**Limits:** No data downloads, common-protocol model comparison, protocol approval or course submission. The next implementation checkpoint is download integrity and non-overwriting artifact persistence; research decisions remain explicit gates.

## 28 September 2026 — Real-run artifact safety checkpoint

**Tool:** OpenAI assistant in Pi, with read-only review.

**Request:** After merging the existing work, continue to the next checkpoint.

**Assistance:** Merged SASRec PR #2 into main alongside the classical workstream, then implemented full gzip/header/trailer verification before download publication. Each download uses a unique temporary file; existing data/manifests are preserved. Reused the shared JSON writer for audits, prepared snapshots, manifests, neural metrics and classical reports, publishing complete finite JSON without overwriting even a racing destination. Kept the current research split and training protocol unchanged. Made single-row SQL aggregate results explicitly non-optional for static checks without changing the queries.

**Validation:** First reproduced 10 failures in 12 offline safety cases (existing-artifact replacement, corrupt/truncated gzip promotion, failure cleanup, orphan manifest handling and CLI behavior). After fixing the shared boundaries, all 33 project tests passed along with Ruff lint/format and fresh primary LSP checks for the changed Python files. Source/wheel builds and a new synthetic CLI demo passed; reload score difference was 0.0. Independent read-only review found no issues. Tests mock HTTP locally and do not download datasets.

**Limits:** Atomic file visibility is not power-loss durability or a two-file transaction. A manifest write failure preserves validated data for inspection; hardlink support is required. Killed processes may leave unpromoted temporary files. No acquisition was approved, no new real-data result was generated, and validation/retention cohort semantics plus classical/neural protocol alignment still require a team decision.

## 28 September 2026 — Shared utilities housekeeping

**Tool:** OpenAI assistant in Pi, with read-only scouting.

**Request:** Pull the latest code, create a folder with `utils.py` for common functions, and do housekeeping.

**Assistance:** Fetched origin and checked the current safety branch with a fast-forward-only pull; it was already up to date. Started `refactor/common-utils` from that checkpoint without merging its still-open PR #3. Moved `save_json`, `sha256` and `millis` into the standard-library-only `src/cp4285/common/utils.py`, removed their duplicate definitions from ingestion, and updated every direct caller/test import. Updated the layout/runbook/context and retained existing AI-log entries. Kept workstream-specific data loaders and research protocols separate.

**Validation:** All 34 tests passed, including a new multi-chunk hash/UTC-date/error-propagation check and the existing download/JSON artifact-safety regressions. Ruff lint/format, eight-file primary LSP diagnostics and source/wheel builds passed. AST comparison confirmed all three helper bodies are unchanged from the safety checkpoint; an isolated import confirmed utilities do not load database, HTTP, numerical or model dependencies.

**Limits:** This is code organisation, not a new experiment or protocol decision. Existing no-overwrite semantics and error propagation are preserved. No dataset was downloaded and no real-data result was produced.

## 28 September 2026 — Public reviewer reference and utilities PR

**Tool:** OpenAI assistant in Pi, with independent read-only publication review.

**Request:** PR the utilities housekeeping together with the existing reference HTML, retaining its structure as a top-level reviewer reference.

**Assistance:** Promoted a copy of the teaching walkthrough to `reference.html`. Preserved the complete original narrative, integration/safety appendix, CSS and five diagrams; removed personal checkout/disposable-page framing and added current-status navigation plus a dated utilities/reviewer checkpoint. Clearly labeled historical Git states, test counts and unpublished local run evidence. Linked the page from README, context and the agent guide. Prepared a stacked review against `fix/real-run-artifact-safety` because PR #3 remains open; documented merging that prerequisite, rebasing and retargeting before merging this change.

**Validation:** All 35 tests passed in 3.36s, including one new offline reference check for unique/resolving anchors, repository-local links, self-containment and publication metadata. Ruff lint/format, nine-file primary diagnostics, source/wheel builds and public-only Gitleaks scanning passed. The source distribution includes the reference and the wheel includes the utilities. Compared the copied narrative/appendix and CSS against the original; confirmed the original temporary file is unchanged. Opened the file directly in a browser at 1440px and 390px, verified working checkpoint navigation, zero loaded external assets and no page-level horizontal overflow. Independent review found no remaining publication issues.

**Limits:** No PR merge, dataset acquisition, new research result, deployment or course submission is authorized by this reference. Historical local run artifacts are not included; source/runbook remain authoritative, and future checkpoints should be appended rather than silently replacing evidence.

## 28 September 2026 — SASRec architecture walkthrough follow-up

**Tool:** OpenAI assistant in Pi, with independent read-only code fact-checking and review.

**Request:** Stack another commit on PR #4 explaining SASRec in the detailed, concrete style of the supplied conceptual questions.

**Assistance:** Appended section 17 to `reference.html`: raw reviews versus model features, frozen catalogue/OOV rules, a consistent five-position reviewer example, tensor shapes, item/position embeddings, original attention/masks/residuals/feedforward, sampled logistic loss, phase-specific labels/negatives, full-catalogue retention arithmetic and changing parameters versus fixed evaluation. Added three captioned diagrams and source links. Clarified held-out events versus reusable product IDs, catalogue allocation versus actual training exposure, independent logistic loss versus pairwise ranking, and gradient flow through history despite masked historical target losses. Preserved the pre-existing HTML formatting, CSS and earlier sections 01–16. Added a link from `docs/SASREC.md`; used the questions as inspiration rather than publishing the supplied conversation.

**Validation:** Added one executable synthetic worked-example regression in `tests/test_sasrec.py`, covering initial/continuation tensors, full-score dot products, sampled loss, a context-only item's gradient and the rank-4/tie arithmetic. The full suite passed all 36 tests in 3.54s; Ruff lint/format, primary test-file diagnostics, source/wheel builds and public-only Gitleaks checks passed. Byte comparison confirmed earlier sections and formatted CSS remain intact. Browser checks at 1440px/390px confirmed eight captioned diagrams, resolving anchors, no external assets and no page-level horizontal overflow, including the expanded architecture details. Independent source review found no inaccuracies requiring changes.

**Limits:** Production model/data/training code is unchanged. Numerical examples are invented software checks, not measured Amazon performance or TensorFlow parity. The fixed-anchor interpretation, OOV coverage and classical/neural comparability remain research-contract questions. No data acquisition, deployment, submission or PR merge.

## 28 September 2026 — Neural package and shared data handling

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Move the deep-learning approach into one folder, move the data handling methods into `common`, rearrange whatever else is needed and open a PR.

**Assistance:** Created `src/cp4285/neural/` holding `model.py` (SASRec port), `pilot.py` (training/evaluation lifecycle) and a new `data.py` with the unchanged cutoff-split `prepare`. Moved protocol-independent download, validated loading, DuckDB connection settings and the A/B audit to `src/cp4285/common/data.py`. Kept `prepare` in `neural/` rather than `common/`, because placing it in common would imply the classical workstream had adopted the cutoff protocol; left the classical loader and split untouched. Updated CLI/test imports, NOTICE attribution paths, README, runbook, SASREC, context and the agent guide. In `reference.html`, repointed section 17's source links, refreshed the stale top status (PRs #1–#4 are merged) and appended section 18 describing the layout; earlier sections' dated file/line references were left as historical evidence.

**Validation:** AST comparison confirmed all 18 moved top-level definitions are unchanged. All 36 tests passed; Ruff lint/format passed; a fresh synthetic `cp4285 demo` completed with reload difference 0.0; the built wheel contains the `common`, `neural` and `classical` packages.

**Limits:** Import paths changed (`cp4285.data`, `cp4285.model`, `cp4285.pilot` no longer exist); CLI commands, config, checkpoints and behavior did not. No dataset was downloaded and no real-data result was produced. Whether classical should adopt the common validated loader or the cutoff split remains a team decision.

## 28 September 2026 — Team deck pointer and slide principles

**Tool:** Claude Code (Claude Opus 5.5), with the in-app browser for read-only viewing.

**Request:** Replace the HTML slides with a pointer to the team's Google Slides deck plus principles for making slides, and integrate the repository's work with that deck. Follow-ups: write the principles for a class audience with no assumed knowledge of SASRec, data techniques, the dataset or the papers, while keeping rigour; add a file of slide issues; open and merge a PR.

**Assistance:** Read the live deck through its public text export and preview, without editing it. Removed `slides/index.html` and added `slides/README.md`: the deck link, the intended audience and a "classmate test", 19 principles (explain before use, rigour in plain words, readable visuals, one deck with four authors), a plain-wording glossary of terms the deck uses, and a pre-export checklist. Added `slides/ISSUES.md`: deck-wide and slide-by-slide problems ranked must/should, covering contradictions, stale status, reviews described as purchases, unexplained jargon and placeholder references. Added a dated section to `docs/SLIDE-FLOW.md` placing two paste-ready neural slides written for the same audience: the empty data slide introduces the dataset, the outdated technical-plan slide becomes N1, and hiding the version history makes room for N2. Updated the implementation-status table there, and the pointers in README, CONTEXT, AGENTS, PROJECT, IMPLEMENTATION and DEFENSE. Appended section 19 to `reference.html` and updated its status notice.

**Validation:** All 36 tests pass, including the reference link/self-containment check; Ruff passes. Slide numbers and content were checked against the export and screenshots of slides 1–5 and 7–13.

**Limits:** No change to code, data, training or evaluation. The Google deck was not edited, and its slide numbers will drift as the team edits it. The neural slides describe built software and a planned experiment; there is still no neural Amazon result. The mixture denominator and classical/neural split alignment remain team decisions.

## 29 September 2026 — Slide figures

**Tool:** Claude Code (Claude Opus 5.5), with headless Chrome to render figures and the in-app browser for read-only viewing of the team deck.

**Request:** Diagrams for the deck, built one slide at a time: the motivation (train once versus keep updating), the proposed-method pipeline, the classical models, two slides each on how the QR model and SASRec handle the data, fit and predict (with tensor shapes and outputs over product IDs), how mixing in Movies & TV works by tensor shape, and next item (Hit@10) versus next items (Recall@10). Then open and merge a PR with anything else useful.

**Assistance:** Read the classical and neural code, configs, design docs, issue #8 and the `classical-study` branch, and drew each figure from them rather than from the deck text. Added nine HTML figures with PNG exports under `slides/figures/`, plus an index and a pointer from `slides/README.md`. Matched the deck's colours by reading its public preview; the deck itself was not edited. Recorded Hongyi's statement that the full classical study keeps only the QR model in `CONTEXT.md`, noting that `MarkovQRSVD` still builds and stores the full transition table, so dropping the Markov baseline saves no training cost. Updated the figures to the issue #8 metric decision after it was made.

**Findings raised with the user:** on `main`, classical `sweep --split test` scores from the latest training review, so it predicts two steps ahead (fixed on the `classical-study` branch); SASRec's output is a score per product, not a probability, and its training loss is a yes/no loss on two products per position; for mixing, Electronics predictions can change only through the shared weights (about 28,500 numbers) as long as random products for Movies & TV examples come from Movies & TV, which today's sampler does not guarantee.

**Limits:** Figures only; no code, data, training or evaluation changed. Example products, ranks and bar heights are illustrative. The mixing figure describes a planned design. Figures that cite the QR test refit, Hit@10 checkpoint selection or the next-items split depend on PR #9 as it stood on 28 September.

## 16 October 2026 — Research reset and two-package boilerplate

**Tool:** OpenAI assistant in Pi, with a fresh-context read-only researcher for meeting distillation and an independent read-only reviewer.

**User request and decisions:** Treat existing code as disposable trial work; preserve experimental understanding and build two extensible packages with shared interfaces and a notebook. Synthesize the supplied reference research-system writeup into a package/building north star for human review. The user subsequently explicitly authorized deleting old code, prioritizing package and testing principles over retaining trials.

**Inputs and assistance:** Read the supplied October 15 transcript/sketch and historical context, including an uncommitted real-neural report. Created `docs/MEETING-2026-10-15.md` and the review draft `docs/PACKAGES-AND-BUILDING.md`, separating meeting support, proposals, actual implementation and unresolved methods choices. Implemented `cp4285.experiment` and `cp4285.models`: data/model ABCs, event identities and holdout checks, fresh update schedules, shared single-target metrics, validation-selected independent branches/frozen control, create-only artifacts and saved-result SVG plots. Added a clearly synthetic counting isolation control and an executable notebook. No research model was selected or ported.

**Retirement and preservation:** Removed the old neural/classical/common pipelines, CLI, pilot config and trial-only tests, and removed unused runtime dependencies. Retained the proven JSON-publication behavior inside the experiment package. Preserved four pre-existing uncommitted config/code/test/report files byte-for-byte under ignored `local/trials-before-reset-2026-10-16/`, with a checksum manifest. Existing ignored datasets/runs/reports were not changed. Historical committed code remains recoverable from pre-reset commit `8b9edd3`; research notes remain with historical labels. Refreshed the root guidance/runbook and appended a reference-HTML checkpoint without replacing its old narrative.

**Validation:** Locked environment sync, all 9 new offline tests, Ruff lint/format, primary Python diagnostics and source/wheel builds passed. The tests exercise the notebook cells in a fresh process, branch independence, deterministic fixed evaluation, score/ranking behavior, mixture accounting/exhaustion, leakage/identity/order rejection and artifact overwrite/race protection. Independent review identified test-target leakage into validation prefixes and an unchosen global chronological constraint; both were corrected with regression checks. This is software/contract validation, not real-data reproduction or scientific validation.

**Limits:** The north star and single-target interface await human review. Real-data loaders/models, durable model checkpoints/resume, large-data execution and multi-seed analysis are not implemented. The full prepared input and learner forks are in memory. No acquisition, real training, new empirical claim, submission, slide edit or publication occurred. The uncommitted historical neural report was read but its datasets/checkpoints were not revalidated. Raw private transcript content and participant details were not published.

## 16 October 2026 — Executable development samples

**Tool:** OpenAI assistant in Pi, with read-only scouting and independent review.

**Request:** Provide simple model and evaluation sample code in documentation, showing how teammates develop with the packages; aggressively remove unnecessary material.

**Assistance:** Replaced the long conceptual building guide with code-first recipes: the complete existing counting model, complete shared evaluator, a small contract check and notebook usage. Reused the model rather than adding another toy implementation or framework. Moved the evaluator unchanged into `experiment/evaluation.py` and retained the public import. Reduced duplicated runbook material and refreshed contributor/context pointers. Added a test that checks the documented model/evaluator syntax trees against maintained source and executes every Python block in a fresh process. The sample check covers nonempty history input, distinct scores under candidate reordering, ranking arithmetic and model-state/branch preservation; the counting model still deliberately ignores history.

**Validation:** All 10 tests, Ruff lint/format, primary changed-Python diagnostics, locked environment sync and source/wheel builds passed. Documentation examples are executable software checks, not scientific reproduction.

**Limits:** No research model, data, metric or selection policy was added or changed. MRR is described only as a possible extension recipe, not an implemented result. The existing backend-specific state/exposure and real-data protocol decisions remain open. Historical/private evidence and pre-existing working-tree changes were preserved.
