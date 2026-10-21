# Project context

Snapshot: **21 October 2026**. Read this first, then [Packages and building rules](docs/PACKAGES-AND-BUILDING.md) and the [October meeting synthesis](docs/MEETING-2026-10-15.md).

**Current exploration:** the user requested ADER replication work on `feat/ader-exploration`, starting from the clean `feat/boiler-plate` checkout. Prior embedding/Amazon trials are saved separately on `archive/experimental-models-2026-10-21` at `69a163a`. [ADER reading and replication protocol](docs/ADER.md) records the paper, authors' source pinned to `348fc2128bceb6bb3f3d8dd25ec8bbbae831af94`, discrepancies and a DIGINETICA ADER-versus-Dropout first milestone. The user chose to begin with the authors' implementation and authorized local verification while investigating SoC cluster access. Thirteen TensorFlow synthetic checks now pass locally, including loss/gradient checks, optimizer restoration and two training windows through upstream `main.py` with a recorded NumPy compatibility fix. The standalone verifier lives in `scripts/`; TensorFlow stays in an ignored isolated environment. The user supplied a successful 13-check GPU report on TITAN V; the first real-data resource pilot also passed in the user-supplied report (220.32 seconds, 20.83 GiB child host RSS, 1.43 GiB sampled GPU memory). The real two-window continuation also passed in the user-supplied report (615.82 seconds, 44.85 GiB host RSS, 945 distillation updates). The user supplied a completed seed-0 comparison across all 16 windows: ADER Recall@20 0.499587 versus Dropout 0.488569, an advantage of 1.101741 percentage points. ADER remains outside the shared model packages. This user-directed exploration does not establish a finalized team protocol or change the shared holdout rules.

ADER uses next-window session recommendation with per-window validation and replay, rather than the existing fixed A/B retention runner. The released code splits validation after expanding prefixes and includes validation in future replay; these choices must not silently enter the shared benchmark. Full-softmax loss, old-vocabulary distillation, actual replay size, complete state restoration and full-catalogue Recall/MRR require explicit handling. Source copies and the paper remain under ignored `local/sources/`; public references and execution limits are in the protocol.

**Cluster handoff:** the user demonstrated SoC SSH/Slurm access and an allocated TITAN RTX (24 GB), driver 580.178.04, with system Python 3.12.3. A separate Python 3.11.17 / TensorFlow 2.15.1 environment installed successfully after moving temporary files to home storage and installation to a CPU allocation. After fixing our strict placement of the legacy int32 `global_step`, all model/data gates passed on the user's GPU run, including executed scoring placement. The separate upstream entry-point smoke then failed because TF 2.15 cannot enforce deterministic GPU scatter updates for legacy reference variables. The verifier now explicitly disables that enforcement only in its disposable GPU smoke copy and records the deviation; run-to-run GPU reproducibility is not established. Slurm visibility and original sources remain preserved. The subsequent user-provided cluster report passed all 13 checks in 34.14 seconds on TITAN V (12 GB). The report identifies the verifier hash and GPU scoring placement; this establishes compatibility execution, not paper results or GPU repeatability.

**Real-data evidence and authorized next run:** the user's two-window continuation report passed in 615.82 seconds, with 44.85 GiB peak child host RSS and 1.43 GiB sampled GPU memory. The second window restored selected step 1,960, used 27,696 replay examples with lambda 0.73569, and completed 945 KD updates. Its test Recall@20 was 0.5040 and MRR@20 0.1818; these are next-window measurements, not the 16-window paper mean. The first-window rerun differed slightly, so exact GPU repeatability is not established. These are user-provided reports, not remotely retrieved artifacts.

**Full comparison completed (user-provided evidence):** all 17 pinned DIGINETICA files are acquired/hash-verified, and all action counts match the paper (993,483 total). `scripts/profile_ader_window.py --windows 16 --method ader|dropout` retains original seed-0 hyperparameters and method-dependent RNG behavior, records exact evaluator metrics plus validation split fingerprints, and checks optimizer/replay handoffs. `scripts/run_ader_full_slurm.sh` requests one A100-40 allocation, 128 GiB RAM and 12 hours for ADER; documented overrides request 16 GiB and three hours for Dropout. Dense Python-list replay at later widths suggests about 101 GiB total host RSS by scaling the measured two-buffer peak; this is an estimate, not a bound. Both methods passed all 16 windows on tiny synthetic CPU inputs; 40 offline tests pass. `scripts/compare_ader_runs.py` checks provenance/completeness and reports unweighted 16-window means, percentage-point differences and differing validation splits. The user submitted both full runs and supplied the comparison output. ADER leads on all four metrics in every continuation window; validation splits differ in periods 2–16. The agent has not retrieved the full remote artifacts. See the result table in `docs/ADER.md`; repeat-seed and isolated-validation checks remain unperformed. Runs start fresh because upstream replay/RNG state is not resumable.

## Current direction and ownership

Study old-task recommendation quality during continued learning, potentially varying model capacity and incoming domain composition. The team wants to explore independently and converge on a common, inspectable benchmark. A decline, no change or improvement is a valid outcome; do not promise catastrophic forgetting or neural superiority.

**User-confirmed reset:** treat previous implementation as disposable trial work. Preserve scientific understanding and the principles of testing and writing two extensible packages. The user explicitly permits deleting old code. The package/building document is a **draft for human review**, not an approved final protocol.

The two responsibility packages are:

- `cp4285.experiment`: source preparation/contracts, schedules, model ABC, shared evaluation/selection, artifacts and graphs.
- `cp4285.models`: concrete neural/non-neural implementations and model-side ablations.

The notebook imports both and explains the experiment. The package implements computation and safeguards; the notebook exposes assumptions and evidence; the eventual report owns the argument. Scientifically different tasks must not be forced through a universal runner.

## Implemented now

- Standard-library-only Python 3.12 boilerplate, in the existing uv-managed distribution.
- `DataSource` and `Recommender` ABCs; typed event/query/example/prepared-experiment records.
- Fresh-target update blocks, constant/changing A/B schedules, no silent stream recycling.
- One concrete **single-target** runner: initial selection on validation NDCG, independent continuation branches, frozen control, fixed old/new task evaluation.
- A standalone shared evaluator in `experiment/evaluation.py`: NDCG/Hit/Recall with explicit cutoff, stable original-item-ID ties and fixed candidates.
- Create-only prepared snapshot, manifest and raw result JSON; SVG plots read saved results.
- `CountingRecommender` and `SyntheticSource` as a plumbing/isolation fixture, **not research models/data**.
- `notebooks/01_shared_experiment.ipynb`; offline contract checks include executing its cells in a fresh Python process.
- A sample-first development guide with complete maintained model/evaluator definitions, a nonempty-history/candidate-alignment/state check and notebook usage. `tests/test_documentation.py` verifies source/example agreement and executes all guide Python blocks. No new model or metric framework was added.

The old `common/`, `neural/`, `classical/`, CLI, pilot config and trial-only tests have been removed. The create-only JSON writer's behavior was retained inside `experiment/artifacts.py`. No old model adapter or compatibility layer is maintained. Historical source is recoverable at [pre-reset commit 8b9edd3](https://github.com/Hong-yiii/cp4285_reccomender_systems/tree/8b9edd3).

**Not implemented:** a real-data loader, real neural/classical model under the new interface, durable model checkpoint/reload/resume, large-data streaming/batched evaluation, multi-positive tasks, multi-seed aggregation or formal transfer metrics. In-memory full-state forks and one-query scoring are deliberate small-pilot limits. A passing synthetic check is not a real-data result or scientific validation.

## Scientific invariants

- Preserve real source identities and domain namespaces; never equate unrelated integer IDs or splice users' histories.
- Split before learning preprocessing. Holdouts enter neither training targets nor prefixes. Test events must not leak into validation-based selection.
- Freeze each task's queries, labels, candidates and ranking policy across checkpoints.
- Select/tune on development data, not retention curves. Keep independent state and declared optimizer/RNG policies across branches.
- Count supervised targets, not padded positions, users or files. Larger B share at fixed total targets means less A exposure; this alone does not isolate harmful B gradients.
- Aggregate score changes are outcomes under a protocol, not proof that a particular weight or memory was erased.
- Source events determine terminology: Amazon reviews are not purchase timestamps or ideal-recommendation ground truth.
- Artifacts are create-only and private by default. Acquisition/expensive computation needs an explicit decision.

## Decisions before a real experiment

1. Review the runnable package/building recipes and single-target baseline interface.
2. Choose model family, meaningful capacity variation, training/state policy and parameter accounting. October did not reaffirm SASRec or QR-Markov as mandatory models.
3. Choose dataset/category pair, observed feedback/filter, split/cohort, history horizon and cold-start/candidate policy.
4. Specify original-task versus past-task matrix, metric/cutoff, constant/changing/sequential schedules, mixture denominator, budget and seeds.
5. Implement one real loader and one defensible model, verify its lifecycle, then compare additional implementations on the same prepared contract.

No paper/model names or appointment deadlines should be inferred from the machine-generated meeting transcript. External scientific claims still require primary-source checks.

## Historical evidence and privacy

September design notes, source discussions, figures and `reference.html` preserve reasoning and dated findings. They do not describe the current runtime; [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) does. The Google Slides deck remains canonical and was not edited.

Before the reset, four pre-existing uncommitted files were copied byte-for-byte to ignored `local/trials-before-reset-2026-10-16/`, with checksums: the real-neural config, baseline code/test and result report. That report describes real A-only runs, but this session did not revalidate their datasets/checkpoints. Preserve this distinction from earlier notes claiming only synthetic evidence. Existing ignored datasets/runs/reports were not modified.

Private transcripts, chats, team/course PDFs, personal IDs and datasets never belong in public commits. [AI_LOG.md](docs/AI_LOG.md) records substantive assistance. The repo must run without `local/` or historical artifacts.
