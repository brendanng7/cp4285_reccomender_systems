# ADER reading and replication protocol

**Draft for review, 21 October 2026.** ADER studies continual **session-based next-item recommendation**: learn from one time window, then predict clicks in the next. A useful first replication is DIGINETICA, comparing ADER with the same SASRec model using dropout but no replay. Complete all 16 evaluations before extending to YOOCHOOSE or the full baseline and ablation tables.

There are two distinct replication targets. **Reproducing the released program** preserves its data and validation choices, including the problems below. **A replication suitable for our shared benchmark** preserves holdout isolation and original identities, which changes those choices and therefore cannot claim an exact reproduction of the published numbers. This protocol specifies both boundaries; it does not adopt upstream validation leakage as a project default. ADER is not integrated into the shared model packages. The user supplied successful initial and continuation profiles; the user has now supplied the completed full comparison. See the latest execution status below.

**Execution update, 21 October 2026:** the user chose to start with the authors' implementation and authorized local verification while arranging SoC cluster access. The [standalone verifier](../scripts/verify_ader.py) now runs their TensorFlow model and a two-window synthetic training smoke check. See [local verification and cluster handoff](#local-verification-and-cluster-handoff). This is not a model integrated into the shared packages or a reproduced paper result.

**Latest, 21 October 2026:** the user supplied the completed seed-0 comparison across all 16 test windows. ADER achieved Recall@20 **49.9587%** and MRR@20 **17.2579%**, versus Dropout **48.8569%** and **16.7772%**. This closely matches the paper's scores and relative advantage under the released protocol. These are user-provided comparison results, not independently retrieved remote artifacts. Validation splits differ after the initial window; single-seed uncertainty and modern TensorFlow compatibility limits remain. See [completed seed-0 results](#completed-seed-0-results).

## Sources and scope

- Paper: Mi, Lin and Faltings, [ADER, arXiv:2007.12000v1](https://arxiv.org/pdf/2007.12000v1). Page numbers below refer to the **nine-page arXiv PDF**, locally `local/sources/ader-2007.12000v1.pdf`, SHA-256 `e21c50751df1996710ac4c89bde3debdbd170e34dd2da5f8e021bd6c20eab610`. Read the full paper; visually checked the algorithms, equations and result tables on pp. 4–7.
- Authors' repository: [doublemul/ADER, commit 348fc2128bceb6bb3f3d8dd25ec8bbbae831af94](https://github.com/doublemul/ADER/tree/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94), the retrieved default-branch head, dated 27 October 2021. Inspected README, both environment files, LICENSE and all seven Python files: `ADER.py`, `EWC.py`, `main.py`, `modules.py`, `util.py`, `data/preprocessing.py`, `data/util.py`.
- An ignored source-only cache is in `local/sources/ADER-348fc21/`. `SOURCE.json` records the commit and Git blob hashes; all eleven fetched files match those hashes. The public protocol remains usable without this cache. At the initial review, data contents were not downloaded. Following explicit authorization, all 17 DIGINETICA files are now downloaded and verified; YOOCHOOSE data remains unacquired.
- The source repository supplies an [MIT license][license]. Retain its notice and relevant inherited attribution when adapting code. The arXiv front page has a placeholder ACM DOI; the authors' README links the publication at [10.1145/3383313.3412218](https://doi.org/10.1145/3383313.3412218).

## What the method does

For a prefix such as `[item A, item B]`, the label is the next item, not a user rating or a domain identity. SASRec produces a representation of the prefix and a score for each currently available item. ADER changes how this model learns across time; it is not a separate replacement for SASRec. The paper uses one dataset at a time, not an Electronics-to-Movies domain switch. [Paper, §§3.1–3.2, p. 3.][paper]

Use window indices `t = 0,…,16`. Train the initial model on window 0 and evaluate it on window 1. Then update on window 1 and evaluate on window 2, continuing through training window 15 and test window 16. There are **16 train/evaluate pairs**, consisting of initial training plus 15 subsequent updates. No model is updated on window 16 in this experiment. A next-window test may become available training data in the following cycle, after its prediction has been recorded; this is a prequential protocol, not a permanently held-out retention test. [Paper, Fig. 1, p. 3 and Table 1, p. 5; main loop][main].

At each later update, ADER combines:

1. Cross-entropy on the current window's training examples, over all items seen through this window.
2. Distillation on stored old prefixes. The old model's softmax distribution is the target, and the new model's softmax is restricted to the **old item vocabulary** before calculating this loss.
3. A weight that grows when the old-item fraction or replay-to-new-data ratio grows:

\[
L = L_{CE}(D_t) + \lambda_t L_{KD}(E_{t-1}),\qquad
\lambda_t = \lambda_{base}
\sqrt{\frac{|I_{t-1}|}{|I_t|}\frac{|E_{t-1}|}{|D_t|}}.
\]

The two losses are separate sample means. The square root covers **both ratios**. Temperature is 1 in the released code. Standard ADER does not add hard-label cross-entropy on replay examples; that is the ER ablation. The first window uses only current-data cross-entropy. [Paper, Eqs. 2–4, pp. 4–5; `ADER.py`, lines 87–138][model].

After restoring the best validation checkpoint, select the next replay memory from current examples and the previous memory. Group by target item; popular targets receive larger quotas. Herding chooses prefixes whose representations approximate their target group's mean. Compute new teacher logits with this selected model, with dropout disabled, including for old examples retained again. Thus teacher outputs are refreshed each cycle. [Paper, Algorithms 1–2, p. 4; exemplar selection][memory].

## Released experiment specification

### Data and examples

Use the [period files at the pinned commit][data] for an author-artifact reproduction. Verify file hashes before running and compare every window's action count with Table 1. These are counts of click rows, **not** counts of expanded training targets:

| Dataset | Window 0 through window 16 action counts | Total |
| --- | --- | --- |
| DIGINETICA | 70739, 37586, 31089, 32687, 30419, 57913, 52225, 57100, 69042, 82834, 82935, 50037, 63133, 70050, 71670, 56959, 77065 | 993483 |
| YOOCHOOSE | 219389, 209219, 218162, 162637, 177943, 307603, 232887, 178076, 199615, 179889, 123750, 153565, 300830, 259673, 187348, 154316, 105676 | 3370578 |

The files contain remapped **session ID, item ID** pairs in row order. They contain no timestamps or original-ID mapping. Preserve file/row identity and the supplied IDs; do not claim these recover raw source identities or treat session IDs as persistent users. Group each period separately by session ID and preserve within-session row order. Cross-period fragments are not stitched by the loader. [Preprocessing output][preprocess], lines 173–200; [loader][util], lines 17–107.

Raw preprocessing removes singleton sessions, then items with fewer than five appearances, then sessions shorter than two. This filtering uses the whole loaded dataset before time partitioning. DIGINETICA uses weekly boundaries anchored backwards from the maximum session-end time, retains the last 17 boundaries, and places older actions into the initial bucket. YOOCHOOSE takes the first 17 daily buckets and discards later actions; its default is not the often-used last 1/64 or 1/4 subset. Assignment is by **action timestamp**, even though bounds use session ends, so a session can be fragmented. DIGINETICA's `timeframe` is scaled by `86400 / max(timeframe)`, and both parsers use naive local-time conversion. Exact raw reconstruction must record timezone, order and these transforms; substituting an ordinary calendar split changes the benchmark. [Preprocessing][preprocess], lines 54–200; [raw readers][raw].

Expand a session of length `L ≥ 2` into `L−1` prefix/next-item examples. Keep at most 50 most recent input items and pad on the left with zero. Upstream also initially includes singleton period fragments, then skips them during batching; report those separately because their counts can affect splits and batch calculations. [Sampler][util], lines 121–239.

For each training window, upstream builds the cumulative item set **before** the 90/10 split, expands prefixes, and randomly holds out 10% of those prefixes using NumPy's shuffle and rounding. Validation is fixed during that window. Joint instead accumulates all historical windows and takes a fresh prefix-level split over the accumulated examples. These splits are not guaranteed identical across methods: replay allocation consumes the global NumPy RNG and sampling/evaluation consume the global Python RNG. Save actual split and sampling state; recording only seed 0 is insufficient for comparison across implementations. [Main data preparation][main], lines 164–200; [Sampler][util], lines 188–216.

For testing, delete every event on an item absent from the cumulative training-window item set, then remove remaining singleton sessions, then expand prefixes. This can turn `[known A, unknown X, known B]` into `[known A] → known B`. It changes adjacency and excludes new-item outcomes. Test scores therefore describe a filtered known-item task. Report raw actions, removed actions, remaining sessions and actual evaluated prefixes for every window. [Loader][util], lines 60–107.

### Model and training settings

| Setting | DIGINETICA | YOOCHOOSE | Source |
| --- | --- | --- | --- |
| Backbone | SASRec variant, width 150, 2 blocks | Same | Paper §4.3, p. 6 |
| Attention heads | 1 | 1 | `main.py` defaults |
| Maximum input length | 50 | 50 | `main.py` defaults |
| Current-data batch size | 256 | 512 | Paper §4.3 and README |
| Evaluation batch size | 64 | 64 | README and defaults |
| Optimizer | Adam, learning rate 0.0005 | Same | Paper §4.3 |
| Maximum epochs per window | 100 | 100 | Paper §4.3 |
| Early stopping | Validation Recall@20; 5 consecutive non-improving epochs | Same | Paper §4.3 and main loop |
| Dropout for ADER, Dropout, Joint | 0.3 | 0.3 | Paper §4.3 |
| Dropout for Finetune, EWC | 0 | 0 | `main.py`, line 141 |
| Nominal replay budget | 30000 | 30000 | Paper §4.3 |
| ADER base weight | 0.8 | 1.0 | Paper §4.3 |
| Embedding L2 | 0 | 0 | `main.py` default |
| Released default seed | 0 | 0 | `main.py` default; paper gives no seed dispersion |
| Allocated item capacity | 43136 plus padding | 25958 plus padding | `main.py`, lines 133–138 |

This SASRec implementation uses learned positional embeddings, scaled input embeddings, causal attention, pre-normalized sublayers, and tied input/output item embeddings. It scores the last prefix position and trains with **full-softmax cross-entropy**, not sampled negatives or sequence-position binary loss. Preserve the actual residual and masking operations in [modules.py][modules]; a generic Transformer layer is not automatically equivalent. Active output candidates are IDs `1…max_item`; the full embedding table is allocated in advance, including future rows. Assert contiguous observed ID coverage rather than assuming `max_item` is always a count. [Model][model], lines 27–103.

Non-Joint methods restore the previous best checkpoint into the next window. `tf.train.Saver` saves optimizer variables as well as weights, so retain Adam slots and step state. Do not reset the optimizer just because the current project sample suggests that as a possible continuation policy. Joint initializes afresh and trains on all accumulated windows each cycle. The default Adam beta/epsilon values are inherited from the chosen TensorFlow runtime and must be recorded when constructing a reference environment or port. [Main][main], lines 193–213 and 271–283.

For ADER, let `B = ceil(current_training_prefix_count / current_batch_size)` and let `M` be the **actual** stored memory size. The upstream replay batch size is `floor(M / B)`. Each current-data step receives a separate replay batch; the replay sampler cycles on its own counter, which is not reset at each current-data epoch boundary. This is not an exact one-pass replay schedule. Reject a zero replay batch or empty effective batch explicitly when testing small fixtures. Log actual current/replay examples and optimizer steps per epoch. Lambda's denominator is the sampler's post-split size, while its numerator uses actual stored memory size. [Main][main], lines 181–200 and 219–248; [Sampler][util], lines 241–273.

### Replay selection

The code samples per-item quotas with a **multinomial** distribution using target frequencies in the candidate pool. The paper's Eq. 1 gives proportional expected counts, without specifying this random allocation. It then caps each quota at the available number of examples. Herding normalizes each feature to unit length, repeatedly chooses the largest dot product with an evolving residual, skips already-selected indices, and stops after roughly `1.1 × quota` attempts. It does not redistribute unused quotas. Consequently, the requested 30000 is an upper budget, not a guaranteed memory size. Preserve these details in an author-code reproduction and record the actual size; changing to deterministic allocation or exact filling is a method change. [Memory][memory], lines 366–461.

The upstream candidate pool includes current **training and validation prefixes** plus previous exemplars. Stored entries contain a truncated prefix, target and dense teacher logits over all currently active items. Frequency counts refer to expanded target examples, not simply raw item occurrences. `ADER_equal` draws multinomial quotas uniformly across all active IDs, including IDs with no candidate examples; it is not an exactly balanced memory. [Main][main], lines 293–312; [memory][memory], lines 383–399.

### Evaluation and reported targets

Rank each eligible next item against all cumulative seen items; no negative candidate sampling and no seen-history masking. Use Recall@10, Recall@20, MRR@10 and MRR@20. Each window averages over expanded prefixes. The reported result is the **unweighted mean of the 16 window metrics**, not a pooled average over all prefixes. Upstream obtains zero-based ranks using two TensorFlow `argsort` operations; equal-score order is not explicitly made stable. Record the reference runtime's tie behavior, test ties separately, and disclose a deterministic tie-rule change in a port. [Evaluator][util], lines 276–339; [model][model], line 103; [main][main], lines 325–330.

For one-based rank `r`, Recall@k contributes `1[r ≤ k]` and MRR@k contributes `1[r ≤ k] / r`. NDCG is not a published ADER metric or the checkpoint-selection criterion.

The following are **paper targets, not results from this checkout**. All values are percentages, from Table 2 on p. 6:

| Dataset | Method | Recall@20 | Recall@10 | MRR@20 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| DIGINETICA | Finetune | 47.28 | 35.00 | 16.01 | 15.16 |
| DIGINETICA | Dropout | 49.07 | 36.53 | 16.86 | 16.00 |
| DIGINETICA | EWC | 47.66 | 35.48 | 16.28 | 15.44 |
| DIGINETICA | Joint | 50.03 | 37.27 | 17.31 | 16.43 |
| DIGINETICA | ADER | 50.21 | 37.52 | 17.32 | 16.45 |
| YOOCHOOSE | Finetune | 71.86 | 63.82 | 36.49 | 35.92 |
| YOOCHOOSE | Dropout | 72.20 | 64.15 | 36.60 | 36.03 |
| YOOCHOOSE | EWC | 71.91 | 63.89 | 36.53 | 35.97 |
| YOOCHOOSE | Joint | 72.22 | 64.16 | 36.65 | 36.08 |
| YOOCHOOSE | ADER | 72.38 | 64.41 | 36.71 | 36.14 |

ADER's reported Recall@20 gain over Dropout is 1.14 percentage points on DIGINETICA and 0.18 on YOOCHOOSE. Its DIGINETICA MRR@20 gain over Joint is only 0.01 point. The paper gives no multi-seed uncertainty or significance-test procedure; numerical ordering alone does not establish a reliable advantage. Its own §4.4 notes that forgetting is less severe when old items reappear. Next-window performance does not directly measure deterioration on fixed old queries. [Paper, pp. 6–7.][paper]

## Discrepancies and execution risks

These are inspection findings, not evidence that the authors' reported runs failed. Synthetic checks exercised extracted upstream methods without importing TensorFlow or accessing real data.

| Finding | Evidence and consequence | Protocol treatment |
| --- | --- | --- |
| Validation examples overlap training contexts | Prefix expansion precedes random splitting. A held-out next-click target can occur in a longer training prefix from the same session. Confirmed with a synthetic session. [Sampler][util], lines 136–143 and 201–208. | Preserve only when explicitly studying the released program. Use session-level validation for the project benchmark and label the resulting method change. |
| Validation feeds future training | The memory candidate pool explicitly appends `valid_subseq`; Joint re-splits history. [Main][main], lines 168–174 and 293–297. | Keep validation permanently out of training and replay in the maintained implementation. |
| Preprocessing sees future frequencies | Global item filtering occurs before temporal partitioning. The released files lack raw identity/time mappings. [Preprocessing][preprocess]. | An artifact reproduction can inherit this historical selection, but cannot claim a strictly online raw-data pipeline. Raw-data reconstruction must retain identity maps and separately specify training-only filtering. |
| Unknown test events are deleted | Confirmed that deleting an unknown middle click creates a known-to-known transition that was not adjacent. [Loader][util], lines 75–96. | Record exact upstream behavior for numerical reproduction; specify exclusions and context handling explicitly in a stricter benchmark. |
| Replay can underfill | Identical synthetic features yielded one selected example from a quota of four. This follows the duplicate skipping and iteration cap. [Memory][memory], lines 419–458. | Log actual quotas and selected counts. Do not silently top up an author-code run. |
| ER-loss ablation does not sort per-example losses | `model.loss` is already a scalar mean; `loss_selection` calls `argsort` on it. A synthetic scalar-return check selected only the first example in a target group. [Model][model], line 93; [memory][memory], lines 482–490. | Exclude this ablation from the first milestone. A corrected per-example loss is a separately named variant. |
| Zero initial validation score is mishandled | `stop_counter` is assigned only on improvement, but the initial best is zero. The first zero score reaches an undefined counter; a later all-zero window can also lack a saved checkpoint. Confirmed the first case. [Main][main], lines 162 and 271–283. | A documented execution fix must initialize the counter and save a valid first checkpoint even at zero. It must not change selection for ordinary positive-score runs. |
| Raw preprocessing has missing imports | `defaultdict` and `np` are referenced but absent from both preprocessing imports and its local `data/util.py` exports. Confirmed the `defaultdict` failure. | Fix imports in a recorded patch if reconstructing raw data. Prefer the published period artifacts for the initial reference specification. |
| Command-line booleans are misleading | Flags use `type=bool`; the string `False` becomes true. | Omit false flags; do not combine method flags. A future maintained CLI should parse booleans properly. |
| Environment sources disagree | README specifies Python 3.7 / TF 2.1.0; `requirments.yaml` has Python 3.7.4 / TF and TF-GPU 2.0.0 / NumPy 1.18.1; `requirements.txt` pins NumPy 1.17.4 and omits TF. The README also misspells environment filenames/commands. | Pick and record a working isolated reference environment; no claim of bitwise reproduction from these files alone. |
| EWC's per-epoch Python updates do not rebuild its penalty | `update_loss` embeds NumPy `F_accum` and `variables_prev` into the graph once before the period. Later assignments in the epoch loop do not replace those graph constants. [EWC][ewc], lines 115–164; [main][main], lines 193–201 and 258–262. | Audit against the intended EWC baseline before claiming full Table 2 replication. Its tuned coefficient/search is not documented; do not treat ADER's default coefficient as the published EWC optimum. |
| One paper ablation number is internally inconsistent | Table 4, p. 7 reports ER-herding MRR@10 = 16.98%, above MRR@20 = 16.85%; nonnegative reciprocal-rank contributions make that impossible on the same queries. Visually confirmed in the PDF. | Preserve the reported values as a discrepancy; do not guess a corrected number or tune to match it. |

The older NumPy code also constructs ragged arrays implicitly in `main.py`. Local execution with NumPy 1.26.4 confirmed failure at line 184 on entering the second window. A list comprehension fixes this ADER replay extraction without changing its contents. The EWC branch contains a separate analogous expression that remains untested. Model execution and optimizer restoration now pass the synthetic checks below; historical-runtime parity and GPU determinism remain unverified.

Dense teacher storage is a material resource constraint: at nominal size 30000 and full catalogue width, float32 logits alone would occupy about **4.82 GiB** for DIGINETICA or **2.90 GiB** for YOOCHOOSE. Actual active width/memory may be smaller. Upstream converts logits to nested Python lists, adding substantial overhead, and computes an entire target group's logits together during herding. Measure peak memory before a full run; a GPU does not remove this host-memory issue. These figures are calculated storage estimates, not measured runtime requirements. [Memory][memory], lines 433 and 447–458.

## Proposed replication sequence

### First milestone

**Scope: DIGINETICA, ADER and Dropout, seed 0, windows 0–16, four published metrics.** Dropout is the essential comparator because it shares ADER's backbone and dropout. A comparison only with dropout-free Finetune would confound the replay/distillation contribution. This first milestone is a bounded result, not full replication of every table.

1. **Freeze provenance and protocol identity.** Choose whether the run is an author-program reproduction or the project protocol below. Record the pinned source, any execution patches, data hashes, complete resolved settings and the chosen environment before results are inspected. Keep datasets under ignored `data/ader/`, runs under new `runs/ader-<run-id>/`, and reports under `reports/`. Never overwrite an existing run.
2. **Audit data before learning.** After an explicit data-acquisition decision, check all 17 file counts against Table 1, positive IDs, cumulative ID contiguity, session fragments, expanded target counts, filtering losses, vocabulary sizes and exact split manifests. A count mismatch stops the claim of matching paper inputs. For raw data, retain original mappings and timestamps as well.
3. **Establish a numerical reference on synthetic inputs.** Use the authors' TensorFlow implementation in an isolated environment first. Verify scores, CE, old-vocabulary KD, lambda, an optimizer update, replay selection and save/reload. A small CPU check can establish correctness; it does not establish affordable full training. Attempt Python 3.7 / TF 2.1.0 as the README's reference; if the environment export is needed instead, record that as a different reference environment. Do not downgrade this repository's Python 3.12 environment or install the entire historical workstation export into it.
4. **Check a small two-window run.** Only after data/runtime scope is authorized, verify real-data loading, validation selection, teacher generation, replay use and complete restart. Use it for correctness and resource measurement. A reduced-data result must never be presented as a Table 2 result, and test results must not drive hyperparameter changes.
5. **Run the full first milestone.** Use the fixed published hyperparameters, 100-epoch ceiling and five-epoch patience. Save each window's validation trace and best complete state; evaluate the next window only from that state. Report all 16 points and the unweighted averages, with direct differences from the appropriate paper targets.
6. **Assess reproducibility before expanding.** Repeat seed 0 from recorded inputs/state; then use predeclared seeds 1 and 2 with frozen hyperparameters as a proposed robustness extension. Report every seed and paired ADER-minus-Dropout differences. Three seeds provide a limited stability check, not a significance guarantee. Do not choose seeds because they match the paper.

An exact match is not a prerequisite for an informative replication. Compare values in **percentage points**, identify whether a discrepancy is due to data/protocol, numerical implementation or stochastic variation, and report unresolved gaps. The paper gives no tolerance interval, so none is invented here. A method claim is supported only by completed controls and uncertainty evidence; passing unit checks or getting positive Recall is not successful replication.

### Preserve project safeguards

For the maintained `cp4285` experiment, use the following explicit methodological replication. These are **proposed project choices**, not statements about the authors' protocol:

- Assign validation by complete original session ID **before** prefix expansion, using a fixed seeded mapping shared by all methods. Keep the same session's fragments on the same side across windows. Approximately 10% of sessions, rather than exactly 10% of prefixes, are held out. Validation events never enter training prefixes, replay memory, Joint history or learned preprocessing.
- Build the cumulative vocabulary from permitted training sessions. For scoring, exclude unknown targets and use only the contiguous known suffix after the most recent unknown context event; require a nonempty context. Apply the same rule to validation and next-window evaluation, and report every exclusion. This deliberately differs from upstream deletion-and-bridging. Freeze each window's scored examples and candidates across methods before evaluating.
- Preserve original raw IDs and timestamps when reconstructing data. Published artifacts can support an explicitly labelled artifact-based comparison using their supplied IDs and row provenance, but not a claim of recovered source identities or purely past-only filtering. A raw-data variant uses training-only filtering and must be reported separately from the historical globally filtered benchmark.
- Evaluate incoming non-validation sessions before their window becomes eligible for training; seal those scores. Later learning from that window is part of prequential evaluation. This lifecycle is explicitly different from the existing permanent-test A/B runner. Do not relax that runner's validation rules or reuse these results as fixed-test retention measurements.
- Keep lambda, nominal budget, actual-memory accounting, herding mechanics, full-softmax objective and published hyperparameters fixed. Correcting unrelated methodological choices at the same time would make discrepancies harder to interpret. Store split/sampling RNG streams independently so one method's replay does not change another method's validation set.

An author-program reproduction and this project variant require separate run identifiers and result tables. The source-faithful specification above is useful for diagnosing numerical agreement; it does not override the repository's holdout rules. A decision to reproduce those upstream methodological choices must be explicit before implementing or running that variant. The present request authorizes reading, inspection and protocol definition only.

### Extend only after the first comparison works

Full Table 2 scope adds Finetune, Joint and a verified EWC configuration, then repeats all five methods on YOOCHOOSE. Joint is a fresh fit on accumulated permitted history, not another warm-start branch. The EWC coefficient and search procedure are unresolved in the published materials; a new validation-only search must be predeclared and reported as ours rather than silently presented as the original setting.

For DIGINETICA Tables 3–4, add ADER memories of 10000 and 20000, ER-herding, ER-random, ADER-equal and ADER-fix. The release's ER variants retain the separately weighted replay term; they are not necessarily equivalent to concatenating all data into one unweighted CE mean. Keep the ER-loss bug and the inconsistent paper table cell visible before attempting that ablation. A corrected ER-loss result is an additional experiment.

Capacity sweeps, Amazon transfer and fixed old-task retention are later project extensions. They would answer different questions and should not be used to rescue a mismatch with the ADER replication.

## Checks and artifacts required before a full run

| Check | Required evidence |
| --- | --- |
| Input preparation | Exact file/count manifest; one session produces exactly its intended targets; no cross-session concatenation; declared treatment of period fragments, unknown items and repeated items |
| Holdout isolation | For the project variant, validation event IDs never occur in training or replay, including as prefix context; all methods use identical window manifests |
| Model parity | Fixed synthetic inputs and identical weights give matching representations, logits, CE and KD; document maximum absolute/relative errors and dtype rather than claiming parity from similar metrics |
| KD boundary | Only old-item logits participate in replay normalization; teacher is detached; new item rows cannot enter the old-vocabulary KD directly; current-data CE still covers the current vocabulary |
| Loss weighting | Check the square-root formula numerically, first-window behavior and separate mean reductions; no accidental hard-label replay CE in ADER |
| Ranking | Known ranks 1, 10, 11, 20 and 21 give correct Recall/MRR; MRR@10 cannot exceed MRR@20; candidate reorder and ties follow the recorded policy; dropout disabled |
| Replay | Verify quota sampling, normalization, duplicate handling, zero-feature handling, iteration cap, actual memory size and teacher refresh; count actual examples consumed |
| Selection | First zero-score checkpoint is valid; ties retain the first best; five non-improvements stop; test metrics never influence training settings |
| State and restart | Save/reload weights, Adam slots/step, teacher logits and vocabulary, exemplars, RNG and sampler positions; the next update after restart matches an uninterrupted run |
| Resource feasibility | Measure peak host/device memory and time for training, full-catalogue scoring and herding on a bounded run; choose compute from measurements |

Each new run should save a configuration/source/environment manifest, file hashes, split and candidate identities, per-window counts and exclusion reasons, validation history, selected epoch, actual replay size and lambda, current/replay exposures, checkpoints and RNG state, raw per-window metrics, and peak memory/runtime. Store large arrays in a suitable binary format with hashes; do not put dense teacher distributions into JSON. Model-only TensorFlow checkpoints are insufficient to restart the entire upstream process because replay memory and Python/NumPy sampler state live outside them.

## Integration with the two packages

The existing `Recommender.train(examples)` contract forbids hidden replay, and the existing `run(...)` assumes one initial validation selection followed by fixed A/B tests. Neither directly expresses ADER's per-window selection, vocabulary growth or distinct current/replay loss terms. [Current model contract](../src/cp4285/experiment/model.py); [building guide](PACKAGES-AND-BUILDING.md).

Before adding an implementation, agree the smallest explicit extension: a model update that receives current examples, replay examples, frozen teacher outputs, vocabulary boundaries and lambda; and a separate temporal experiment function inside `cp4285.experiment`. The experiment package owns period preparation, validation, replay scheduling, shared evaluation and reporting. The model owns SASRec computation, losses, optimizer and feature/logit extraction for selection. Expose only the operations actually needed; do not bury replay inside the existing `train` method or create a general plugin framework.

Extend `experiment/evaluation.py` with shared MRR and a deliberate candidate/tie policy, updating the matching guide and contract tests. Keep full-catalogue scoring batched. A notebook should import these components and explain saved evidence, not contain its own splitter, metric or training loop. Implement durable state before long runs. A PyTorch port should follow a small TensorFlow reference comparison, because switching framework, preprocessing and evaluation together makes failures hard to localize.

## Inspection validation

The initial inspection passed eight isolated synthetic checks: prefix-level validation overlap; unknown-item bridging; herding underfill; scalar-loss selection; Recall/MRR cutoff arithmetic; missing preprocessing import; zero-score early stopping; boolean flag parsing. The checks are retained in ignored `local/sources/ADER-348fc21/audit_checks.py`; they extracted relevant upstream definitions and ran with the then-installed NumPy 2.5.4, without TensorFlow. At that inspection stage no model training occurred. The subsequent execution evidence follows.

## Local verification and cluster handoff

The portable [verification script](../scripts/verify_ader.py) checks the pinned upstream Git blob hashes before importing any authors' code. It refuses an existing output directory and refuses Python's `-O` mode, which would disable its assertions. It uses an isolated environment, leaving TensorFlow out of the shared package dependencies.

On the M2 Pro laptop, **12 checks passed** with Python 3.11.4, TensorFlow 2.15.1 and NumPy 1.26.4 on CPU:

- Period loading, unknown-item filtering, prefix expansion/truncation and Recall/MRR cutoff arithmetic; the known validation overlap is reproduced, not endorsed as an isolation pass.
- Full-softmax cross-entropy and old-vocabulary distillation agree with independent NumPy calculations. Distillation gradients exclude new-item logits and current-example rows.
- Eight deterministic updates reduce the tiny fixture's loss from approximately 4.075 to 0.00394. Inference is deterministic, candidate aligned and leaves variables unchanged. These are debugging observations, not recommendation-quality results.
- Herding returns actual examples and teacher logits matching dropout-free inference. This fixture retains three of four requested examples, exercising actual replay size in lambda.
- All weights, Adam slots and global step restore exactly. The next optimizer update matches uninterrupted execution with dropout disabled. Training dropout is separately active and produces a finite continuation update.
- The authors' `main.py` completes two training/evaluation windows, two epochs each, including replay, validation selection and checkpoint reload, on three generated period files. This smoke check retains their full DIGINETICA embedding allocation but has only four to six active items. Recall@20 is therefore trivial; its metrics are not scientific evidence.

The unmodified entry point first failed at the modern-NumPy ragged-array conversion. The verifier copies the source into its fresh output directory, replaces that conversion with `[record[0] for record in exemplar_data_logits]`, and records the exact diff in `main-numpy-compat.patch`. CPU runs also record a configuration edit disabling GPU devices in that copy. The original source stays unchanged. No changes to validation, replay selection, loss or early stopping are included; the previously identified zero-score early-stopping bug is not exercised by this tiny catalogue.

Recreate the compatibility environment and run from the repository root:

```sh
uv venv --python 3.11 local/ader-verify-env
uv pip install --python local/ader-verify-env/bin/python -r scripts/ader-verify-requirements.txt
local/ader-verify-env/bin/python scripts/verify_ader.py \
  --source local/sources/ADER-348fc21 \
  --output runs/ader-verify-new --device cpu
```

Reuse an existing environment rather than recreating it. On another machine, copy or retrieve the eight pinned files listed in `SOURCE_HASHES` in the script, preserving `data/` paths and the license. No research dataset is needed. Outputs include `verification.json`, TensorFlow checkpoints, generated inputs, an exact entry-point patch and console/training logs. Use a new output directory on every attempt. The local completed run is `runs/ader-verify-20261021-03/`; the ignored artifact directory also contains the installed package versions. The dependency file pins the three direct dependencies; the captured environment records transitive versions.

For the cluster, establish its supported Python, NVIDIA driver/CUDA runtime, scheduler and available GPU/RAM allocation. Run the same verifier in an allocated GPU job with `--device gpu`: the numerical model checks require a visible GPU and verify the executed scoring operation's GPU placement. CPU placement is allowed for unsupported operations such as the legacy int32 training-step variable. The authors' separate entry-point smoke retains their soft-placement behavior, so it is not a GPU placement audit. Do not treat a passed newer-runtime check as equivalence to their Python 3.7 / TensorFlow 2.1 environment; rerun the checks in whichever historical or compatibility environment is selected and record the choice before comparing published numbers. Subsequent user-run cluster diagnostics are recorded below.

The initial and continuation resource gates subsequently passed in user-provided cluster reports. Full DIGINETICA acquisition is complete; the agreed 16-window ADER-versus-Dropout comparison is prepared below. Full process resume including sampler/replay/RNG state, stochastic restart equivalence, tie behavior across devices, other baselines and cross-version numerical parity remain unchecked.

### SoC GPU handoff

The user confirmed Slurm access on 21 October, with a TITAN RTX 24 GB allocation and driver 580.178.04. The login node has Python 3.12.3, so create a separate Python 3.11 environment for this TensorFlow version. Install [uv using its official installer](https://docs.astral.sh/uv/getting-started/installation/), then run from the transferred verification directory:

```sh
uv python install 3.11
uv venv --python 3.11 local/ader-verify-env
uv pip install --python local/ader-verify-env/bin/python \
  -r scripts/ader-verify-requirements.txt 'tensorflow[and-cuda]==2.15.1'
mkdir -p runs
uv pip freeze --python local/ader-verify-env/bin/python > runs/cluster-environment.txt
sbatch scripts/verify_ader_slurm.sh
```

The [TensorFlow 2.15.1 package metadata](https://pypi.org/pypi/tensorflow/2.15.1/json) specifies the CUDA 12 dependencies for `and-cuda`. This environment is a proposed Linux counterpart to the tested Mac compatibility environment, not yet a verified cluster installation. The job requests one task, one `nv` GPU, two CPUs, 8 GiB host RAM and ten minutes; these are synthetic-check allocations, not full-training recommendations. The job exposes the installed NVIDIA libraries to TensorFlow and uses a unique Slurm job ID for outputs. Inspect `runs/ader-gpu-JOBID.log` and `runs/ader-gpu-JOBID/verification.json` after completion; failed runs retain their evidence.

The verifier now removes the authors' unconditional `CUDA_VISIBLE_DEVICES` assignment in its disposable copy and includes that change in the generated patch. This preserves Slurm's device mapping; GPU 0 within the job refers to the visible allocation. The earlier CPU run predates this additional execution-only edit. A transfer bundle may include the working-tree verifier after commit `40e2027`; its file manifest and the verification report record exact hashes. No remote account details or research data belong in the public repository.

**First cluster attempt and correction:** the user installed Python 3.11.17 and TensorFlow 2.15.1 successfully after redirecting `TMPDIR` to home storage and running dependency installation through a CPU allocation with 8 GiB RAM. The GPU job then failed on `global_step`: our verifier had disabled soft placement while assigning the whole graph to GPU, but TensorFlow's legacy int32 `VariableV2` supports only CPU in this runtime. This is a verifier placement defect, not evidence against ADER or a reason to reinstall CUDA. The verifier now enables soft placement, consistent with upstream `main.py`, and uses executed partition-graph metadata to require that the scoring matrix multiplication runs on the requested device. Missing evidence or CPU scoring in GPU mode fails explicitly. See [TensorFlow's placement guidance](https://www.tensorflow.org/guide/gpu). The new check adds one verification gate; `scoring_devices` is recorded in successful reports. Full GPU execution still awaits the user's rerun. Transfer the updated `scripts/verify_ader.py` over the original bundle version; the original archive is retained and new reports record the updated script hash. Existing failed run directories remain untouched.

**Second cluster attempt:** model/data gates, including GPU scoring placement, loss/gradient checks and optimizer continuation, completed successfully. The upstream entry-point subprocess then failed on `ScatterAdd` because `main.py` sets `TF_DETERMINISTIC_OPS=1`, while TF 2.15 has no deterministic GPU implementation for scatter updates on legacy reference variables. The compatibility verifier now sets that flag to `0` in the disposable GPU smoke copy and its child environment, leaving CPU smoke and the pinned originals unchanged. It prints an explicit notice, records the exact diff, and reports `main_smoke.deterministic_ops_enforced=false`. This allows testing the training loop without promising identical GPU results across runs; it is not a full reproduction protocol change. TensorFlow documents [unsupported deterministic operations and reproducibility limitations](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism). For a later reproducibility-focused run, assess a historical runtime or separately validate resource-variable semantics rather than treating this compatibility smoke as sufficient evidence. The corrected GPU subprocess remains unverified pending another cluster run.

## First real-window resource pilot

The initial authorized pilot was **DIGINETICA period 0 training, period 1 testing**, using only those two released files at the pinned source commit. This is initial cross-entropy training followed by replay construction; no distillation update occurs until the next window. A one-window metric cannot be compared directly with Table 2's 16-window mean or establish an advantage over Dropout.

The downloaded files remain ignored under `local/ader-window-inputs/`. Their Git blob hashes are `5f2f70ec5544f82ee74b8df78b77334f637e7883` (period 0) and `c02fc6de4fe15a41f4d686c78cb6d5b43259abb6` (period 1), verified against the pinned repository tree. The [first file](https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data/DIGINETICA/period_0.txt) contains 70,739 actions, 15,076 sessions, 18,569 contiguous item IDs and 55,663 next-item prefixes. Five singleton fragments add five rows to the upstream sampler before its split. The [second file](https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data/DIGINETICA/period_1.txt) has 37,586 actions and 7,896 sessions; upstream known-item filtering and singleton removal leave 29,952 actions, 6,847 sessions and 23,105 test prefixes. Counts were audited locally without real-data training.

`scripts/profile_ader_window.py` verifies source and data hashes before creating a fresh run directory, copies exactly two period files by default (three with `--windows 2`), and runs upstream `main.py`. It uses seed 0, width 150, two blocks, length 50, current batch 256, evaluation batch 64, learning rate 0.0005, dropout 0.3, up to 100 epochs with validation Recall@20 patience 5, and replay budget 30,000. The tested NumPy and GPU-visibility fixes are reused; deterministic GPU operations remain unenforced. The profile also initializes the early-stop counter and makes the first checkpoint eligible at zero validation score, as required by the inspection finding. The exact combined diff is retained in `profile-main.patch`. No validation, filtering, ranking, herding or loss changes are made.

The Slurm job requests one A100 40 GB allocation, four CPUs, 64 GiB host memory and a three-hour limit. These are provisional ceilings, not observed requirements. Host memory accounts for upstream Python-list teacher logits; GPU memory alone does not describe replay storage. No environment reinstall is required. After transferring/extracting the pilot bundle into the existing cluster directory:

```sh
mkdir -p runs
sbatch scripts/profile_ader_window_slurm.sh
```

Use the printed job ID to inspect `runs/ader-window0-JOBID.log` and `runs/ader-window0-JOBID/console.log`. The latter receives upstream training progress. The directory also retains:

- `manifest.json`: source/data/script hashes, all installed package versions, command, counts and limits.
- `profile.json`: final status, elapsed time, selected epoch, test Recall/MRR, epoch timings, herding time and actual replay size. Upstream metrics are rounded to four decimals.
- `gpu-memory.jsonl`: samples for the training PID only, at approximately two-second intervals. Peak GPU memory is a sampled lower bound; unsupported MIG readings remain null, not zero.
- `upstream-main/results/DIGINETICA-ADER/`: original training log, selected checkpoint, runtime/build information, training/validation prefix contents and candidate IDs, per-epoch validation/timing history and replay metadata.

Host peak RSS is the maximum across child processes (normally the training process), not total job RAM; Slurm accounting can provide a separate view. If the scheduler kills the job, logs, manifest and completed checkpoints remain but a final profile may be absent. Use a new job/run directory for retries. Upstream checkpoints do not persist replay and sampler/RNG state, so this pilot is not a resumable first stage of the full 16-window run. Do not silently increase epochs or change hyperparameters based on period 1 test metrics.

Local validation: 24 offline tests, Ruff checks and the unchanged 13-gate CPU verifier pass. An additional synthetic CPU run of the instrumented entry point checks epoch/split/replay artifact creation and process monitoring. It uses only tiny generated data and is labelled accordingly. The two real input files pass `--prepare-only`, which does not train. The user subsequently supplied a successful first-window report, summarized below. The agent has not submitted a cluster job.

## First-window result and real continuation pilot

**User-provided evidence, 21 October 2026:** the first real-window profile reports `status=passed`, return code 0 and 220.32 seconds total. Peak child host RSS was 21,328.70 MiB (20.83 GiB); sampled GPU peak was 1,468 MiB (1.43 GiB), with no monitor errors. Epoch 10 was selected by validation Recall@20 (0.49353), with training stopping after epoch 15. Test Recall@20 was 0.4922, MRR@20 0.1771, Recall@10 0.3760 and MRR@10 0.1690 (four-decimal upstream precision). Replay construction took 121.09 seconds and stored 27,717 examples for 18,569 active items. These values were pasted by the user, not independently retrieved. The metrics cover only the initial window, which has no distillation, and do not establish the paper's comparison with Dropout.

The user authorized **one real continuation window** next. Run initial period 0 → test 1 and continuation period 1 → test 2 together in a fresh process. The first pilot's checkpoint lacks the replay buffer and sampler/RNG state, so it cannot faithfully resume this workflow by itself. The rerun may differ numerically because GPU deterministic-op enforcement remains disabled; preserve both runs.

The additional [period 2 input](https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data/DIGINETICA/period_2.txt) has pinned Git blob hash `3e1fe1dde546f4bc98df2d60a3dee1065fbf0020`. At continuation, cumulative known items grow to 22,692. The training period has 37,586 actions, 7,896 sessions and 29,690 next-item prefixes; 87 singleton fragments add rows to the released sampler. Period 2 has 31,089 raw actions in 6,453 sessions. Filtering with the continuation vocabulary leaves 26,522 actions, 5,863 sessions and 20,659 test prefixes. Future test items never enlarge the current candidate set in this audit. Upstream reuses period 1 as training after its initial test evaluation, as specified by the chronological next-window protocol.

The same profiler now accepts `--windows 2`; its default remains one window. The new Slurm entry point requests one A100 40 GB allocation, four CPUs, **64 GiB RAM**, and three hours. No environment reinstall or change to the paper hyperparameters is needed. After extracting the continuation bundle into the existing cluster directory:

```sh
mkdir -p runs
sbatch scripts/profile_ader_continuation_slurm.sh
```

For the returned job ID, inspect `runs/ader-continuation-JOBID.log`, `runs/ader-continuation-JOBID/console.log` and, after completion, `runs/ader-continuation-JOBID/profile.json`. The report contains process-wide memory/time and a `windows` list with separate metrics, validation histories, selected epochs and replay timings. Under `upstream-main/results/DIGINETICA-ADER/profile-period1/` and `profile-period2/`, the profiler records splits, candidates, runtime, replay input/batch counts, adaptive lambda, optimizer steps before/after training and at checkpoint selection, and completed KD update counts. A successful report requires the second window to restore the first window's **selected** step, consume its actual replay size, use the expected adaptive lambda and complete one distillation update per training batch. These checks complement the existing synthetic loss/gradient and full optimizer-restoration gates; the real pilot does not independently re-prove tensor-level state equivalence.

Local validation: **31 repository tests pass**, including refusal of incomplete logs, missing distillation, incorrect lambda, altered replay sizes and inconsistent optimizer steps. A two-window synthetic CPU self-check also passes with the new instrumentation and report collection; it verifies restoration from the selected epoch rather than the final training epoch. Real inputs pass preparation only; no real-data training was run locally. The user subsequently supplied a successful continuation report, summarized below. Memory sampling limitations, upstream validation overlap/reuse, omitted resumable replay/RNG state and unverified GPU repeatability still apply. The full 16-window baseline comparison is not started.

## Full 16-window comparison

**Continuation evidence supplied by the user:** both windows passed in 615.82 seconds. Peak child host RSS was 45,930.21 MiB (44.85 GiB), sampled peak GPU memory was 1,468 MiB, and no monitor errors were reported. The first selected checkpoint was step 1,960 (epoch 10); continuation restored that step, consumed 27,696 replay examples in batches of 263, used lambda 0.7356937873 and completed 945 KD updates (105 batches × 9 epochs). Continuation selected epoch 4 and tested at Recall@20 0.5040 / MRR@20 0.1818. Its replacement replay had 27,965 examples. These figures came from pasted output; the agent did not retrieve or independently rerun the real experiment. First-window Recall@20 changed from 0.4922 to 0.4939 across runs, consistent with the disclosed lack of exact GPU repeatability. Cross-window metric differences use different test periods and do not isolate a distillation effect.

**Frozen scope:** one complete seed-0 ADER run and one complete seed-0 Dropout run, each training periods 0–15 and testing periods 1–16. Use the fixed defaults above, not settings chosen from the pilot test scores. The Dropout command adds `--dropout True` and changes only the output label; dropout rate remains 0.3, with no replay or distillation. Both restore model/Adam state from the previous selected checkpoint. Keep the authors' validation split generation, unknown-item filtering and RNG consumption. Consequently, the methods can have different validation splits after the initial window. Saved split hashes disclose this; matched validation splits would be a separate methodological variant. This is an author-artifact comparison under the recorded modern TensorFlow compatibility changes, not an exact historical-environment reproduction.

All 17 DIGINETICA files under ignored `local/ader-window-inputs/` are hash-pinned in the profiler. Every period action count matches Table 1, totaling **993,483**. Active training catalogue sizes, in order, are 18,569; 22,692; 25,306; 27,636; 29,619; 32,585; 34,596; 36,691; 38,501; 40,135; 41,197; 41,868; 42,369; 42,826; 43,023; 43,105. Period 16 is only evaluated and never expands the training catalogue. Per-window counts/filtering losses and file SHA-256s are retained in each manifest.

**Resource estimate:** on 64-bit CPython, a stored Python float plus a list pointer is approximately 32 bytes. Two nominal 30,000-example buffers at the final two training widths (43,023 and 43,105) account for about **77.00 GiB** before other working memory. Scaling the measured continuation RSS by the ratio of these logit-cell counts to the observed old/new replay buffers gives about **100.88 GiB**. This assumes comparable overhead and is not a guaranteed peak. Retain the original storage representation for this author-code run; do not silently switch to NumPy arrays. Request **128 GiB host RAM** and one A100-40 allocation for ADER, with a **12-hour ceiling on `gpu-long`**. Dropout has no dense replay: request **16 GiB** and a **three-hour ceiling on `gpu`**. These requests use the partitions reported by the user; their current queue availability and whether a node can satisfy 128 GiB have not been queried remotely. Limits are not runtime forecasts. If submission is rejected, inspect the scheduler reason before changing the scientific settings.

After transferring/extracting the full bundle into the existing cluster working directory, reuse the installed environment:

```sh
mkdir -p runs
sbatch scripts/run_ader_full_slurm.sh ader
sbatch --partition=gpu --mem=16G --time=03:00:00 --job-name=dropout-full \
  scripts/run_ader_full_slurm.sh dropout
```

The jobs are independent. ADER writes `runs/ader-full-ADER_JOBID/` and `runs/ader-full-ADER_JOBID.log`; Dropout writes `runs/dropout-full-DROPOUT_JOBID/` and the matching `.log`. Follow each run's `console.log` for progress. Check `squeue -j JOBID` while queued/running and `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode` after completion. A scheduler-killed job can lack `profile.json`; preserve its evidence. Neither job resumes the pilot, and a timed-out process cannot resume solely from its model checkpoints. No remote submission was performed by the agent.

Each completed report contains full-precision evaluator metrics for every window, the **unweighted 16-window mean**, selection traces, selected optimizer steps, continuation/KD counts, replay size/time, split hashes and process resource measurements. Full-precision `metrics.json` files are written from the existing test evaluator result without running test inference again. Upstream four-decimal text logs remain available for consistency checks. A successful ADER report requires the replay/lambda/optimizer checks at every continuation; a successful Dropout report requires no replay or KD at any window. The last ADER window still builds replay, preserving the upstream loop even though no further update consumes it.

When **both** jobs have passed, replace the job-ID placeholders and run:

```sh
local/ader-verify-env/bin/python scripts/compare_ader_runs.py \
  --ader runs/ader-full-ADER_JOBID \
  --dropout runs/dropout-full-DROPOUT_JOBID \
  --output runs/ader-vs-dropout-seed0.json
```

The comparison refuses missing periods, failed runs, non-default commands, differing source/data/script/environment fingerprints or candidate catalogue sizes. It recomputes the unweighted means, reports ADER-minus-Dropout and differences from Table 2 in **percentage points**, and lists which validation splits match. It does not select a seed/run or invent a significance threshold. Use a new comparison filename if one already exists.

**Local checks:** 40 repository tests and Ruff pass. Both ADER and Dropout completed all 16 synthetic CPU windows, testing repeated checkpoint handoffs and the last-window collection path. Only those ignored synthetic fixtures reduce the allocated item table to 32 and use two epochs/tiny replay; real-run settings retain all paper defaults. Real-data preparation checks all hashes/counts without training. This verifies execution plumbing, not full-data performance, later-window peak memory or GPU numerical parity. The complete real comparison, repeat-seed checks, seeds 1/2, other baselines, YOOCHOOSE and resumable replay/RNG state remain outside the completed work.

## Completed seed-0 results

The user supplied output from `compare_ader_runs.py` for the two completed cluster runs. The comparison command checks successful status, all 16 ordered windows, default commands, matching source/data/script/package fingerprints and candidate catalogue sizes, then recomputes unweighted means. The agent reviewed this pasted output; it has not downloaded the original reports/checkpoints or independently rerun the real experiment. No full-run peak memory or ADER elapsed-time figure was included in that comparison, so the earlier resource estimate remains an estimate.

All scores below are **percentages**, averaged equally across the 16 test windows. Paper targets are Table 2, p. 6 of `ader-2007.12000v1.pdf`.

| Metric | Paper ADER | Observed ADER | Paper Dropout | Observed Dropout | Observed ADER advantage (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall@20 | 50.21 | 49.9587 | 49.07 | 48.8569 | +1.1017 |
| Recall@10 | 37.52 | 37.3614 | 36.53 | 36.3926 | +0.9688 |
| MRR@20 | 17.32 | 17.2579 | 16.86 | 16.7772 | +0.4807 |
| MRR@10 | 16.45 | 16.3859 | 16.00 | 15.9134 | +0.4725 |

ADER is below the paper by 0.2513 pp in Recall@20 and 0.0621 pp in MRR@20. Its advantage over Dropout closely tracks the paper's 1.14 pp Recall@20 and 0.46 pp MRR@20 differences. It leads on every metric in each of the 15 continuation windows. The initial window has no distillation; small method-run differences there must not be attributed to replay.

Validation split fingerprints match only for test period 1 and differ for periods 2–16, as anticipated from method-dependent RNG consumption. This result reproduces the released procedure, including its overlapping prefixes and validation reuse, rather than establishing holdout isolation. It supports the statement that this seed-0 modern-runtime reproduction closely matches the published scores and direction of the ADER/Dropout comparison. It does not establish statistical significance, exact historical-runtime parity or an isolated causal effect of distillation.

Proposed next checks, not yet executed: preserve complete remote run artifacts, repeat seed 0 to measure execution variability, then run predeclared seeds 1 and 2 with frozen settings. A matched, session-isolated validation variant would be a separate methodological comparison.

[paper]: https://arxiv.org/pdf/2007.12000v1
[main]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/main.py
[model]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/ADER.py
[modules]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/modules.py
[util]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/util.py
[memory]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/util.py#L353-L522
[preprocess]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data/preprocessing.py
[raw]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data/util.py
[ewc]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/EWC.py
[data]: https://github.com/doublemul/ADER/tree/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/data
[license]: https://github.com/doublemul/ADER/blob/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94/LICENSE
