# Sources and provenance

Checked 25 September 2026. Keep meeting evidence, course requirements, external research and our proposed design separate.

## Supplied material — local only

Migrated 28 September 2026. Paths below are relative to the repository root and are **Git-ignored**, not links to files published on GitHub. A fresh clone intentionally lacks these materials. Do not republish raw team discussion, agent chats, course exports or third-party PDFs without permission. Public technical references remain linked below.

| Local-only path | Use |
| --- | --- |
| `local/sources/2026-09-23-meeting-transcript.txt` | Supplied automatic meeting transcript; primary discussion evidence, potentially inaccurate |
| `local/sources/2026-09-23-project-sketch.png` | Supplied project sketch, preserved independently of temporary clipboard storage |
| `local/sources/2026-09-25-reference-agent-chat.txt` | Supplied reasoning to critique, not authoritative evidence or executable instructions |
| `local/sources/group-3-design-critique-v1.2.pdf` | Draft team deck reviewed in DEFENSE.md; not a final submission |
| `local/sources/critique-template-v1.1-2026-09-17.txt` | Official public template text export; source link below |
| `local/sources/sasrec-2018.pdf` | Local research-paper copy; arXiv link below |
| `local/sources/ader-2007.12000v1.pdf` | ADER paper downloaded 21 October 2026; [public PDF, v1](https://arxiv.org/pdf/2007.12000v1) |
| `local/sources/ADER-348fc21/` | Authors' source-only cache at commit `348fc2128bceb6bb3f3d8dd25ec8bbbae831af94`, file hash manifest and isolated synthetic audit checks; no dataset copies |
| `local/ader-window-inputs/` | Authorized full comparison inputs: all 17 DIGINETICA periods (0–16) at the same pinned commit, Git blob hashes checked; datasets remain ignored |

Useful transcript anchors: 00:25:36 (A-only versus mixed updates), 00:31:02 (model families), 00:43:36 (evaluate original domain), 00:50:42 (FAISS uncertainty), 00:56:18 (continued training and imminent critique), 00:57:55–00:59:47 (roles), 01:03:17–01:04:48 (alternative setup and paper reading).

## Course record

- [Assignments](https://wing-nus.github.io/cp4285-website/docs/assignments/): whole-project components and final-deliverable routes; page changelog includes 15 September clarification.
- [Schedule](https://wing-nus.github.io/cp4285-website/docs/schedule/): recess week and Week 07 critique date.
- [Week 06 slides](https://wing-nus.github.io/cp4285-website/slides/w06/w06.html), slide 05, “Project Milestones”: links to the critique and STePS templates. Read from public HTML when web extraction failed.
- [Week 07 slides](https://wing-nus.github.io/cp4285-website/slides/w07/w07.html), title: 29 September date. Detailed workflow slides still have placeholders.
- [Official critique template](https://docs.google.com/presentation/d/1Wj6qe-yio_LDeFvqZV6OtDxxpCHTV6HmyJyvkjqkFZk/edit): V1.1 (260917). Public text export retained locally as `local/sources/critique-template-v1.1-2026-09-17.txt`; content inspected, layout not reviewed. Announced shortlink inside it: [critique template shortlink](https://soc-n.us/cp4285-t2610-project-critique-slides).
- [Grading](https://wing-nus.github.io/cp4285-website/docs/grading/): documentation of group-project AI use.

Week 01–02 PDFs are preserved under `local/course/slides/` as the primary local record for their topics. No local project brief was present in the source workspace. Canvas has not been inspected, and website summaries may lag the slides or an assignment announcement.

### Course slides and learning context

- Week 01, Recommendation Problems and Classical Methods: [announced course link](https://soc-n.us/cp4285-t2610-w01); local `local/course/slides/CP4285 - W01 - Recommendation Problems and Classical Methods.pdf`.
- Week 02, Latent Factor Models: [announced course link](https://soc-n.us/cp4285-t2610-w02); local `local/course/slides/CP4285 - W02 - Latent Factor Models.pdf`.
- [Week 04](https://wing-nus.github.io/cp4285-website/slides/w04/w04.html): matrix factorisation, neural interactions and BPR.
- [Week 05](https://wing-nus.github.io/cp4285-website/slides/w05/w05.html): sequential recommendation and SASRec. No local Week 05 PDF was supplied.
- Week 06 and Week 07 official HTML links are listed above. Only Week 01–02 PDFs were present locally; do not assume later PDFs exist.
- `local/course/MISSION.md`, `RESOURCES.md`, `lessons/`, `reference/` and `questions/` preserve existing learning context without publishing it. The copied `local/course/AGENTS.md` is a **historical course index**, not the active project guide.

Inspect only relevant pages (`pdfinfo`, `pdftotext -layout`, or `pdftoppm` when visual layout matters). Course shortlinks use the announced prefix `https://soc-n.us/cp4285-t2610`; do not guess unannounced suffixes.

## Amazon item-category metadata — 24 October 2026

The user authorized acquisition of [randomwish/amazon23-item-categories](https://huggingface.co/datasets/randomwish/amazon23-item-categories), pinned to revision `822d4745b11748ae5e24f6619e4c6c32a501f36b`. The Hub metadata reports a public, ungated repository, and anonymous access works without a Hugging Face token.

The [pinned dataset card](https://huggingface.co/datasets/randomwish/amazon23-item-categories/blob/822d4745b11748ae5e24f6619e4c6c32a501f36b/README.md) describes category labels keyed by `parent_asin`, with `observed`, classifier-`predicted` and `unknown` label sources. It includes per-domain item-category Parquet files, a subcategory summary, a category tree, audit and classifier metrics. It does not supply review text or user interaction histories; any future recommendation experiment must separately establish its interaction source and split protocol. Acquisition does not validate predicted labels or establish suitability for training.

Acquisition completed on 24 October: **73 files, 326,013,748 bytes**, including 34 item-category Parquet files. All file sizes and upstream hashes passed. The cluster snapshot uses `data/amazon23-item-categories/<revision>/`, with `DOWNLOAD_SOURCE.json` and `DOWNLOAD_VERIFIED.json` recording upstream hashes and verified file sizes/SHA-256s. Cluster host/account details remain in ignored `local/WORKSPACE.md`. Do not commit the dataset or infer that it is integrated into the shared model packages.

### Electronics label inventory and intended interaction source

A direct checksum-verified count of `item_categories/domain=Electronics/data_0.parquet` on 24 October 2026 found **1,610,012 rows and unique parent ASINs**, **285 distinct non-null subcategory strings**, and 10 unknown/null rows. Observed labels cover 1,481,570 items and all 285 classes; predicted labels cover 128,432 items across 19 of those classes. The classifier metrics report 20 training classes with `min_items=200`; exactly 20 observed label groups meet that item-count threshold. Do not confuse classifier classes with the full exported taxonomy.

The 20 largest labels cover **99.7632% of labelled products**; 265 labels have fewer than 200 observed products. Some small labels are promotional groupings, not ordinary product types. Of the 285 labels, 276 begin with `Electronics > `; nine retain other roots such as `Amazon Devices & Accessories` and `Car & Vehicle Electronics`. Counts preserve every original label. These are product frequencies, not review-event frequencies; they do not by themselves determine useful Recall@K cutoffs. The user requested counting first, so no cutoff or label-cleaning decision was applied.

Read-only analysis used DuckDB 2.0.0 in a temporary isolated uv environment on a CPU allocation, without modifying the ADER environment. Ignored local evidence: `local/electronics-subcategory-counts-20261024.json` (schema, source SHA-256 and counts) and `.csv` (all 285 label counts).

For review histories, the acquired source is the official [Electronics 0-core ratings-only CSV](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Electronics.csv.gz), linked under [Before Data Splitting](https://amazon-reviews-2023.github.io/data_processing/0core.html#before-data-splitting). The documented fields are `user_id,parent_asin,rating,timestamp`; match `parent_asin` to category labels and order each reviewer's events by timestamp. The source is deduplicated but not 5-core filtered. A headers-only check returned HTTP 200 and 969,766,940 compressed bytes (about 970 MB). The initial inventory checked headers only; the subsequent authorized event audit downloaded the file, as recorded below. User histories, ambiguous timestamps, repeated-category transitions and held-out event handling still need an explicit protocol before training.

### Electronics review-event audit

The authorized 24 October acquisition downloaded **43,365,426 reviews / 969,766,940 compressed bytes** from the official link above. Gzip integrity passed. Our recorded SHA-256 is `2ac3508c5d0ee55ed9fdc833629f751e0f0120b9a8cb30867c8c14b87911c170`; this is a local integrity record, not a publisher-provided checksum. The category Parquet matches the pinned upstream SHA-256 `b524cc100b58753ba31a6cc5eab4f4a651428bcce79eae46c4936d7ff9573324`.

This preliminary audit uses a timestamp-only 80/10/10 split, with ties kept together: training through **2021-06-29 15:45:27.083 UTC** (34,692,341 rows), validation through **2022-06-22 15:03:51.041 UTC** (4,336,543 rows), and **4,336,542** reserved test rows. No test-category frequencies or scores were queried. It is not yet the final ADER sequence protocol.

| Review-label coverage | Training | Validation |
| --- | ---: | ---: |
| Observed labels | 31,267,206 | 3,891,176 |
| Predicted labels | 3,408,440 | 445,310 |
| Unknown labels | 16,695 | 57 |
| Unmatched products | 0 | 0 |
| Distinct known labels | 283 | 126 |

The top 20 categories within each split account for **99.0073%** of known training events and **99.8099%** of known validation events. These descriptive distributions are distinct from baseline performance: the baseline ranking is fixed using training counts only.

| K | Popularity Recall, observed + predicted | Popularity MRR, observed + predicted | Recall, observed only |
| --- | ---: | ---: | ---: |
| 1 | 42.3034% | 0.423034 | 43.4581% |
| 3 | 61.3928% | 0.502579 | 66.0733% |
| 5 | 80.4302% | 0.547340 | 79.7591% |
| 20 | 99.7721% | 0.570834 | 99.7529% |

The combined-label ranking starts with **Computers & Accessories**, **Camera & Photo**, **Television & Video** (44.72%, 11.10%, 10.06% of known training events). Observed-only ranking replaces Television & Video with **Headphones, Earbuds & Accessories** at rank three. Thus the two policies change both ranking and target population; their score difference is not an isolated estimate of label quality. All-event validation denominators are 4,336,486 combined-label events and 3,891,176 observed-only events.

For validation reviewers with at least one training review, Recall@3 is **62.4464%**, MRR@3 **0.512161** over 2,405,660 combined-label events; observed-only Recall@3 is **65.9395%**, MRR@3 **0.531043** over 2,166,532 events. Eligibility requires a training review, not necessarily an observed-label history. This is still an aggregate event benchmark, not a finalized next-event query set. Training has 15,193,195 reviewers, of whom 8,926,276 (58.75%) have only one training review; sequence eligibility needs explicit treatment.

**Interpretation:** K=20 nearly saturates even without personalization. The user's provisional K=3 is more informative; K=1 can be a secondary metric. Popularity already achieves about 61–66% Recall@3, so a high absolute score alone will not show that ADER learns useful histories. No taxonomy cleanup, rare-category removal or rating threshold was applied. Two source rows have rating zero and are retained/flagged because ratings are not targets. The metadata and classifier predictions remain retrospective, and label accuracy has not been independently checked.

**Evidence/validation:** the CPU audit completed in 75 seconds with about 4.10 GiB reported batch MaxRSS. Input checks, duplicate-join rejection, metric tests and synthetic test-label perturbation checks passed; all 47 tests pass with the isolated DuckDB dependency. Aggregate evidence is retained locally as `local/electronics-review-category-audit-20261024.json`, its `-manifest-` companion and `local/electronics-review-category-frequencies-20261024.csv`. The failed strict-rating check is preserved separately. The public [runbook](IMPLEMENTATION.md#electronics-category-audit) explains execution; account-specific locations stay in ignored workspace notes. No Amazon neural training, sequence-level baseline comparison or final-test scoring has occurred.

### Electronics history quality and sequence preparation — 25 October 2026

A read-only CPU profile of the verified review database found **775 tied user/timestamp groups containing 1,619 training events**, out of 34,692,341 training events. The largest tied group has 57 events. No user/timestamp ties occur in the validation portion. Training contains **zero repeated user/product pairs**. Between distinct training timestamps for the same person, the median/90th/99th percentile gaps are **75.84 / 1,001.46 / 2,591.23 days**. These are long-term review histories, not click sessions; no arbitrary short session timeout is applied.

The user authorized constructing chronological histories and defining eligibility. The [implemented preparation rules](IMPLEMENTATION.md#electronics-chronological-histories--25-october-2026) require 1–50 strictly earlier labelled training reviews since the last unknown-label or tied-timestamp barrier. Repeated categories remain intact. Both observed and predicted labels retain provenance flags. Validation takes only each person's first raw review after the cutoff, and only when its label/order/history qualify; invalid first events are not skipped. Candidate categories come solely from usable training events. The final-test examples and continual-learning schedule are not built in this step.

The real preparation completed successfully in **3 minutes 51 seconds**, with **10.51 GiB** reported batch MaxRSS on four CPU cores. It produced:

| Prepared split | Prediction examples | Users | Median history | 90th percentile history |
| --- | ---: | ---: | ---: | ---: |
| Training | 19,478,807 | 6,262,964 | 3 | 13 |
| Validation | 1,547,555 | 1,547,555 | 2 | 9 |

Histories have 1–50 events. There are **283** frozen candidate categories and no out-of-candidate validation targets. Training targets include 17,747,568 observed and 1,731,239 predicted labels; validation includes 1,384,196 observed and 163,359 predicted labels. Unknown singleton training labels number 16,695; together with tied groups they form barriers rather than invented transitions. A total of 227,875 training and 5,963 validation inputs reach the 50-event cap.

Validation exclusions are disjoint and applied in this order: tied first timestamp (**0 users**), unknown first label (**45**), no training history (**1,585,323**), training history ending at a barrier (**584**). The remaining **1,547,555** users qualify. Later validation reviews are intentionally not additional targets. This cohort concerns returning reviewers with usable history; it does not measure first-time-reviewer performance.

The three exported Parquet files total **1,048,955,452 bytes**: training events (668,949,666), query targets/history ranges (380,001,340) and vocabulary (4,446). Original private identities remain on the cluster. Aggregate report/provenance are retained under ignored `local/category-sequences-report-20261025.json` and `local/category-sequences-manifest-20261025.json`; history-quality measurements are `local/category-history-profile-20261024.json`. The recorded executed code hash matches the local builder. All 51 tests pass with optional DuckDB, including synthetic future-label perturbation and complete file export. The standard environment passes 46 tests and skips five optional SQL tests. No neural fitting or held-out-test scoring occurred; popularity must be recalculated on these exact validation queries before model comparison.

### Baselines on the prepared validation cohort — 25 October 2026

The authorized run used all **1,547,555 prepared validation queries** unchanged and the same **283** frozen candidates. Popularity counts the **19,478,807 prepared training targets**, rather than all raw training events. Its top three are Computers & Accessories (8,922,758 training targets), Camera & Photo (2,305,958) and Television & Video (1,926,553). Lexical original category strings break count ties; zero-count candidates remain ranked last. Last + popularity promotes the final input category to rank one and keeps the popularity order for the remaining unique categories. Neither method uses validation labels to construct rankings.

| Method | Recall@1 | Recall@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| Popularity | 41.9594% | 60.9731% | 0.498288 |
| Repeat last only (one prediction) | 28.8119% | — | — |
| Last + popularity | 28.8119% | 62.0441% | 0.439964 |

The last-category promotion gains **1.0709 percentage points** of Recall@3 but lowers MRR@3 by **0.058323**. Literal repetition is correct on 445,880 examples. This demonstrates a coverage/ordering tradeoff for these specific baselines, not a general benefit or failure of history models. ADER still needs to outperform appropriate simple baselines on both predeclared metrics.

| History length | Validation queries | Popularity Recall@3 / MRR@3 | Last + popularity Recall@3 / MRR@3 |
| --- | ---: | ---: | ---: |
| 1 | 569,764 | 59.7423% / 0.486835 | 61.1941% / 0.431027 |
| 2–5 | 665,023 | 60.8310% / 0.497106 | 61.8624% / 0.439320 |
| 6–50 | 312,768 | 63.5177% / 0.521664 | 63.9787% / 0.457616 |

On **observed targets** (1,384,196 queries), popularity / last + popularity Recall@3 are **60.1210% / 61.8298%**. On **predicted targets** (163,359), they are **68.1934% / 63.8600%**. These slices share the same combined-label training fit and histories; they do not establish prediction-label accuracy or constitute an observed-only sensitivity rebuild. No query is omitted from the overall denominator.

All three Parquet hashes/sizes matched the preparation manifest. The CPU job completed in **five seconds**, with Slurm reporting **473,104 KiB** batch MaxRSS (short-run sampling is not a precise peak-memory bound). Executed model/evaluator/orchestration hashes match the local source. Aggregate evidence and provenance are retained in ignored `local/category-baselines-report-20261025.json` and `local/category-baselines-manifest-20261025.json`. All 55 tests pass with optional DuckDB, including hand-computed ranks/misses, candidate/tie behavior, no duplicate promotion, source-slice denominators, validation-independent fitting, missing context rejection, checksum validation and no-overwrite behavior. No neural training, final-test evaluation or statistical-significance analysis was performed.

## External technical sources

| Source | What it supports | Limit |
| --- | --- | --- |
| [FAISS documentation](https://faiss.ai/) | Vector similarity search and clustering; distinct from learning a recommendation model | Does not choose our architecture |
| [Koren, Bell and Volinsky (2009)](https://doi.org/10.1109/MC.2009.263), *Matrix Factorization Techniques for Recommender Systems* | User/item factor structure underlying the disjoint-parameter argument | No-interference conclusion is our conditional deduction |
| [Krichene and Rendle (2020)](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/), *On Sampled Metrics for Item Recommendation* | Sampling candidate sets can distort evaluation comparisons | Does not mandate a specific feasible catalogue size |
| [Mi, Lin and Faltings (2020)](https://arxiv.org/abs/2007.12000), *ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation* | Continual session recommendation with replay/distillation; user supplied this paper on 21 October | Identification in earlier meeting audio remains an inference; no team-wide reproduction choice established |
| [ADER authors' implementation](https://github.com/doublemul/ADER/tree/348fc2128bceb6bb3f3d8dd25ec8bbbae831af94) | Inspected SASRec, replay, training, evaluation and preprocessing code; see [protocol and discrepancies](ADER.md) | Local TensorFlow synthetic verification and two-window smoke passed with a recorded compatibility fix; no research result reproduced |

The “ADA” identification is an inference from the EPFL affiliation and similar-sounding dataset names. The matching datasets are e-commerce clickstreams. Do not report the transcript's “Digitec/YouTube” or its RNN recollection as verified paper facts. Ask the team to confirm its original link before choosing a reproduction target.

Follow-up: the supplied reference chat explicitly names ADER, so it is now a confirmed reference in Hongyi's material, while a team decision to reproduce it remains unconfirmed. The [full paper](https://arxiv.org/html/2007.12000v1), §3.2, uses next-window evaluation; our proposed fixed-A retention test differs. The other chat's full bibliography has not been exhaustively audited.

### ADER inspection on 21 October 2026

Read `ader-2007.12000v1.pdf` in full and visually checked Algorithms 1–2 and Eqs. 1–4 (pp. 4–5), Table 2 (p. 6), and Figure 2 / Tables 3–4 (p. 7). Inspected all seven Python files, README, both environment descriptions and license at the pinned authors' commit. The [replication protocol](ADER.md) distinguishes paper statements, released behavior, synthetic checks and proposed project changes, including validation overlap/reuse, filtered test contexts, replay underfill, an ER-loss bug and an inconsistent Table 4 metric. Published period files were inventoried from repository metadata but not acquired. All numerical paper targets remain external claims.

Additional sources checked for the neural workstream:

- [SASRec, Kang and McAuley (2018)](https://arxiv.org/abs/1808.09781): primary model reference. Our proposed domain-specific interfaces are an adaptation, not an original-paper claim.
- [Monolith, Liu et al. (2022), §2.2](https://ceur-ws.org/Vol-3303/paper8.pdf): initial batch training, streaming updates and synchronization to serving. A production example, not evidence of a universal deployment lifecycle.
- [CCTL, Zhang et al. (2023)](https://arxiv.org/abs/2306.16425): primary abstract supports negative-transfer concerns during cross-domain transfer/fine-tuning. It is background, not our selected implementation.

## SASRec walkthrough — 27 September 2026

- [Original paper on arXiv](https://arxiv.org/pdf/1808.09781), with local-only copy `local/sources/sasrec-2018.pdf`: Figure 1, p. 1; architecture and training in §III, pp. 3–5; experimental protocol/results in §IV, pp. 6–9. This is the research paper, not a course PDF.
- [Authors' model implementation](https://github.com/kang205/SASRec/blob/master/model.py) and [evaluation implementation](https://github.com/kang205/SASRec/blob/master/util.py): checked scoring, positive/negative loss, prediction versus optimizer operations, and sampled candidates.
- [Week 05: SASRec and causal attention](https://wing-nus.github.io/cp4285-website/slides/w05/w05.html#/sasrec-causal-self-attention-for-next-item-prediction): inspected the named SASRec slides and surrounding chronology material in the official HTML deck. There is no Week 05 PDF in the local slides folder.

Interpretation for the project: original SASRec studies next-item recommendation from histories; continued multi-domain updates and separate domain interfaces are our proposed additions. Its sampled-candidate benchmark scores are not directly comparable with full-catalogue evaluation.

## SASRec implementation selection — 28 September 2026

Hongyi requested the original [kang205/SASRec](https://github.com/kang205/SASRec) model. The implementation source is pinned to commit `e3738967fddab206d6eeb4fda433e7a7034dd8b1`; `model.py`, `modules.py`, `sampler.py`, `main.py` and the Apache-2.0 license were inspected. Our native PyTorch port replaces the generic encoder while retaining the original block equations and sequence-wise initial supervision. See [SASREC.md](SASREC.md) for immutable source links, attribution, verification and deviations from the legacy TensorFlow runtime and benchmark protocol.

### Chronological training periods — 25 October 2026

On training targets only, eight contiguous timestamp octiles each contain **2,434,850–2,434,852** examples. Equal target timestamps remain in one period. These are not equal-duration calendar windows: period 1 spans 1999-11-17 to 2014-08-18, and period 8 spans 2020-10-23 to the training cutoff on 2021-06-29. Per-period target/category counts and exact UTC boundaries appear in the runbook's comparison protocol.

Slurm job **1010792** completed in **45 seconds**, batch MaxRSS **7,654,884 KiB**, exit zero. Its 429,551,212-byte `train_periods.parquet` SHA-256 is `ea21da657fd12dd6f22daaf031e72ebfcdee62449e4d5ccc709500218e47b84e`. Job 1012758's manifest records the final exact-code hash and its report adds cumulative seen-category counts. Period target-category counts are 205, 212, 207, 199, 172, 148, 137, 119; cumulative target-label counts are 205, 233, 246, 257, 264, 267, 268, 271. Cumulative usable history-label counts are 216, 246, 259, 268, 275, 278, 279, 283. All candidate categories have been seen in targets or histories by period 8; 12 are never supervised targets. The local profile/report/manifest are ignored `local/category-period-{profile,report,manifest}-20261025.json`, with the final outputs carrying a `-v3-` suffix. Periods use no validation/test timestamps to choose cuts. Comparison settings and limits remain proposals, not neural results or a team-approved paper reproduction.
