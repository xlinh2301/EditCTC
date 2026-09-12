# Session journal: EditCTC NERD

## 2026-09-10 — E34-A audit recovered

- Recovered Arbor session `editctc-nerd-lcb` and E34-A branch `agent/loop-e34-a`.
- E34-A evaluated 410 B_dev rows from the E33 checkpoint.
- Substitution oracle: Top-3 `12/16` token positions and `4/8` fully-fixable
  sequences; Top-5 `14/16` and `6/8` fully-fixable sequences.
- E34-A inference remains all KEEP: `changed/helped/hurt = 0/0/0`.
- Next action: resume Arbor coordinator and continue with E34-B using the
  candidate-utility/reranking hypothesis.

## 2026-09-10 — E34-B dispatched

- Native `arbor run --resume` could not start because the configured
  Anthropic provider has no `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`.
- Continued the durable loop manually: recorded E34-A as node `11`, created
  node `11.1`, and assigned isolated worktree/branch `agent/loop-e34-b`.
- Generated the executor prompt at
  `.arbor/sessions/editctc-nerd-lcb/experiments/11.1/executor_prompt.md`.
- E34-B executor is now running in the isolated worktree; B_dev only.

## 2026-09-10 — E34-B complete

- Branch `agent/loop-e34-b`, commit `37e9905`.
- Implemented `tools/e34b_ranker.py` with KEEP plus CTC Top-5 replacements,
  delta-edit-distance supervision, natural-error/hard-negative balancing, and
  an 80/20 B_dev split.
- Full B_dev: baseline `0.89512195`, E34-B `0.88536585`,
  `changed/helped/hurt = 14/1/5`; the `helped > hurt` pass gate fails.
- Held-out diagnostic: accuracy `0.85185185`, helped `1`, hurt `1`.
- No B_test was run.
- Report: `worktree/agent__loop-e34-b/results/11.1-keep-replace-ranker/results.md`.
- Metrics: `worktree/agent__loop-e34-b/results/11.1-keep-replace-ranker/ranker_metrics_0.05.json`.

## 2026-09-10 — E34-B2 queued

- E34-B failed with `helped=1, hurt=5`; do not spend another cycle on
  threshold tuning of posterior-only scores.
- Next hypothesis: visual-conditioned KEEP-vs-REPLACE ranking using pooled CTC
  frame-span features, candidate/seed embeddings, visual-candidate interaction,
  and risk-weighted listwise supervision.
- Required diagnostics: AUROC/AUPRC, good-candidate-over-KEEP rate,
  KEEP-over-harmful rate, plus helped/hurt on B_dev.
- Scope remains substitution-only; no DELETE, INSERT, LCB, or B_test.

## 2026-09-10 — E34-B2 dispatched

- Added Arbor node `11.2` under E34 and created isolated branch
  `agent/loop-e34-b2`.
- Executor prompt saved at
  `.arbor/sessions/editctc-nerd-lcb/experiments/11.2/executor_prompt.md`.
- Executor is running B_dev-only visual-conditioned ranking and diagnostics.

## 2026-09-10 — E34-B2 complete

- Branch `agent/loop-e34-b2`, commit `1b3d38d`.
- The branch added `tools/e34b2_visual_ranker.py` and ran a visual-span oracle
  probe with candidate/seed embeddings and interaction features.
- B_dev deployed path stayed at accuracy `0.89024390`,
  `changed/helped/hurt = 0/0/0` because the audit does not expose encoder span
  tensors for integrated inference.
- Probe diagnostics: all-data AUROC `0.3836`, AUPRC `0.0155`,
  good-candidate-over-KEEP `0.0`, KEEP-over-harmful `1.0`; heldout AUROC
  `0.5239`.
- E34-B2 fails the visual discrimination gate. Next experiment should expose
  pooled high-resolution encoder span features and rerun the probe; do not add
  threshold tuning, extra layers, DELETE, INSERT, LCB, or B_test yet.
- Report: `worktree/agent__loop-e34-b2/results/11.2-visual-conditioned-ranker/results.md`.
- Metrics: `worktree/agent__loop-e34-b2/results/11.2-visual-conditioned-ranker/visual_metrics.json`.

## 2026-09-10 — E34-B3 dispatched

- Added probe-only node `11.3` and isolated branch `agent/loop-e34-b3`.
- Prompt saved at `.arbor/sessions/editctc-nerd-lcb/experiments/11.3/executor_prompt.md`.
- Executor is exposing actual pooled high-resolution encoder span features and
  will report the visual oracle diagnostics on B_dev.

## 2026-09-10 — E34-B3 complete

- Branch `agent/loop-e34-b3`, commit `f71bf4b`.
- Exposed inclusive CTC token frame spans, mean pooled encoder features, and
  Top-5 nonblank candidate IDs/probabilities in a probe-only audit.
- B_dev deployed path: accuracy `0.89024390`, `changed/helped/hurt = 0/0/0`.
- Heldout probe: AUROC `0.7801`, AUPRC `0.0654`, good-over-KEEP `0.75`,
  KEEP-over-harmful `0.2549`, oracle top-1 beneficial rate `0.0`.
- Encoder spans contain useful signal, but KEEP protection is unsafe. Next step
  is a risk-aware visual scorer; do not use threshold-only tuning yet.
- Report: `worktree/agent__loop-e34-b3/results/11.3-encoder-span-probe/results.md`.
- Metrics: `worktree/agent__loop-e34-b3/results/11.3-encoder-span-probe/probe_metrics.json`.

## 2026-09-10 — E34-B4 dispatched

- Added node `11.4` and isolated branch `agent/loop-e34-b4`.
- Prompt saved at `.arbor/sessions/editctc-nerd-lcb/experiments/11.4/executor_prompt.md`.
- Executor is testing risk-aware visual scoring with B_dev only.

## 2026-09-10 — E35 pivot queued

- E34 is closed as a candidate/representation/safety diagnosis; no more
  posterior-only or loss-only ranker variants will be run on the 16 natural
  substitution tokens.
- Next direction is E35-A: build a Natural CTC Error Bank from out-of-sample
  predictions, preferring historical CTC checkpoint replay when K-fold OOF is
  too expensive.
- Keep positive CTC-wrong/Top-K-GT examples balanced with low-margin or
  high-entropy CTC-correct hard KEEP examples. Do not use random corruption as
  a substitute. B_test remains reserved.

## 2026-09-10 — E35-A dispatched

- Added Arbor node `12` and isolated branch `agent/loop-e35-a`.
- Prompt saved at `.arbor/sessions/editctc-nerd-lcb/experiments/12/executor_prompt.md`.
- Executor is building the natural error bank and will report its provenance,
  coverage, balance, and validation checks without training a ranker.

## 2026-09-10 — Parallel loop enabled

- Increased `coordinator.budget_policy.max_parallel_executors` from `1` to
  `2` in `research_config.yaml`.
- E35-A historical-checkpoint replay remains active. A separate OOF-pilot
  branch will use isolated artefact paths so the two bank-building approaches
  can run concurrently without overwriting each other.

## 2026-09-10 — E35-A OOF pilot dispatched

- Added independent Arbor node `13` and branch `agent/loop-e35-a-oof`.
- Prompt saved at `.arbor/sessions/editctc-nerd-lcb/experiments/13/executor_prompt.md`.
- Node `12` (historical replay) and node `13` (bounded OOF pilot) are now
  running in parallel with separate output paths.

## 2026-09-10 — E36-A1 parallel dispatch

- Increased the parallel executor limit to `3`.
- Added E36-A1 as node `14`: actual encoder span + candidate-aware contrastive
  loss with the last lightSVTR block unfrozen.
- E36-A1 runs representation preparation/probe in parallel with E35 bank
  construction, but scorer training/deployment remains gated on the selected
  E35 bank. A0/A2 remain closed until A1 evidence is available.

## 2026-09-10 — E37 verifier loop queued

- E37 is defined as Verifier-Guided Counterfactual Refinement: full-sequence
  candidate groups, an independent visual verifier, and selective abstention.
- The first runnable slice is E37-B/C: frozen CTC proposer plus preference
  verifier trained with `good > seed > harmful`; no RL or operation-specific
  DELETE/INSERT objective yet.
- E37 will run in parallel with E35 and E36 using the freed executor slot after
  the OOF Slurm job was submitted. It will use isolated artefacts and remain
  B_dev-only.

## 2026-09-10 — E34-B4 complete

- Branch `agent/loop-e34-b4`; based on the E34-B3 encoder-span exposure.
- Added `scripts/e34_b4_risk_scorer.py`: pooled encoder span features, KEEP
  plus CTC Top-5 candidates, listwise delta-ED supervision, and risk weight
  `2.0` on harmful replacements.
- B_dev deployed path remains baseline accuracy `0.89024390`, with
  `changed/helped/hurt = 0/0/0`; the scorer stayed probe-only because its
  integration gate failed.
- Heldout diagnostics: 383 groups, 4 positive groups, candidate
  `good_candidate_gt_KEEP_rate = 0.0`, `KEEP_gt_harmful_rate = 0.7092`, and
  oracle top-1 beneficial rate `0.0`. Full candidate AUROC `0.6532`, AUPRC
  `0.00585`.
- Risk weighting improves KEEP protection over E34-B3 but cannot rank a
  beneficial replacement above KEEP on heldout natural errors. No B_test was
  run.
- Report: `worktree/agent__loop-e34-b4/results/11.4-risk-aware-visual-scorer/results.md`.

## 2026-09-11 — E36/E37 first slices complete; E35 OOF fixed

- E36-A1 completed on `agent/loop-e36-a1` (commits `d2b622f`, `0403890`).
  Unfreezing the last lightSVTR block with contrastive preservation improved
  held-out `KEEP > harmful` only from `0.3269` to `0.4231` while
  `good > KEEP` fell from `1.0` to `0.6667`; the gate fails and no inference
  integration is allowed. Report: `results/14-e36-a1-contrastive/results.md`.
- E37-B/C completed on `agent/loop-e37-vgcr` (commit `87a6180`). The full
  sequence verifier built 410 counterfactual groups, but only four were
  wrong-to-correct; held-out `P(V_good > V_seed)=0.0` and
  `P(V_seed > V_harmful)=1.0`. It protects KEEP but collapses to abstention,
  so verifier deployment is gated. Report: `results/15-vgcr-verifier/results.md`.
- E35-A OOF pilot was invalidated after its first run: scratch training for
  one epoch produced out-of-vocabulary CTC strings and a multi-gigabyte audit
  log. The pilot script was fixed on `agent/loop-e35-a-oof` (commit
  `5601527`): 10-epoch fold training, optional verified generic pretraining
  only when mounted, explicit scratch fallback, invalid-string/zero-accuracy
  guards, and compact top-k posterior logging. Slurm job `70515` was stopped
  by the guard under `e35-a-oof-fixed`; its output is isolated from the
  invalid `e35-a-oof` artefact and remains B_test-free.
- The first fixed run (`70515`) correctly stopped at fold 0 because its
  10-epoch scratch model still decoded empty CTC strings. The available
  text-pretrained path is absent; an old full checkpoint references the
  removed `DATA/crop` set and is excluded to avoid leakage. A second bounded
  run (`70516`) therefore uses valid scratch OOF training with batch size 32
  and 20 epochs per fold, plus the same compact logging and contamination
  guards, under `e35-a-oof-fixed-v2`.
- V2 was stopped after log inspection showed `MultiScaleSampler.first_bs=128`
  still limited training to three updates per epoch despite the loader override.
  V3 (`70517`) sets both loader batch size and sampler `first_bs` to 32 with
  fixed batch sizing, using 10 epochs/fold (about 160 updates/fold), under
  `e35-a-oof-fixed-v3`.
- V3 reached 9 updates/epoch and sharply lowered CTC loss, but the first
  empty decode still aborted the fold. V4 (`70518`) changes this to row-level
  invalid-decode filtering with explicit invalid counts/rates, fails only when
  a fold has no valid rows, and trains 30 epochs/fold under
  `e35-a-oof-fixed-v4`.
- V4 completed with disjoint folds and compact audits. It retained 424 valid
  rows and filtered 88 invalid decodes; all 424 were CTC-wrong and there were
  no hard KEEP rows. Sequence audit found 285 substitution alignments, GT
  Top-3 `71/285` (24.9%), Top-5 `118/285` (41.4%), and fully-fixable
  sequences Top-3 `50`, Top-5 `91`. This is leakage-safe but fails the E35-A
  bank-quality gate, so E35-B must use the historical replay bank (285 rows)
  and the OOF artifact remains diagnostic only.

## 2026-09-11 — Next gated branches dispatched

- E35-B node `17`: visual KEEP-vs-REPLACE scorer on the selected historical
  bank, with held-out benefit/risk and fixed-harm gates.
- E36-B node `18`: candidate-query visual cross-attention probe using the same
  bank/split after E36-A1 failed.
- E37-A node `19`: counterfactual transition-bank densification before any
  verifier retraining, since E37-B/C had only four wrong-to-correct groups.
- All three run in isolated agent worktrees; no inference or B_test is allowed
  until their gates pass.

## 2026-09-11 — E35-B/E36-B/E37-A completed

- E35-B node `17` used the historical bank with an aligned image-span proxy.
  AUROC was `0.9714` and `KEEP > harmful = 0.9565`, but `good > KEEP = 0.50`,
  `helped = 0`, and `hurt = 2`; selective gate failed. Report:
  `worktree/agent__loop-e35-b/results/16-e35-b-visual/results.md`.
- E36-B node `18` tested candidate-query attention. It reached
  `KEEP > harmful = 0.9519`, but `good > KEEP = 0` and AUROC `0.1474`; gate
  failed and inference remains unchanged. Report:
  `worktree/agent__loop-e36-b/results/18-e36-b-candidate-query/results.md`.
- E37-A node `19` completed counterfactual densification with 695 groups,
  23,001 candidates, zero source overlap, and a balanced 280-row trajectory
  bank (70 rows per class). Taxonomy totals were wrong-to-correct `70`,
  wrong-to-better-wrong `159`, wrong-to-worse `2,107`, and correct-to-wrong
  `19,327`. Report:
  `worktree/agent__loop-e37-a/results/19-e37-a-counterfactual-bank/results.md`.
- No branch passed a deployment gate; no inference or B_test was run.

## 2026-09-11 — E37-B2 and E36-E dispatched

- E37-B2 node `20` retrains the independent sequence verifier on the balanced
  280-row E37-A transition bank; it remains probe-only until held-out
  `good > seed` and `seed > harmful` gates pass.
- E36-E node `21` runs the visual oracle with KEEP removed to test whether the
  actual encoder and candidate set can select GT at all.

## 2026-09-11 — E36-E oracle complete

- E36-E node `21` is diagnostic-only and fails visual selection: held-out
  candidate coverage was `10/12 = 0.8333`, visual token oracle accuracy
  `3/10 = 0.30`, fully-fixable sequences `1/3`, and full-sequence visual
  oracle `0/1`. This confirms the current span representation cannot safely
  choose GT even after removing the KEEP gate. Report:
  `worktree/agent__loop-e36-e/results/21-e36-e-oracle/results.md`.

## 2026-09-11 — E37-B2 verifier complete

- E37-B2 retrained the independent verifier on the balanced E37-A transition
  bank. It over-promoted edits: held-out `P(V_good > V_seed)=1.0`, but
  `P(V_seed > V_harmful)=0.0`, with `changed/helped/hurt=82/0/77` and zero
  edit precision. The densified bank alone does not solve verifier safety;
  inference remains locked. Report:
  `worktree/agent__loop-e37-b2/results/20-e37-b2-verifier/results.md`.

## 2026-09-11 — Representation and risk pivots dispatched

- E36-F node `22` tests candidate-aware local visual representation training
  after the E36-E oracle failed at 30% token accuracy.
- E37-C node `23` tests asymmetric risk-aware verifier training with explicit
  correct-to-wrong replay after E37-B2 over-edited (`82/0/77`).
- Both remain offline-only; inference and B_test stay locked.

## 2026-09-11 — E36-F and E37-C complete

- E36-F node `22` tested span-only and previous/current/next local context.
  Candidate coverage stayed `10/12 = 0.8333`; visual oracle token accuracy
  improved only `0.30 -> 0.40`, while sequence oracle stayed `0/1`. The
  representation gate fails. Report:
  `worktree/agent__loop-e36-f/results/22-e36-f-representation-v2/results.md`.
- E37-C node `23` tested risk weights `1,2,4,8` with harmful-edit replay.
  All variants had `P(good>seed)=1.0`, `P(seed>harmful)=0.0`, and
  `changed/helped/hurt=82/1/77`; edit precision was about `1.28%`. Risk loss
  does not repair the verifier score geometry. Report:
  `worktree/agent__loop-e37-c/results/22-e37-c-risk-verifier/results.md`.
- No inference integration or B_test is permitted after these failures.

## 2026-09-11 — E36-G final pooling diagnostic dispatched

- E36-G node `24` tests a high-resolution local crop oracle for aligned token
  spans. It is the final diagnostic before pivoting away from post-hoc
  substitution correction; inference and B_test remain locked.

## 2026-09-11 — E36-G complete; post-hoc substitution branch closed

- E36-G used raw high-resolution local crops with image-disjoint evaluation.
  Candidate coverage stayed `10/12 = 0.8333`; padding 2 gave visual token
  oracle `3/10 = 0.30`, while padding 3 reached `0.50`. Full-sequence oracle
  remained `0/1` for both. More local pixels add some token signal but do not
  make sequence correction identifiable. The E36 post-hoc substitution
  branch is therefore closed; no scorer/verifier inference integration is
  allowed.

## 2026-09-11 — Cross-data evaluation-only loop dispatched

- User requested cross-data testing because Indomain has few natural errors.
- Cross split: `1148` labels, `1145` evaluated (`3` missing images). Existing
  checkpoint-only audit is used; no training or fitting on Cross-data.
- Parallel audits dispatched:
  - Arbor node `27`: candidate/error taxonomy and E34/E35-style Top-K audit.
  - Arbor node `25`: E36 span/raw-crop visual oracle comparison.
  - Arbor node `26`: E37 counterfactual sequence-transition audit.
- Compare against existing Indomain test audit (`562` evaluated) before changing
  the roadmap. Append artifacts and cross-vs-Indomain conclusions when reports
  complete.

## 2026-09-11 — Cross-data audits complete; roadmap updated

- Candidate/error audit (`results/cross-audit-candidate/report.md`): Cross CTC
  accuracy is `91.79%` (`94/1145` wrong) versus Indomain `95.73%` (`24/562`).
  Cross errors are dominated by insertion/length failures: `52` insertion-only,
  `12` deletion-only, `4` mixed-length, and `26` substitution-only, compared
  with `7/4/0/13` on Indomain. Existing NERD is KEEP-only on both (`0/0/0`).
  Wrong rows remain partly high-confidence (mean CTC confidence `0.976`, mean
  minimum frame margin `0.793`). Persisted branch logs lack Top-3/Top-5 IDs.
- Sequence counterfactual audit (`results/cross-audit-sequence/report.md`):
  using persisted NRTR and NERD proposals, exact candidate coverage on wrong
  rows is `13.8%` Cross vs `20.8%` Indomain; beneficial oracle coverage is
  `20.2%` vs `29.2%`; wrong-to-correct candidates are `13` vs `5`; harmful
  correct-to-worse proposals are `376` vs `82`. Cross supplies more absolute
  errors but a weaker current proposal space and a much larger abstention-risk
  surface.
- E36 visual audit (`results/cross-audit-visual/report.md`) is explicitly
  blocked by the persisted schema: no encoder spans or frame Top-K identities
  were stored, so a valid cross visual oracle cannot be claimed. The confidence
  proxy is not a substitute for visual discrimination. No Cross-data training,
  inference checkpoint update, or B_test was performed.

### Cross-data decision

The central E34–E37 diagnosis does not change; it is stronger. Indomain error
scarcity was real, but Cross exposes predominantly length-changing errors where
the current substitution candidates are unavailable. The next experiment must
be an evaluation-only fresh dump that records full CTC frame Top-K identities,
collapsed spans, and encoder features on Cross and Indomain using the same
checkpoint. Then measure length-aware full-sequence candidate oracle before any
new training. If candidate coverage remains low, fix proposal generation first;
do not train another verifier on a candidate set that cannot contain the answer.
If coverage is adequate, train only on non-Cross data and evaluate the verifier
on Cross as a locked external test, with `helped > hurt` and correction recall at
  fixed harm rates as gates.

## 2026-09-11 — E40 rendered visual pretraining started

- User proposed learning edit mechanics from hundreds of thousands of rule
  samples and then adding synthetic images before real-pipeline integration.
- This revises the earlier E31/E32 lesson: E31 text-only pretraining generated
  `22,158` rows but produced zero real-image edits; E32 visual seed corruption
  also produced zero B_dev edits. Raw text/rule seeds are therefore mechanics
  bootstrap only, not transfer evidence.
- Added `tools/generate_visual_edit_corpus.py` and OpenSpec
  `openspec/changes/e40-rendered-visual-edit-pretrain/`. It renders clean GT
  meter-like images at `[3,48,320]`, creates confusion-aware
  KEEP/REPLACE/INSERT/DELETE seeds separately, and records a JSONL manifest.
- A `200,000`-row corpus generation job is running under
  `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_synth/visual_edit_v1`.
  Frozen CTC inference must filter rows to authentic CTC errors/ambiguities
  before NERD transfer. Cross-data remains evaluation-only.
- Arbor node `28` tracks E40. No model checkpoint has been changed yet.
- Added `tools/pretrain_editrefine_text.py` for Stage-0 mechanics pretraining.
  It uses the existing `EditRefineDecoder` with null visual memory, balanced
  operation weights, and saves only `head.edit_refine_head.*` tensors for later
  visual loading. The script is syntax-checked but not run locally because the
  CPU environment lacks Paddle's CUDA runtime; GPU execution belongs on the
  configured training node after the corpus is complete.
- Added `tools/filter_ctc_synth_manifest.py`: after frozen-CTC inference, it
  replaces rule seeds with actual CTC seeds and retains natural errors plus
  low-margin hard KEEP rows. This is the required bridge from synthetic
  mechanics to visual transfer; raw rule rows are never treated as deployment
  evidence.
- Added `tools/pretrain_editrefine_visual.py` for Stage-1 GPU pretraining. It
  freezes the backbone/CTC path, passes the manifest seed explicitly into the
  existing visual EditRefineDecoder, optionally loads Stage-0 text tensors, and
  saves only edit-head weights. It has not been run yet; the local environment
  cannot load Paddle CUDA.
- Submitted GPU jobs `70647` (text mechanics) and `70648` (visual synthetic).
  Job `70648` waits for both the corpus summary and the Stage-0 text checkpoint,
  so visual training cannot start prematurely. Job `70647` failed immediately
  after the corpus became ready because the standalone script did not put the
  repo root on `sys.path` (`ModuleNotFoundError: ppocr`); this was fixed in
  commit `8a848c5` and text job `70652` was resubmitted. The visual job remains
  waiting for the corrected text checkpoint.
- Text job now logs held-out synthetic non-KEEP recall, KEEP precision, and
  token-target accuracy each epoch; a falling loss alone is not considered a
  successful edit-pretraining gate.
- Retry `70652` completed successfully after the module-path fix, and visual
  job `70648` completed successfully. Checkpoints are stored under
  `.../e40/text_mechanics/editrefine_pretrained.pdparams` and
  `.../e40/visual_synthetic/editrefine_visual_pretrained.pdparams`.
- Added `tools/eval_editrefine_visual.py` and submitted validation job `70698`
  for explicit-seed synthetic `helped/hurt`, operation recall, and KEEP
  precision. This remains pre-integration; real Indomain/Cross evaluation must
  use natural CTC seeds afterward.
- E40 visual synthetic evaluation job `70698` initially reported
  `changed=3985`, `helped=58`, `hurt=954`, but that run loaded the edit-only
  checkpoint as if it were the full model and therefore left the CTC/backbone
  random. Its metrics are invalid and must not be used as a gate.
- Fixed the evaluator in commit `4009a29` to load the frozen s1024 CTC
  checkpoint first and the visual edit tensors second. Corrected validation job
  `70701` completed successfully. On 4,000 held-out rendered rows it obtained
  `wrong_seed=3,039`, `seed_exact=961`, `changed=3,040`, `helped=2,678`,
  `hurt=6`, `nonkeep_recall=0.9806`, `keep_precision=0.9975`, and
  `token_accuracy=0.9689`. This passes the synthetic mechanics/visual gate,
  but is not deployment evidence because the seed is explicit/rule-corrupted.
- Added additive edit-head loading to `tools/infer_rec.py` in commit
  `9010cba`, preserving the complete frozen CTC checkpoint and replacing only
  matching `head.edit_refine_head.*` tensors. The two-set evaluation launcher
  now accepts `E40_OUT_ROOT` and `E40_EDIT_PRETRAINED` overrides.
- Submitted real-pipeline transfer evaluation job `70702` for both Indomain and
  Cross (evaluation-only; no Cross training and no B_test). Its output root is
  `Data/EditCTC_eval_s1024/e40_visual_transfer`.
- Job `70702` completed and fails the real-data gate catastrophically. Cross:
  CTC `0.9179`, E40 final `0.1135`, `changed=1012`, `helped=3`, `hurt=924`.
  Indomain: CTC `0.9573`, E40 final `0.2883`, `changed=401`, `helped=1`,
  `hurt=377`. Offline operation-confidence thresholds up to `0.99` still had
  `hurt > helped`, so this is not a simple threshold-calibration issue.
- Interpretation: E40 learned the edit mechanics and visual synthetic task,
  but the synthetic rendered visual/seed distribution does not transfer to
  natural CTC spans. The head edits even CTC-correct real strings, so the
  checkpoint is explicitly rejected for deployment. Arbor node `28` is done
  with score `0`.
- Next controlled step is E40-B/E35 bridge: run frozen base CTC on non-Cross
  training images, build a natural-CTC/low-margin hard-KEEP bank, then adapt
  only the edit head with the synthetic checkpoint as initialization and hold
  Cross locked for evaluation. Add a selective verifier/gate only after this
  bank produces `helped > hurt` on held-out Indomain.
- Submitted natural-bank dump job `70745`, which exposed a stale config
  override pointing to a deleted text-pretrain path; no inference data was
  produced. The launcher now explicitly sets
  `Global.edit_refine_pretrained=None` for the frozen CTC dump (commit
  `e6fe708`) and was resubmitted as job `70746`.
- Corrected dump job `70747` completed over all 2,462 Indomain train images.
  `build_natural_ctc_manifest.py` retained 41 authentic wrong seeds and 706
  low-confidence hard KEEP rows (686 train / 61 held-out val); no Cross rows
  are present.
- Natural adaptation job `70748` initialized from E40 visual weights, repeated
  natural errors 10x (roughly 1:2 against hard KEEP), and trained only the edit
  head for 8 epochs (`loss 1.4117 -> 0.2614`). Transfer job `70750` improved
  final accuracy but still failed safety: Cross `24.63%`, `helped=10`,
  `hurt=779`; Indomain `44.13%`, `helped=2`, `hurt=292`. This confirms the
  natural bank moves the model toward the right distribution but is too small
  to learn a safe correction boundary. The separate held-out evaluator
  `70749` was canceled after hanging without output; it is not used as a gate.
- Offline replay of job `70750` with only `REPLACE` operations allowed reduced
  damage substantially (Cross `103` hurt / `1` helped; Indomain `34` hurt /
  `1` helped), while the full operation set produced `779`/`292` hurt. This
  isolates INSERT/DELETE as the dominant over-edit source, but replacement
  discrimination is still absent. The next branch stays substitution-only
  with explicit CTC Top-K candidates and a KEEP gate; DELETE/INSERT remain
  disabled for synthetic-head transfer.
- Implemented the eval-only substitution mask for both `MultiHeadEditRefine`
  and `MultiHeadEditRefineUncertainty` (the first `70752` run was invalid
  because the uncertainty path had not yet received the mask). Correct rerun
  `70754` gives Cross `65.76%`, `changed=343`, `helped=3`, `hurt=301`, and
  Indomain `79.72%`, `changed=101`, `helped=1`, `hurt=91`. This confirms the
  operation mask removes much of the length-edit damage, but REPLACE token
  discrimination still fails `helped > hurt` by a wide margin.
- E40 is therefore closed as a mechanics/pretraining probe, not a deployable
  correction model. The next implementation must expose full CTC frame Top-K
  identities and collapsed token spans, then train a substitution-only
  candidate scorer/verifier with KEEP always present; no synthetic INSERT or
  DELETE transfer will be connected to inference.
-
## 2026-09-12 — E41 integrated NRTR EditCTC prototype

- User requested replacing the separate EditCTC decoder with an NRTR-pretrained
  decoder that consumes feature-map memory, the original image, and the CTC
  seed, while removing the standalone NRTR loss branch.
- Added `MultiHeadEditRefineNRTR`: CTC remains the seed path; the pretrained
  `gtc_head` Transformer is reused under the same checkpoint keys, receives
  CTC seed ids mapped into NRTR vocabulary, and cross-attends to concatenated
  CTC memory plus compact raw-image tokens. Its hidden states feed only edit-op
  and replacement-token heads. The training output contains CTC, length, and
  edit losses; no `gtc` output or `NRTRLoss` is used.
- `BaseModel` now passes the input image to heads that opt into
  `use_original_image`. New config:
  `config/PP-OCRv6_small_rec_s1024_nrtr_integrated.yml`.
- Smoke job `70800` exposed missing output-channel registration and retry
  `70801` exposed a Linear vocab-axis initialization bug; both were superseded.
  These were fixed in commits `9cf6127` and `bb4d7fb`. Retry `70802` then
  exposed the launcher passing the string `None` as `Global.checkpoints`; the
  smoke launcher now leaves that field unset and job `70803` is the clean retry.
  No production checkpoint has been changed.
- Arbor loop was reset to the E41-E44 improvement tree. Nodes `29`–`32` are
  registered with isolated prompts: integrated NRTR architecture, Top-K/span
  instrumentation, larger Indomain bridge bank, and conditional verifier.
  Node `29` is done after job `70803`: architecture loaded and trained one
  epoch with finite losses (`EditActivationRate≈0.11`), validation accuracy
  `0.8854` versus `0.8902` baseline. Nodes `30`–`32` remain pending behind
  their stated coverage/data gates.
- E41 long run job `70822` completed all 30 epochs at
  `Data/EditCTC_arbor_runs/editctc-nerd-lcb/e41/nrtr_integrated_30ep`.
  Best Indomain validation was epoch 26: `acc=0.8853658321`,
  `norm_edit_dis=0.9658536594`, versus immutable CTC baseline `0.89024390`.
  Training CTC accuracy reached `0.98–1.00`, while
  `EditPredChangeRate/Replace/Delete/InsertRate` stayed `0.0` throughout the
  late run. The integrated NRTR head therefore trains and remains numerically
  stable, but 30 epochs do not produce useful edits and slightly trail the
  baseline; this is an architecture/training probe, not a deployable model.
  Best checkpoint is `.../e41/nrtr_integrated_30ep/best_accuracy` and Cross
  data remains evaluation-only.
- To address E41 overfit, added `CTCSpanBlurAug` plus
  `tools/build_ctc_span_index.py`. Job `70826` built a leakage-free frozen
  CTC span index for all 2,462 Indomain train images (mean 4.56 collapsed
  spans/image); the transform samples one span online and applies a local
  Gaussian blur with clean examples retained. A first 70827 launch exposed a
  malformed-index reader bug and was stopped; the corrected smoke job `70829`
  ran cleanly and showed a real perturbation (`EditPredChangeRate=0.746` on
  the first batch before the head reverted to KEEP).
- Joint E41-B run `70830` was stopped after validation collapsed to `0.3122`
  at epoch 6, confirming that allowing the shared CTC/backbone to adapt to
  50% blurred images causes clean-data overfit. Added `freeze_ctc_backbone`
  in `tools/train.py` and relaunched frozen correction/image-head training as
  `70831` (`e41/nrtr_blur_online_frozen`). At epoch 6 its Indomain validation
  is `0.8926829`, above the `0.8902439` CTC baseline, with stable CTC loss;
  edit change rate is still near zero after warm-up, so correction utility is
  not established yet. A separate blur-effect audit is queued for the next
  free GPU to measure `clean-correct -> blur-wrong` directly.
- User requested `EditCTC` loss weight `1.0`; preserved the completed `weight_edit=0.15` run and created config/launcher variant `PP-OCRv6_small_rec_s1024_nrtr_blur_online_w1.yml` / `scripts/e41_nrtr_blur_online_w1_train.sh`. This keeps targeted CTC-span blur (`prob=0.50`, `min_span_fraction=0.06`) and `freeze_ctc_backbone=true`, changing only `Loss.weight_edit: 0.15 -> 1.0`; output is `.../e41/nrtr_blur_online_frozen_w1`. Job `70836` is running. At early steps the edit loss is initially high but the edit head has already moved toward KEEP by step 30; CTC remains stable and no collapse is visible yet. Continue monitoring through validation before judging the weight change.
- E41-B `weight_edit=1.0` run `70836` completed cleanly after 30 epochs at `.../e41/nrtr_blur_online_frozen_w1`. Best Indomain validation is epoch 26: `acc=0.8926829051`, `norm_edit_dis=0.9671544723`; this matches the best accuracy of the `weight_edit=0.15` frozen-blur run (`0.8926829051`) and is only a marginally higher normalized edit distance. Epoch 6/11 were lower (`0.8878048564`), epoch 21 returned exactly to the CTC baseline (`0.8902438807`). CTC loss stayed stable because the CTC/backbone path was frozen. `EditPredChangeRate`, `EditPredReplaceRate`, `EditPredDeleteRate`, and `EditPredInsertRate` became zero after the first warm-up batches and stayed zero through epoch 30. Conclusion: increasing edit loss weight alone does not induce useful edits; the model chooses the safe KEEP solution because targeted blur creates only about 2.18% clean-correct-to-blur-wrong cases and does not provide enough explicit positive correction supervision. Arbor node `29.1` is now done with score `0.8926829051`; Cross remains untouched and evaluation-only.
- Read-only test inference for E41-B `weight_edit=1.0` best checkpoint (`.../e41/nrtr_blur_online_frozen_w1/best_accuracy`) completed on jobs `70840` (Indomain test) and `70841` (Cross). The integrated `MultiHeadEditRefineNRTR` head does not currently return `branch_debug`, so the wrapper parser initially failed after inference; predictions were complete and final accuracy was recomputed directly, and `scripts/e41_eval_checkpoint_job.sh` was patched to support final-only summaries (commit `8940b11`). Indomain test: `540/562 = 0.9608540925`, versus persisted CTC baseline `0.9573` on the same split. Cross: `1037/1145 = 0.9056768559`, with 3 missing labeled files (`cross_00090.jpg`, `cross_00333.jpg`, `cross_00613.jpg`) excluded; persisted CTC baseline is `1051/1145 = 0.9179039301`. Thus the w1 integrated model is slightly higher on Indomain test but `-0.012227` absolute below CTC on Cross. No Cross row entered training. Branch-level helped/hurt cannot be reported for this head until debug outputs are added; training logs still show zero edit rates after warm-up.
- Added inference-only `branch_debug` to `MultiHeadEditRefineNRTR` (commit `edd41c4`) and reran tests on jobs `70843`/`70844`. Aggregated branch audit (commit `f0b0c00`): Indomain test `562` evaluated, CTC/seed/final all `540` correct (`0.9608540925`), `changed=0`, `helped=0`, `hurt=0`, operation counts KEEP `2579`, REPLACE/DELETE/INSERT `0`. Cross `1145` evaluated after 3 missing files, CTC/seed/final all `1037` correct (`0.9056768559`), `changed=0`, `helped=0`, `hurt=0`, KEEP `5825`, all other operations `0`. The integrated EditCTC output is exactly the frozen CTC seed on both test splits: perfect KEEP protection but zero correction utility. `nrtr` branch is intentionally absent for this integrated head and is not used in these metrics. Arbor node `29.1` was updated with the branch-level result.
- Architecture analysis after branch-debug audit: current per-token 4-way `{KEEP, REPLACE, DELETE, INSERT_AFTER}` head is structurally imbalanced. Real CTC errors are rare (targeted blur induced only about 2.18% clean-correct-to-blur-wrong cases), so `weight_edit=1.0` scales an overwhelmingly KEEP-heavy loss and does not create positive edits. Recommended replacement is a factorized residual editor: a balanced binary correction gate (`EDIT?`) plus a replacement-token head trained only on guaranteed wrong/visual-corruption positions, with clean low-margin CTC positions as hard KEEP negatives; start substitution-only and add a sequence verifier or edit budget for safety. Candidate listwise ranking and full-sequence denoising are secondary alternatives; DELETE/INSERT should wait until substitution has nonzero helped/hurt evidence.
- E42 factorized replacement experiment implemented and run. `MultiHeadEditRefineNRTR` now supports `edit_mode=factorized`: binary KEEP/REPLACE gate, replacement token head, and training-only one-token seed corruption (`train_seed_corrupt_prob=0.80`); `EditLossFactorized` uses positive-weighted gate CE and token CE only at replacement targets. Smoke job `70848` confirmed nonzero replacement gradients and `EditPredReplaceRate` falling from 0.50 to 0.24 in one epoch. Full 20-epoch job `70849` completed; edit activity stayed about 0.18 but best dev remained baseline `acc=0.8902438807`, `norm_edit_dis=0.9658130`. Test jobs `70850`/`70851` branch audit: Indomain `541/562=0.962633`, changed=0/helped=0/hurt=0; Cross `1042/1145=0.910044`, changed=1/helped=0/hurt=0. Factorized gate fixes training imbalance but synthetic seed corruption does not transfer to natural visual CTC errors; no pass. Arbor node `29.2` is done.

## 2026-09-12 — Direct token-refinement alternative (E42-token)

Because the binary/four-way KEEP gate still collapsed at test time, added a distinct edit mechanism: `edit_mode: token_refine`. It removes the imbalanced EDIT/KEEP CE from the decision path. The replacement token head is trained on every aligned KEEP/REPLACE slot (with optional seed denoising corruption), then inference edits only when the token head's argmax differs from the seed and passes probability and probability-delta thresholds. Added `EditLossTokenRefine`, `MultiLossEditRefineToken`, config `PP-OCRv6_small_rec_s1024_token_refine.yml`, and `scripts/e42_token_refine_train.sh`; committed as `1ce1342`. Job 70854 submitted. E42 factorized-visual job 70852 remains under monitoring.

## 2026-09-12 — E42 visual and direct token refinement results

- E42 factorized visual (`prob=1.0`, `min_span_fraction=.10`, gate weight 8) completed job 70852. Best dev `0.8878048564` (below baseline `0.8902438807`). Correct branch-debug Cross rerun job 70859 (the first attempt used a wrong label path and had zero rows): Cross `1036/1145=0.9048035`, seed/final equal, changed=0/helped=0/hurt=0. Indomain job 70855: `541/562=0.9626335`, changed=0/helped=0/hurt=0. Augmentation and larger gate weight do not induce useful deployment edits.
- Added E42-token direct denoising editor (`edit_mode=token_refine`): no binary KEEP/EDIT loss; replacement token head is trained on all aligned KEEP/REPLACE slots with 70% one-token seed corruption; inference uses token confidence and token-vs-seed probability margin. Initial job 70854 failed validation due an undefined `seed_ids` reference; fixed and committed `152c990`, retry job 70857 completed 20 epochs. Best dev is exactly baseline `0.8902438807` at epoch 11.
- Token retry test jobs 70861/70862: Indomain `541/562=0.9626335`, changed=3/helped=0/hurt=0; Cross `1037/1145=0.9056769`, changed=3/helped=0/hurt=1. The new mechanism does perform edits, but all observed changes are incorrect or neutral; it does not exceed CTC and fails the `helped>hurt`/accuracy gate. Therefore direct denoising removes the imbalance symptom but confirms the deeper bottleneck is visual discrimination/transfer from synthetic seed corruption to natural CTC errors.

## 2026-09-12 — E42-token-CTC-alt and threshold audit closed

- Added CTC Top-2 alternative corruption (`train_seed_corrupt_mode=ctc_alt`) so direct token refinement sees posterior-neighbor confusions instead of random digits. Jobs were initially canceled twice to correct output/config paths; final job 70865 used isolated `e42/token_ctc_alt` output and completed 20 epochs.
- Best dev was `0.8804877834` at epoch 16, below baseline (`0.8902438807`). Test jobs 70869/70870: Indomain CTC `542/562`, final `541/562`, changed=5/helped=1/hurt=2; Cross CTC `1039/1145`, final `1033/1145`, changed=20/helped=2/hurt=8. Edit is active, but `helped<hurt` and accuracy drops on both final outputs. Threshold sweep for direct token mode on dev (`.2/.3/.4`) produced changed=0 and baseline accuracy, so threshold alone is not the bottleneck.
- E42 alternatives therefore closed: factorized gate, visual blur gate, direct token denoising, and CTC Top-2 denoising all fail the required test accuracy/helpful-edit gate. The recurring failure is visual transfer/discrimination on natural CTC mistakes, not only KEEP class imbalance.

## 2026-09-12 — E43 natural cached seed adaptation

- Added `tools/adapt_nrtr_token_natural.py` and `scripts/e43_natural_cached_token_train.sh`. It trains only integrated editor token/op tensors with explicit cached Indomain natural CTC seeds; 32 natural error rows are repeated 20x alongside 706 hard KEEP rows. Cross is never read for training. Additive editor loading was added to `scripts/e41_eval_checkpoint_job.sh` (commit `8b4d5c2`).
- E43 test: Indomain CTC 541/562 -> final 540/562, changed=3, helped=0, hurt=1. Cross CTC 1038/1145 -> final 1040/1145, changed=8, helped=3, hurt=1. This is the first branch with genuine positive correction signal, but still fails the absolute baseline gate (persistent Cross baseline 1051/1145) and harms Indomain.
- Added a CTC Top-2 candidate-only safety gate (commit `40dc06a`). On E43 Cross it reduced changed edits 8 -> 6, retained all 3 helped and the single hurt, final remained 1040/1145; on dev it changed nothing. Candidate restriction improves precision of neutral edits but does not solve safety/generalization.

## 2026-09-12 — E43-B same-length natural substitution isolation

- Filtered natural adaptation to the 12 same-length Indomain substitution errors (30x oversampled) plus hard KEEP, excluding length-changing rows from token loss. Job 70879 completed with loss `0.474 -> 0.268`.
- Test jobs 70880/70881: Indomain `541 -> 540`, changed=4/helped=0/hurt=1; Cross `1038 -> 1040`, changed=9/helped=3/hurt=1. Metrics are essentially identical to E43 full natural adaptation, so length-noise filtering alone is insufficient. Natural supervision is promising for edit activity but the available bank is too small and not representative enough to beat baseline.

## 2026-09-12 — Current architecture snapshot

- Current model is `PPLCNetV4-small` over `[3,48,320]`, followed by a CTC head with a 2-block LightSVTR neck (`dims=120`). The CTC output produces posterior, greedy collapsed seed, margin, and optional Top-2 alternative.
- `MultiHeadEditRefineNRTR` integrates a 4-layer, 384-dimensional NRTR decoder into the EditCTC head; there is no separate NRTR loss/output branch. Decoder memory concatenates projected CTC sequence features with 80 tokens from a small raw-image Conv2D path, then conditions on `[BOS + CTC seed tokens]`.
- In the active `token_refine` path, the token head predicts a token at every seed slot. Inference applies `KEEP/REPLACE` implicitly: replace only when argmax differs from seed and passes probability (`>=0.5`), probability-delta (`>=0.05`), and optional Top-2 candidate gates. DELETE/INSERT are not active in this path. The physical 4-way op head remains for compatibility but is unused by token-refine inference; the length head is auxiliary and does not gate edits.
- E43 is an additive natural-seed editor checkpoint loaded on top of the E42 token-refine model. It trains editor heads only using cached natural CTC seeds plus hard KEEP examples; backbone, CTC, and integrated NRTR decoder remain frozen.

## 2026-09-12 — CNN–Transformer input research direction

- Literature review points to the input representation, not another KEEP/EDIT loss, as the main architectural gap. Original NRTR uses a shallow modality-transform block to map 2D image features to a sequence and warns that excessive convolutional subsampling loses detail. SATRN keeps a 2D feature map with adaptive 2D positional encoding before Transformer attention. SVTRv2 preserves higher spatial resolution and uses feature rearrangement to satisfy CTC alignment. MATRN and ABINet explicitly exchange visual and semantic features instead of blindly concatenating them. PARSeq uses position queries, cross-attention and permutation/iterative refinement; CSD-2025 reports gains from context-aware cloze decoding and differential cross-attention.
- Current integrated head instead concatenates collapsed LightSVTR CTC memory with an independently initialized raw-image path that immediately applies `AdaptiveAvgPool2D((1,80))`; it has no modality/type embedding or explicit 2D positional encoding. The NRTR query is only `[BOS + greedy CTC seed]`, so the editor is strongly anchored to the same CTC evidence that made the error.
- Proposed next architecture: shared CNN multi-scale features → lightweight 2D local/global Transformer visual encoder with row/column positional embeddings → visual memory; seed tokens plus CTC margin/entropy/Top-2 features become parallel position queries; decoder cross-attends to typed CTC memory and high-resolution visual memory. Keep CTC unchanged initially, freeze it, train only new visual projection/decoder/editor, then unfreeze the last backbone block. This is the planned E44 input redesign before further loss or operation-space changes.

## 2026-09-12 — E44-A implementation and gates

- Implemented `SharedHighResVisualMemory`: reuses PPLCNetV4 `recon_feat` (`[B,384,3,80]`) before height collapse, adds depthwise local mixing, row/column 2D positional embeddings, a visual modality embedding, and a lightweight self-attention block. CTC memory now receives a separate CTC modality embedding; the old raw-image pooled branch remains as fallback when E44 is disabled.
- Added weak-reference backbone wiring in `BaseModel`. Directly storing a Paddle Layer in the head duplicated backbone parameters and state-dict keys; the weak reference removes that issue. Added config `PP-OCRv6_small_rec_s1024_e44_highres.yml`, unit tests, GPU unit-test job, smoke job, and full training job.
- Local static tests passed (`3 passed, 2 Paddle runtime tests skipped`); GPU unit-test jobs `70887` and `70898` passed all `5/5`. GPU smoke job `70888` passed one epoch with finite losses and no duplicate-backbone load warnings. A dynamic-width interpolation path was added after evaluation exposed test widths 98/102 versus the training grid width 96.
- Full E44-A training job `70889` completed 30 epochs with frozen CTC/backbone. Best validation was epoch 11, `acc=0.8926829051`, `norm_edit_dis=0.9685772365`, slightly above the immutable validation baseline `0.8902438807`; edit predictions were active during training.
- Read-only E44-A test audits `70899` (Indomain) and `70900` (Cross) completed. Indomain: CTC `537/562=.955516`, final `535/562=.951957`, `changed=5`, `helped=1`, `hurt=3`. Cross: CTC `1036/1145=.904803` (3 missing images excluded), final `1033/1145=.902183`, `changed=9`, `helped=2`, `hurt=5`. Both fail the required `helped>hurt` and accuracy gates.
- Baseline sanity audits `70904`/`70905` using the exact E44 config and the original frozen checkpoint reproduced the persisted baseline CTC outputs: Indomain `538/562=.957295`, `changed=0`; Cross `1051/1145=.917904`, with the same 3 missing files and only one harmful edit from the integrated head. Therefore the E44 checkpoint's CTC path itself lost `1` Indomain and `15` Cross correct samples relative to baseline, despite being configured frozen; the E44 final branch additionally lost `2`/`3` samples versus its own CTC. The high-resolution memory is technically valid but does not yet provide safe correction evidence. Before another architecture variant, investigate why the E44 training checkpoint's supposedly frozen CTC outputs differ from the source checkpoint, then pivot to an independent verifier or natural-error adaptation rather than more threshold tuning.
