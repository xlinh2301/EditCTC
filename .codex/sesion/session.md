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
