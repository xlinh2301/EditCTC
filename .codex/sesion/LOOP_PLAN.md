# EditCTC gated experimental loop: E35-E39

## Current state

E34 established candidate availability and partial encoder visual signal, but
safe selective correction failed. E35-A is the active data-distribution pivot.

## Gates and order

1. **E35-A: Natural CTC Error Bank**
   Compare historical checkpoint replay with OOF. For each bank record natural
   errors, fixable substitutions, hard KEEP negatives, Top-1/3/5 GT coverage,
   confidence/margin/entropy, checkpoint/fold provenance, and leakage checks.
   Select the bank closest to deployment with enough positives. Replay may be
   pretrain data and OOF validation data when OOF is sparse.
2. **E35-B: visual candidate scorer**
   KEEP + seed + CTC Top-K replacement candidates, actual encoder span
   features, candidate/seed embeddings, and posterior metadata. Use pairwise
   supervision on natural fixable errors and hard KEEP negatives (1:1 or 1:2).
   Offline gate: `good > KEEP > 50%` and `KEEP > harmful > 95%`. No inference
   deployment if either fails.
3. **E35-C: selective correction**
   Sweep `delta_score = best_edit - KEEP` and report correction recall at harm
   rates 0%, 0.5%, 1%, 2%, and 5%, plus helped/hurt/changed/edit precision.
   Main gate: `helped > 0`, `helped > hurt`, and KEEP protection near 95%.
4. **E36-A** only if E35-B/C fail: candidate-aware contrastive encoder
   fine-tuning on real confused digit pairs, then retry the same E35-B scorer.
5. **E36-B** only if E36-A fails: compare mean/max pooling, attention pooling,
   and candidate-query cross-attention over local span features. Stop the
   substitution direction if the oracle still cannot discriminate.
6. **E37** only after substitution passes: candidate-formulation DELETE/INSERT
   with an oracle operation-coverage audit first.
7. **E38** only after DELETE/INSERT works: soft LCB or direct `delta_length`
   candidate feature, then length consistency; require rescue without extra
   hurt on CTC-correct cases.
8. **E39** only after single-pass safety: two-step re-align/refine; keep it only
   if step 2 improves fix rate without increasing harm.

## E37 verifier-guided counterfactual refinement

E37 is an independent correction judge, not another token ranker. Build groups
`{seed, proposal_1, ..., proposal_K}` and label each full transcription by
`delta_ED`, exact match, and transition type (`wrong->correct`,
`wrong->better-wrong`, `wrong->wrong`, `wrong->worse`, `correct->correct`,
`correct->wrong`). Train a frozen-CTC sequence visual verifier with preference
`good > seed > harmful`, then add explicit correction reward and balanced
trajectory sampling only after the verifier has signal. Inference is
`propose -> verify -> abstain` with KEEP as candidate zero.

E37 offline gates are `P(V_good > V_seed) > 70%` and
`P(V_seed > V_harmful) > 95%`; deployment still requires `helped > hurt` and
correction recall at harm rates 0.5%, 1%, and 2%. Do not start RL, LCB, or
DELETE/INSERT-specific objectives before the supervised verifier passes.

## Stop rule

Each hypothesis family gets at most 2-3 controlled variants. An oracle-positive
family that still cannot achieve `helped > hurt` after that limit must pivot to
the next bottleneck instead of accumulating more weight/loss sweeps.

## Parallel policy

At most three independent executors may run concurrently. E35-A bank building
and E36-A1 representation preparation may overlap; E36 scorer training must
consume a selected E35 bank. Dependent stages stay gated: E35-B waits for bank
selection; E36 waits for E35-B/C failure for deployment, while E36-A1 may run
the representation probe in parallel; E37-E39
wait for substitution pass. All routine scoring uses B_dev; B_test is reserved.
