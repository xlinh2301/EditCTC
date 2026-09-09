# Arbor Research Contract: EditCTC NERD/LCB

## Target

- Project: `/datastore/cndt_thangcpd/linhtruong/workspace5/release_EditCTC/code`
- Base branch: `main`
- Arbor trunk: `arbor/trunk/editctc-nerd-lcb`
- Checkpoint anchor: `/datastore/cndt_thangcpd/linhtruong/workspace5/release_EditCTC/checkpoints/s1024/best_accuracy`
- One seed: `1024`

## Objective

Test the proposed NERD and LCB changes in the order described by the existing
OpenSpec changes. The first question is whether balanced edit supervision makes
NERD predict useful non-KEEP operations. LCB changes are evaluated only after
NERD has measurable activity.

The primary score is exact-match `nerd_final_accuracy` on B_dev, maximized. The
baseline is `unknown - measure during INIT`; the previous checkpoint audit on
the held-out sets is evidence only and is not a B_dev baseline.

The evaluator also records CTC, NRTR, seed, LCB length, NERD changed/helped/
hurt counts, operation counts, wrong cases, and per-image branch traces.

## Evaluation contract

### B_dev: allowed during iteration

- Images: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain/crops/valid`
- Labels: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain/crops/valid_label.txt`
- Size: 410 labeled images at setup time
- Command template:

```text
cd {cwd} && bash scripts/arbor_eval_dev.sh --run-name editctc-nerd-lcb --node-id {node_id}
```

The command prints `score: <nerd_final_accuracy>`. It submits one Slurm job
with `--gres=mps:1`, uses the current worktree's source/config, and stores
artifacts below:

```text
/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_arbor_runs/
  editctc-nerd-lcb/<node_id>/
```

An experiment checkpoint is read from that node directory when present;
otherwise the immutable s1024 checkpoint is used. Training commands must write
their node checkpoint there and must use the real `Data/Indomain/crops/train`
and `valid` paths rather than the stale `DATA/...` paths in the historical
config.

The standard training command is:

```text
cd {cwd} && bash scripts/arbor_train.sh --config <config-in-worktree> --run-name editctc-nerd-lcb --node-id {node_id}
```

It submits one `--gres=mps:1` Slurm job, overrides the historical absolute
dataset paths with the real paths above, and writes
`<node>/checkpoints/best_accuracy`. The dev evaluator automatically selects
that checkpoint for the same node.

### B_test: protected until final verification

- Cross-data: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Cross-data/crops`
- Cross-data labels: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Cross-data/crops/crossdata_label.txt`
- In-domain test: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain/crops/test`
- In-domain labels: `/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain/crops/test_label.txt`
- Existing final evaluator: `/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/editctc_eval_two_sets.sh`

B_test must not be used for routine idea selection, tuning, or executor
iteration. Run it only on the selected Arbor trunk after the dev decision is
complete. Do not edit either test set, its labels, the checkpoint anchor, or
the scoring logic to influence results.

## Isolation and history

Every Arbor node is a separate Git worktree and branch created from the Arbor
trunk. Never edit or train directly in `main`. Keep only one GPU executor
active at a time. Each node owns its external output/checkpoint directory and
its Arbor artifact directory under `.arbor/sessions/`; no node may reuse
another node's output directory.

All worktrees must be created below:

```text
/datastore/cndt_thangcpd/linhtruong/workspace5/worktree/
```

The native CLI uses `scripts/arbor_run.sh`, which exports `TMPDIR` before
starting Arbor. The local fallback uses the same location by default and also
accepts `ARBOR_WORKTREE_ROOT` for an explicit override.

The planned nodes and their durable specs are:

| Node | OpenSpec change | Purpose |
| --- | --- | --- |
| E0 | `openspec/changes/e0-baseline-s1024` | baseline anchor |
| E1 | `openspec/changes/e1-balanced-edit-supervision` | all edit targets plus sampled KEEP |
| E1-freeze | `openspec/changes/e1-freeze-nerd-diagnostic` | isolate NERD edit supervision |
| E2a | `openspec/changes/e2a-edit-weight-030` | `weight_edit=0.30` |
| E2b | `openspec/changes/e2b-edit-weight-050` | `weight_edit=0.50` |
| E2c | `openspec/changes/e2c-op-class-weight` | light non-KEEP class weighting |
| E3 | `openspec/changes/e3-synthetic-one-error-seeds` | one-error synthetic seeds |
| E4 | `openspec/changes/e4-soft-length-conditioning` | soft LCB conditioning |
| E5 | `openspec/changes/e5-length-consistency` | LCB/NERD length consistency |

Do not change NERD depth or LCB architecture in the first rounds. Compare one
seed at a time and preserve all failed results as evidence.

## Budget and review

- Maximum completed/skipped/failed experiment nodes per run: 6
- Maximum parallel executors: 1
- Human interaction mode: `review`
- No provider secrets are stored in this repository
