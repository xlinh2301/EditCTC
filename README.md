# EditCTC

Standalone training/eval/inference repo for the EditCTC water-meter OCR
recognition model, extracted from a full checkout of PaddleOCR
(`PaddleOCR_repo/`) so it can be versioned, shared, and run without carrying
the ~250MB upstream framework.

## What this is

A PaddleOCR SVTR/CTC recognizer (backbone `PPLCNetV4`, algorithm
`SVTR_LCNet`) with a custom head + loss stacked on top:

- **Head**: `MultiHeadEditRefineUncertainty` — a `MultiHead` (CTC branch with
  a `lightsvtr` neck + NRTR/GTC branch + auxiliary length-prediction branch)
  plus a CTC-Seeded Edit Refinement Decoder, gated by an uncertainty signal.
- **Loss**: `MultiLossEditRefineUncertainty` — wraps `CTCLoss` + `NRTRLoss` +
  `LengthLoss` (via `MultiLoss`) and adds `EditLossUncertainty` for the edit
  branch.
- **PostProcess**: `CTCLabelDecode`.
- **Metric**: `RecMetric`.

Custom (non-upstream) source files:

- `ppocr/modeling/heads/rec_edit_refine_head.py` — `EditRefineDecoder`,
  `apply_edit_ops`, `build_refined_ctc_probs`, `MultiHeadEditRefine`
- `ppocr/modeling/heads/rec_edit_refine_head_uncertainty.py` —
  `MultiHeadEditRefineUncertainty`, `build_seed_with_frames`
- `ppocr/losses/rec_edit_loss.py` — `levenshtein_ops`, `EditLoss`,
  `IGNORE_INDEX`
- `ppocr/losses/rec_edit_loss_uncertainty.py` — `EditLossUncertainty`
- `ppocr/losses/rec_multi_loss_editrefine_uncertainty.py` —
  `MultiLossEditRefineUncertainty`

Everything else under `ppocr/` and `tools/` is copied unmodified (or
trimmed-but-behavior-preserving, see below) from PaddleOCR, traced by
following the real import graph starting at `tools/train.py`,
`tools/eval.py`, and `tools/infer_rec.py`.

## The 5 configs

`config/` holds 5 configs that differ **only** in `Global.seed`,
`Global.save_model_dir`, `Global.checkpoints`, and `Global.save_res_path`
(diffed to confirm — everything else, including `gate_mode: random`, is
identical):

- `PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml`
- `PP-OCRv6_small_rec_s2048_uncertainty_random_50ep.yml`
- `PP-OCRv6_small_rec_s4096_uncertainty_random_50ep.yml`
- `PP-OCRv6_small_rec_s8192_uncertainty_random_50ep.yml` — **paper's
  headline result** (seed 8192, `gate_mode: random`)
- `PP-OCRv6_small_rec_s16384_uncertainty_random_50ep.yml`

## Commands

```
python tools/train.py -c config/PP-OCRv6_small_rec_s8192_uncertainty_random_50ep.yml

python tools/eval.py -c config/PP-OCRv6_small_rec_s8192_uncertainty_random_50ep.yml -o Global.checkpoints=<path>

python tools/infer_rec.py -c config/PP-OCRv6_small_rec_s8192_uncertainty_random_50ep.yml -o Global.infer_img=<path> Global.checkpoints=<path>
```

Run from the `EditCTC/` root. `tools/*.py` do their own `sys.path`
manipulation (two levels up from `__file__`), which resolves correctly given
this repo mirrors PaddleOCR's `tools/` + `ppocr/` layout — verified by
actually running `tools/train.py` from this repo (see Verification below);
no fix was needed.

## Arbor experiment workflow

Arbor is installed in the untracked `.arbor-venv/` environment and its
project-local skills are under `.agents/skills/arbor-*`. The durable contract
is [`ARBOR_CONTRACT.md`](ARBOR_CONTRACT.md), and the machine-readable settings
are in [`research_config.yaml`](research_config.yaml).

Each experiment must run in its own Arbor worktree/branch. Iteration uses B_dev
through `scripts/arbor_eval_dev.sh`; the two test sets remain reserved for the
final trunk evaluation. A typical launch from a clean `main` checkout is:

```bash
.arbor-venv/bin/arbor doctor
scripts/arbor_run.sh run --yes \
  --yes-cwd "$PWD" \
  --config research_config.yaml \
  --run-name editctc-nerd-lcb \
  --max-cycles 6 \
  "Improve NERD edit supervision using the OpenSpec experiment sequence"
```

The dev evaluator prints `score: <nerd_final_accuracy>` and writes detailed
branch traces, wrong cases, summaries, and per-node Slurm logs under
`/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_arbor_runs/`.
For training, an Executor can use `scripts/arbor_train.sh`; it applies the
real `Data/Indomain/crops/{train,valid}` paths and writes a node-local
`checkpoints/best_accuracy` before evaluation.

## Hardcoded cluster/machine-absolute paths

All 5 configs (values below are from `s8192`; the other 4 differ only by the
seed number embedded in `save_model_dir`/`checkpoints`/`save_res_path`):

| Config key | Current value | Needs changing in a new environment? |
|---|---|---|
| `Global.pretrained_model` | `/datastore/cndt_thangcpd/linhtruong/workspace5/workdir_text_rec/PPOCRv6/checkpoints/ppocrv6_small_rec_pretrained` | Yes — cluster-absolute path |
| `Global.edit_refine_pretrained` | `/datastore/cndt_thangcpd/linhtruong/workspace5/workdir_text_rec/PPOCRv6/checkpoints_editrefine_textpretrain/editrefine_pretrained` | Yes — cluster-absolute path |
| `Global.save_model_dir` | `E:/NCKH/workdir_waterclock/Checkpoint/EditCTC/s<seed>` | Yes — local Windows path |
| `Global.checkpoints` | `E:/NCKH/workdir_waterclock/Checkpoint/EditCTC/s<seed>/best_accuracy` | Yes — local Windows path |
| `Train.dataset.data_dir` | `/datastore/cndt_thangcpd/linhtruong/workspace5/DATA/crop/images/train` | Yes — cluster-absolute path |
| `Train.dataset.label_file_list` | `/datastore/cndt_thangcpd/linhtruong/workspace5/DATA/crop/train_label.txt` | Yes — cluster-absolute path |
| `Eval.dataset.data_dir` | `/datastore/cndt_thangcpd/linhtruong/workspace5/DATA/crop/images/valid` | Yes — cluster-absolute path |
| `Eval.dataset.label_file_list` | `/datastore/cndt_thangcpd/linhtruong/workspace5/DATA/crop/valid_label.txt` | Yes — cluster-absolute path |
| `Global.character_dict_path` | `ppocr/utils/dict/ppocrv6_dict.txt` | No — relative, resolves inside this repo (file is included) |
| `Global.save_res_path` | `./output/rec/predicts_PP-OCRv6_small_rec_s<seed>_uncertainty_random_50ep.txt` | No — relative |

All 8 machine-specific keys above can be overridden on the command line with
`-o Key.Path=value` without editing the YAML, e.g.
`-o Train.dataset.data_dir=/new/path Global.pretrained_model=/new/path`.

## What was dropped and why

Traced from `tools/train.py` / `tools/eval.py` / `tools/infer_rec.py`
(imports followed transitively; registry `__init__.py` files under
`ppocr/modeling/*`, `ppocr/losses`, `ppocr/postprocess`, `ppocr/metrics`,
`ppocr/data` read directly to resolve YAML class names to files). Not
included:

- All detection architectures/heads/necks/losses/postprocess (DB, EAST,
  SAST, PSE, FCE, CT, PG, DRRG, ...)
- `ppstructure/` (layout/table/structure) — not imported by the rec
  training/eval/infer path at all
- All other recognition backbones (ResNet variants, MobileNetV3, HGNet,
  ViT*, SVTRv2, Donut-Swin, ...) — only `PPLCNetV4` is used
- All other recognition heads/losses not reachable from `MultiHead(EditRefine)*`
  (SRN, ABINet, VisionLAN, RFL, CAN, LaTeXOCR/UniMERNet/PPFormulaNet, ParseQ,
  CPPD, SATRN, SPIN, RobustScanner, ...)
- `SARHead`/`SARLoss` source files were *kept* because `rec_multi_head.py`
  and `rec_multi_loss.py` import them unconditionally at module level even
  though this model's configs never select the SAR branch (`gtc_encode:
  NRTRLabelEncode`, not SAR) — dropping them would have broken those
  unmodified files.
- kie/, table/, vqa/token/, e2e/, cls/ (besides the light-weight `ClsHead`
  import path, which was also dropped since it's never selected) algorithm
  families
- LMDBDataSet/PGDataSet/PubTabDataSet/LaTeXOCRDataSet and their augmentation
  pipelines (iaa_augment, make_border/shrink_map, east/sast/pg/table/ct/fce/
  drrg processors, latex/unimernet aug) — configs only use `SimpleDataSet`
  (Eval) and `MultiScaleDataSet`+`MultiScaleSampler` (Train)
- `ppocr/data/imaug/vqa/token/` (VQATokenPad, VQASerTokenChunk, ...) — only
  `vqa/augment.py:order_by_tbyx` is needed (pulled in transitively by
  `label_ops.py`), so `vqa/__init__.py` was trimmed to not import `.token`
- Pretrained-model download scripts / `deploy/`, `applications/`,
  `benchmark/`, docs, `mcp_server/`, `langchain-paddleocr/`,
  `paddleocr-js/`, top-level debug/probe scripts (`ctc_cut_digits.py`,
  `debug_ctc_*.py`, `verify_*.py`, etc.)

### Registry `__init__.py` files trimmed

Registries (`ppocr/modeling/backbones/__init__.py`,
`ppocr/modeling/heads/__init__.py`, `ppocr/losses/__init__.py`,
`ppocr/postprocess/__init__.py`, `ppocr/metrics/__init__.py`,
`ppocr/data/__init__.py`, `ppocr/data/imaug/__init__.py`,
`ppocr/data/imaug/vqa/__init__.py`) were trimmed to only import/register the
classes this model's configs actually select — each trimmed file has a
comment block at the top listing exactly what was removed and why. Nothing
that showed up in the real static trace or the runtime smoke test below was
removed.

`ppocr/optimizer/__init__.py` and `ppocr/modeling/transforms/__init__.py`
were **not** trimmed: `optimizer/__init__.py`'s imports are already lazy
(inside the function, only pulling in the small `learning_rate.py` /
`regularizer.py` / `optimizer.py` submodules, all of which were kept whole),
and `transforms/__init__.py`'s `build_transform` is never called by this
model (`Architecture.Transform` is `None` in every config) so its lazy
internal imports never execute — trimming it would have added risk for zero
benefit.

## Verification

Verified with a locally installed `paddlepaddle-gpu==3.0.0` (Python 3.13):

```
python -c "from ppocr.modeling.architectures import build_model; \
           from ppocr.losses import build_loss; \
           from ppocr.postprocess import build_post_process; \
           from ppocr.metrics import build_metric; \
           from ppocr.optimizer import build_optimizer; \
           from ppocr.data import create_operators, build_dataloader"
```

and by actually constructing the real model/loss/postprocess/metric/
optimizer/transform-ops objects from `config/PP-OCRv6_small_rec_s8192_uncertainty_random_50ep.yml`,
and by running `tools/train.py` unmodified from the repo root, which loads
the config, builds the model, and reaches the dataset-loading stage before
failing on the (expected, machine-specific) missing `/datastore/...`
paths — i.e. the entrypoint, `sys.path` handling, and every import in the
chain are confirmed working end to end; only the data itself is absent on
this machine.

`python -m py_compile` was additionally run over every copied `.py` file
(clean; only two pre-existing upstream `SyntaxWarning`s for `\d`/`\W` regex
literals, not errors) and every `from .` / `from ppocr...` import in every
copied file was checked programmatically against the set of copied files —
zero unresolved local imports.

No import in the traced set was left unverified.
