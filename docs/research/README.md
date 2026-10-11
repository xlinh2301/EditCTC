# Research reports

## Backbone study

- [`backbone_research_20261011.md`](backbone_research_20261011.md) — verified shortlist, advantages, disadvantages, evidence labels, and the proposed OCR experiment order.
- [`assets/backbone_taxonomy.svg`](assets/backbone_taxonomy.svg) — editable backbone taxonomy and decision path.
- [`assets/alignment_pipeline.svg`](assets/alignment_pipeline.svg) — shape-safe alignment-before-CTC pipeline and audit controls.
- [`backbone_architecture_deep_dive.md`](backbone_architecture_deep_dive.md) — block-by-block mechanics, tensor changes, OCR integration points, and risks.
- [`assets/backbone_block_mechanics.svg`](assets/backbone_block_mechanics.svg) — internal block diagrams for LCNetV4, RepViT, SVTRv2, and SSM families.

## Reading rule

The report separates three claims:

1. **Direct transfer:** PaddleOCR config/checkpoint and tensor contract are available.
2. **Transfer candidate:** the paper provides a visual backbone, but EditCTC must train and evaluate it.
3. **SOTA claim:** valid only for the named benchmark and protocol in the primary source.

Do not use ImageNet scores as OCR scores. Do not call a checkpoint pretrained until its source, hash, and tensor matching are recorded.
