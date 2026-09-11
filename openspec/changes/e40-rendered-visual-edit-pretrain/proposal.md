# E40 — Rendered visual edit pretraining

## Problem

NERD learns an all-KEEP policy on natural images. Earlier E31 text-only and
E32 rule-corrupted seed experiments showed that an edit decoder can receive
synthetic operation supervision without learning to transfer the decision to
natural CTC errors. The next pretraining corpus must expose both the edit
operation and visual evidence for the target digit.

## Proposal

Generate a large, deterministic corpus of meter-like numeric images. Each image
renders the clean target string `gt`; a separate `seed` field contains a
confusion-aware corruption. The seed is never rendered into the image. Rows
contain clean KEEP examples plus REPLACE, INSERT, DELETE, and mixed edits.

The first stage teaches edit mechanics from `(image, seed, gt)` while freezing
the CTC recognizer. Before transfer, run the frozen CTC recognizer over the
rendered images and retain rows whose predicted seed is wrong or ambiguous.
This CTC-filtered subset is the transfer corpus; the raw rule seed is only a
bootstrap diagnostic. Cross-data remains evaluation-only.

Rendering uses the same `[3,48,320]` input geometry and domain-randomizes font,
foreground/background polarity, spacing, blur, illumination, glare, occlusion,
noise, and contrast. Confusions are digit-aware (for example `3↔8`, `5↔6`,
`1↔7`) rather than random labels.

## Gates

1. Synthetic mechanics gate: on a held-out synthetic split, operation recall
   for REPLACE/INSERT/DELETE is nonzero and KEEP precision is high.
2. Frozen-CTC transfer gate: report how many rendered rows produce a natural
   CTC error, Top-K coverage, and error-type distribution. Do not train on the
   Cross-data split.
3. Real-image gate: after mixing a small natural error bank, require
   `helped > hurt`, KEEP protection `>95%`, and nonzero correction recall at a
   fixed harm rate on Indomain and Cross external evaluation.

Pure text pretraining is not expected to identify which digit the pixels
support; it is limited to learning operation mechanics and must not be used as
evidence that visual correction transfers.
