Mechanism: Train an independent candidate/sequence verifier from NRTR-derived visual features over KEEP plus explicit CTC Top-K replacement candidates.
Hypothesis: Separating proposal ranking from selective acceptance can retain corrections while protecting CTC-correct sequences.
Observable: good>KEEP, KEEP>harmful, preference AUROC, and correction recall at harm 0.5/1/2 percent.
Conflicts: Run only after E42 coverage gate; no DELETE/INSERT, RL, Cross training, or B_test before supervised gates pass.
