Mechanism: Validate the new MultiHeadEditRefineNRTR architecture and one-epoch GPU smoke job 70802.
Hypothesis: Reusing the pretrained NRTR decoder with CTC seed, feature-map memory, and original-image tokens can replace the separate NRTR branch without breaking CTC.
Observable: build/load success, finite CTC/edit losses, train/eval output shapes, and CTC validation preservation.
Conflicts: No Cross training, no B_test, no deployment; report any checkpoint-key or vocabulary-mapping mismatch.
