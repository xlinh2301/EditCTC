- [ ] Generate 200K deterministic rendered rows with manifest and summary.
- [ ] Run frozen CTC over synthetic images and build a CTC-filtered transfer
      manifest without using Cross-data for training.
- [ ] Add an explicit-seed data path to the edit head so `(image, seed, gt)` is
      used during pretraining while preserving the natural inference path.
- [ ] Train the edit decoder with frozen CTC first; evaluate synthetic
      operation metrics and real-image helped/hurt gates.
- [ ] Only then unfreeze the final visual block and mix the natural error bank.
