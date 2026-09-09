## Context

The current decoder converts LCB logits to a hard argmax class and stops the gradient before looking up the length embedding. LCB has useful but imperfect length predictions, so hard conditioning can inject the wrong length and prevents NERD loss from training that interface.

## Goals / Non-Goals

**Goals:**

- Preserve LCB uncertainty when conditioning NERD.
- Create a differentiable NERD-to-LCB interface.
- Test soft conditioning independently before adding consistency loss.

**Non-Goals:**

- No change to the LCB classifier architecture or NERD depth.
- No change to operation argmax or inference beam search.

## Decisions

- Compute `p_len = softmax(length_logits)` and zero the reserved class zero before renormalizing.
- Form `len_emb = p_len @ length_embed.weight` and broadcast it over seed positions.
- Keep the same E3 seed mixture and natural evaluation protocol.

## Risks / Trade-offs

- [Soft distribution may blur length signal] → Compare length accuracy, NERD operation metrics, and final accuracy against hard conditioning.
- [New gradient path can destabilize LCB] → Start from E3, use the same learning rate, and inspect LCB loss/gradient norms.
- [Class-zero handling bug] → Add a unit test asserting zero probability for class zero and sum one over valid classes.

