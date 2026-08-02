# MultiHead + CTC-Seeded Edit Refinement Decoder.
#
# Subclasses MultiHead and reuses everything (CTC head/neck, NRTR gtc branch,
# the top-level LengthBranch) unchanged -- mirrors the additive pattern used
# by MultiHeadInterCTC (rec_multi_head_interctc.py): build the parent first,
# then bolt on the new module.
#
# Pipeline (train mode only -- the edit branch is not used at inference,
# matching how "length"/"gtc" outputs are also train-only in MultiHead):
#   1. CTC greedy-decode + blank/repeat collapse on this batch's own CTC
#      logits -> seed token sequence S (stop-gradient; it is only used as an
#      embedding lookup index, not backprop through argmax).
#   2. LengthBranch's predicted length class (stop-gradient) -> length embed,
#      broadcast onto every seed position.
#   3. Seed token embeddings (+ positional embed + length embed) go through a
#      small bidirectional Transformer decoder that cross-attends into the
#      CTC encoder's visual features (ctc_encoder, the same (B,T,C) sequence
#      the CTC head reads).
#   4. Per seed position: a 4-way edit-op head {keep, replace, delete,
#      insert-after} and a vocab-sized token head (used for replace / the
#      token attached to insert-after).
#
# This is intentionally single-step (no iterative re-decoding, no diffusion,
# no CFG, no text-only pretraining) -- see rec_edit_loss.py for how targets
# are constructed and the documented simplifications that keep it that way.

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

from .rec_multi_head import MultiHead


class EditRefineDecoder(nn.Layer):
    """Small bidirectional Transformer decoder with cross-attention into the
    encoder's visual features. Input = seed token embeddings + length embed.
    Output per seed position = edit-op logits + token-prediction logits.
    """

    # CTC blank is index 0 in this codebase's CTCLabelDecode convention; the
    # collapsed seed reuses 0 as its pad id too (never a valid emitted token).
    BLANK_ID = 0

    def __init__(
        self,
        in_channels,
        vocab_size,
        edit_dim=128,
        num_layers=3,
        nhead=4,
        max_seed_len=25,
        length_max=25,
        dropout=0.1,
        head_dropout=0.2,
        use_cross_attn=True,
    ):
        super().__init__()
        self.max_seed_len = max_seed_len
        self.vocab_size = vocab_size
        self.edit_dim = edit_dim
        # Additive flag for text-only pretraining (no image -> no encoder
        # visual features to cross-attend into). When False, forward()
        # substitutes a single learned "null" memory token for `memory`
        # instead of requiring real visual features. Default True preserves
        # the original joint-training behavior exactly.
        self.use_cross_attn = use_cross_attn
        self.null_memory = self.create_parameter(
            shape=[1, 1, edit_dim],
            default_initializer=nn.initializer.TruncatedNormal(std=0.02),
        )
        self.add_parameter("null_memory", self.null_memory)

        self.token_embed = nn.Embedding(vocab_size, edit_dim)
        self.pos_embed = self.create_parameter(
            shape=[1, max_seed_len, edit_dim],
            default_initializer=nn.initializer.TruncatedNormal(std=0.02),
        )
        self.add_parameter("pos_embed", self.pos_embed)
        # +1 because LengthBranch's classifier has max_length+1 classes
        # (index 0 unused, see rec_length_branch.py).
        self.length_embed = nn.Embedding(length_max + 1, edit_dim)

        self.mem_proj = (
            None if in_channels == edit_dim else nn.Linear(in_channels, edit_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=edit_dim,
            nhead=nhead,
            dim_feedforward=edit_dim * 4,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Regularization: dropout on the decoder output before the
        # classification heads. Standard nn.Dropout -- active in train(),
        # a no-op in eval() -- applied identically on both the
        # use_cross_attn=True (joint training) and False (text-pretraining)
        # paths since it sits after `self.decoder(...)`, not inside either
        # branch. Does not change output shape.
        self.head_dropout = nn.Dropout(p=head_dropout)

        self.op_head = nn.Linear(edit_dim, 4)  # keep, replace, delete, insert-after
        self.tok_head = nn.Linear(edit_dim, vocab_size)

    def build_seed(self, ctc_logits):
        """Greedy CTC decode + blank/repeat collapse, done with numpy on the
        detached logits (no gradient needed -- the seed is only used as a
        discrete embedding-table index)."""
        ids = paddle.argmax(ctc_logits, axis=2)
        ids.stop_gradient = True
        ids_np = ids.numpy()
        batch_size = ids_np.shape[0]
        seeds = np.zeros((batch_size, self.max_seed_len), dtype="int64")
        seed_lens = np.zeros((batch_size,), dtype="int64")
        for b in range(batch_size):
            prev = -1
            out = []
            for t in ids_np[b].tolist():
                if t != self.BLANK_ID and t != prev:
                    out.append(t)
                    if len(out) >= self.max_seed_len:
                        break
                prev = t
            seed_lens[b] = len(out)
            if out:
                seeds[b, : len(out)] = out
        return seeds, seed_lens

    def forward(
        self,
        ctc_logits=None,
        memory=None,
        length_logits=None,
        seed_ids=None,
        seed_lens=None,
    ):
        # `seed_ids`/`seed_lens` let a caller supply the seed directly
        # (text-only pretraining: seeds come from synthetic corruption, not
        # from a CTC forward pass). Default None preserves the original
        # joint-training path (seed derived from `ctc_logits`) unchanged.
        if seed_ids is None:
            seeds_np, seed_lens_np = self.build_seed(ctc_logits)
            seeds = paddle.to_tensor(seeds_np, dtype="int64")
            seeds.stop_gradient = True
            seed_lens_out = paddle.to_tensor(seed_lens_np, dtype="int64")
        else:
            seeds = seed_ids
            seed_lens_out = (
                seed_lens
                if seed_lens is not None
                else paddle.full([seeds.shape[0]], seeds.shape[1], dtype="int64")
            )

        seed_len_dim = seeds.shape[1]
        tok_emb = self.token_embed(seeds)
        tok_emb = tok_emb + self.pos_embed[:, :seed_len_dim, :]

        if length_logits is not None:
            len_cls = paddle.argmax(length_logits, axis=1)
            len_cls.stop_gradient = True
            len_emb = self.length_embed(len_cls).unsqueeze(1)
            tok_emb = tok_emb + len_emb

        if not self.use_cross_attn:
            mem = self.null_memory.expand([seeds.shape[0], 1, self.edit_dim])
        else:
            mem = memory if self.mem_proj is None else self.mem_proj(memory)
        dec_out = self.decoder(tok_emb, mem)
        dec_out = self.head_dropout(dec_out)

        op_logits = self.op_head(dec_out)
        tok_logits = self.tok_head(dec_out)

        return {
            "op_logits": op_logits,
            "tok_logits": tok_logits,
            "seed_ids": seeds,
            "seed_lens": seed_lens_out,
        }


def apply_edit_ops(seed_ids, op_ids, tok_ids):
    """Apply per-position edit ops to a seed sequence, producing the refined
    id sequence. This MUST be the exact inverse of how rec_edit_loss.py's
    `levenshtein_ops` builds (op, tok) targets from (seed, gt) during
    training -- see that function's docstring for the alignment convention.

    Convention (mirrors rec_edit_loss.KEEP/REPLACE/DELETE/INSERT_AFTER = 0..3):
      KEEP         -> emit seed_ids[i] unchanged.
      REPLACE      -> emit tok_ids[i] instead of seed_ids[i].
      DELETE       -> emit nothing (seed_ids[i] dropped).
      INSERT_AFTER -> emit seed_ids[i], THEN emit tok_ids[i] right after it.
                      (levenshtein_ops anchors an insertion on the seed
                      position that precedes the gap and always overwrites
                      that position's default KEEP with INSERT_AFTER --
                      see the "only keep the insert" simplification in its
                      docstring -- so the anchor token itself is always kept,
                      never replaced/deleted, when this op fires.)

    Args:
        seed_ids: list[int] (or 1-D array-like) of length n -- the CTC
            greedy-decoded, blank/repeat-collapsed seed for ONE sample,
            already truncated to that sample's true seed_len (no padding).
        op_ids: list[int] of length n, argmax(op_logits, axis=-1) per
            seed position, one of KEEP/REPLACE/DELETE/INSERT_AFTER.
        tok_ids: list[int] of length n, argmax(tok_logits, axis=-1) per
            seed position (only meaningful where op is REPLACE or
            INSERT_AFTER; ignored otherwise).

    Returns:
        list[int]: the refined id sequence. Handles n == 0 (empty seed ->
        returns []) and an all-DELETE seed (-> returns []) naturally, since
        both are just the degenerate cases of the same per-position loop.
        INSERT_AFTER at the last seed position is also handled naturally --
        it simply appends the inserted token at the end of the output.
    """
    out = []
    for sid, op, tid in zip(seed_ids, op_ids, tok_ids):
        op = int(op)
        if op == KEEP:
            out.append(int(sid))
        elif op == REPLACE:
            out.append(int(tid))
        elif op == DELETE:
            pass
        elif op == INSERT_AFTER:
            out.append(int(sid))
            out.append(int(tid))
        else:  # pragma: no cover - defensive; op_head only emits 0..3
            out.append(int(sid))
    return out


# Local mirror of rec_edit_loss.py's op-id convention (kept in sync
# manually -- both files agree KEEP=0, REPLACE=1, DELETE=2,
# INSERT_AFTER=3; see that module for the authoritative definition used to
# build training targets).
KEEP, REPLACE, DELETE, INSERT_AFTER = 0, 1, 2, 3


def build_refined_ctc_probs(refined_ids_per_sample, vocab_size, blank_id=0):
    """Pack variable-length per-sample refined id sequences into a dense
    (B, T, vocab_size) pseudo-probability tensor that CTCLabelDecode can
    consume EXACTLY like a real CTC softmax output, with no changes needed
    on the decode side.

    Why this is necessary (not just "reshape and go"): CTCLabelDecode always
    decodes with `is_remove_duplicate=True` (it collapses consecutive
    identical *frame* predictions, which is correct for raw per-timestep CTC
    logits where the same character legitimately repeats across several
    adjacent frames). `refined_ids` is already the FINAL character sequence
    (one entry per emitted character, no frame repetition) -- so packing it
    1 char = 1 frame and running it through that same collapse would
    silently delete real consecutive duplicate digits (e.g. "1122" ->
    "12"), which is exactly the kind of silent-corruption bug this task is
    trying to avoid repeating. Water-meter digit strings frequently contain
    repeated adjacent digits, so this is not a hypothetical edge case.

    Fix: interleave a blank frame between every pair of adjacent refined
    tokens before packing. `is_remove_duplicate` can then only ever collapse
    a real char frame against an identical NEIGHBORING blank frame (which is
    impossible -- blank_id != any char id) or two identical real-char frames
    that are no longer adjacent (impossible, we always insert a blank
    between them). Blanks are stripped by CTCLabelDecode's own
    ignored-token filtering (blank_id is index 0, always ignored), so the
    decoded text is exactly the refined sequence, unmodified.

    Returns a paddle.Tensor of shape (B, T, vocab_size), values in {0, 1}
    (one-hot per frame), safe to use with CTCLabelDecode.decode(is_remove_
    duplicate=True) and with .max(axis=2) as a (trivial, all-1.0) confidence
    proxy.
    """
    batch_size = len(refined_ids_per_sample)
    frame_lens = []
    framed = []
    for ids in refined_ids_per_sample:
        frames = []
        for k, tid in enumerate(ids):
            if k > 0:
                frames.append(blank_id)
            frames.append(int(tid))
        framed.append(frames)
        frame_lens.append(len(frames))
    max_t = max(frame_lens) if frame_lens else 0
    max_t = max(max_t, 1)  # avoid a zero-width tensor if the whole batch is empty

    probs = np.zeros((batch_size, max_t, vocab_size), dtype="float32")
    # Pad positions default to an all-blank one-hot frame (index blank_id),
    # which decodes to nothing -- consistent with padding past a sample's
    # real length.
    probs[:, :, blank_id] = 1.0
    for b, frames in enumerate(framed):
        for t, tid in enumerate(frames):
            probs[b, t, :] = 0.0
            probs[b, t, tid] = 1.0
    return paddle.to_tensor(probs, dtype="float32")


class MultiHeadEditRefine(MultiHead):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__(in_channels, out_channels_list, **kwargs)
        assert self.use_length_head, (
            "MultiHeadEditRefine requires use_length_head: true -- the "
            "length embedding is one of the edit decoder's inputs."
        )
        self.edit_refine_head = EditRefineDecoder(
            in_channels=self.ctc_encoder.out_channels,
            vocab_size=out_channels_list["CTCLabelDecode"],
            edit_dim=kwargs.get("edit_hidden", 128),
            num_layers=kwargs.get("edit_layers", 3),
            nhead=kwargs.get("edit_nhead", 4),
            max_seed_len=kwargs.get("length_max", 25),
            length_max=kwargs.get("length_max", 25),
            dropout=kwargs.get("edit_dropout", 0.1),
            head_dropout=kwargs.get("edit_head_dropout", 0.2),
        )

    def forward(self, x, targets=None):
        if self.use_pool:
            x = self.pool(
                x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2])
            )
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder
        # eval mode
        if not self.training:
            return self._eval_refined_output(ctc_out, ctc_encoder)
        if self.use_length_head:
            head_out["length"] = self.length_head(ctc_encoder)
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, targets[1:])
            head_out["sar"] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), targets[1:])
            head_out["gtc"] = gtc_out

        # ctc_out is raw logits here (training mode -> CTCHead skips softmax).
        head_out["edit"] = self.edit_refine_head(
            ctc_out, ctc_encoder, head_out.get("length")
        )
        return head_out

    def _eval_refined_output(self, ctc_out, ctc_encoder):
        """Eval/inference path: actually apply the edit-refine branch's
        predicted ops to the CTC seed, instead of silently discarding them
        (the bug this fix addresses -- previously this method didn't exist
        and `forward()` just `return`ed `ctc_out` unchanged at eval time, so
        the edit decoder's op_logits/tok_logits were computed during
        training for EditLoss only and never consulted at inference).

        ctc_out here is post-softmax probs (CTCHead applies softmax when
        not self.training), shape (B, T, vocab_size) -- exactly what
        EditRefineDecoder.build_seed()/argmax expect.
        """
        # 1. CTC greedy decode -> seed (same code path used during joint
        # training -- build_seed() argmaxes ctc_out internally, softmax
        # doesn't change the argmax so reusing it here is safe).
        length_logits = (
            self.length_head(ctc_encoder) if self.use_length_head else None
        )
        seeds_np, seed_lens_np = self.edit_refine_head.build_seed(ctc_out)
        seeds = paddle.to_tensor(seeds_np, dtype="int64")

        # 2. Run the edit decoder for real, with real encoder features
        # (use_cross_attn=True by construction/default -- unchanged from
        # training), to get op_logits/tok_logits.
        edit_out = self.edit_refine_head(
            memory=ctc_encoder,
            length_logits=length_logits,
            seed_ids=seeds,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
        )
        op_ids = paddle.argmax(edit_out["op_logits"], axis=2).numpy()
        tok_ids = paddle.argmax(edit_out["tok_logits"], axis=2).numpy()

        # 3. Apply ops per-sample (variable seed lengths -> looped, not a
        # batched tensor op -- see apply_edit_ops docstring for exact
        # semantics, which must mirror rec_edit_loss.levenshtein_ops).
        batch_size = seeds_np.shape[0]
        refined_ids_per_sample = []
        for b in range(batch_size):
            n = int(seed_lens_np[b])
            refined = apply_edit_ops(
                seeds_np[b, :n].tolist(),
                op_ids[b, :n].tolist(),
                tok_ids[b, :n].tolist(),
            )
            refined_ids_per_sample.append(refined)

        # 4. Pack into a (B, T, vocab_size) pseudo-prob tensor so
        # CTCLabelDecode (postprocess) can consume this exactly like a
        # normal CTC output with zero changes to decode-side code -- see
        # build_refined_ctc_probs() docstring for why the blank-interleave
        # is required (protects legitimate repeated digits from being
        # collapsed by CTCLabelDecode's is_remove_duplicate=True).
        return build_refined_ctc_probs(
            refined_ids_per_sample,
            vocab_size=self.edit_refine_head.vocab_size,
            blank_id=EditRefineDecoder.BLANK_ID,
        )
