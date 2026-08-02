# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.necks.rnn import (
    Im2Seq,
    EncoderWithRNN,
    EncoderWithFC,
    SequenceEncoder,
    EncoderWithSVTR,
    trunc_normal_,
    zeros_,
)
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead
from .rec_nrtr_head import Transformer


class FCTranspose(nn.Layer):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose([0, 2, 1])
        else:
            return self.fc(x.transpose([0, 2, 1]))


class AddPos(nn.Layer):
    def __init__(self, dim, w):
        super().__init__()
        self.dec_pos_embed = self.create_parameter(
            shape=[1, w, dim], default_initializer=zeros_
        )
        self.add_parameter("dec_pos_embed", self.dec_pos_embed)
        trunc_normal_(self.dec_pos_embed)

    def forward(self, x):
        x = x + self.dec_pos_embed[:, : x.shape[1], :]
        return x


from .rec_length_branch import LengthBranch, LengthBranchAttn, LengthBranchAttnReg
from .rec_orient_branch import OrientBranch
from .rec_recon_branch import ReconBranch


class MultiHead(nn.Layer):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop("head_list")
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)
        self.in_channels = in_channels
        if self.use_pool:
            self.pool = nn.AvgPool2D(kernel_size=[3, 2], stride=[3, 2], padding=0)
        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                # sar head
                sar_args = self.head_list[idx][name]
                self.sar_head = eval(name)(
                    in_channels=in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **sar_args,
                )
            elif name == "NRTRHead":
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get("max_text_length", 25)
                nrtr_dim = gtc_args.get("nrtr_dim", 256)
                num_decoder_layers = gtc_args.get("num_decoder_layers", 4)
                if self.use_pos:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2),
                        FCTranspose(in_channels, nrtr_dim),
                        AddPos(nrtr_dim, 80),
                    )
                else:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2), FCTranspose(in_channels, nrtr_dim)
                    )

                self.gtc_head = Transformer(
                    d_model=nrtr_dim,
                    nhead=nrtr_dim // 32,
                    num_encoder_layers=-1,
                    beam_size=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length,
                    dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list["NRTRLabelDecode"],
                )
            elif name == "CTCHead":
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]["Neck"]
                encoder_type = neck_args.pop("name")
                self.ctc_encoder = SequenceEncoder(
                    in_channels=in_channels, encoder_type=encoder_type, **neck_args
                )
                # ctc head
                head_args = self.head_list[idx][name]["Head"]
                self.ctc_head = eval(name)(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args,
                )
            else:
                raise NotImplementedError(
                    "{} is not supported in MultiHead yet".format(name)
                )

        # optional auxiliary count branch; absent unless the config asks
        self.use_length_head = kwargs.get("use_length_head", False)
        self.length_no_detach = kwargs.get("length_no_detach", False)
        # Learnable fusion weight for CTCCountAwareDecode, trained via
        # LengthCountFusionLoss. Lives here (not in the loss module) because
        # build_optimizer only tracks model.parameters() -- a parameter
        # created inside a loss module is silently never updated.
        self.learn_count_beta = kwargs.get("learn_count_beta", False)
        if self.learn_count_beta:
            self.count_beta_raw = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(
                    value=kwargs.get("count_beta_init", 0.5)
                ),
            )
        if self.use_length_head:
            length_head_type = kwargs.get("length_head_type", "mlp")
            if length_head_type == "attn":
                self.length_head = LengthBranchAttn(
                    self.ctc_encoder.out_channels,
                    max_length=kwargs.get("length_max", 25),
                    hidden=kwargs.get("length_hidden", 128),
                    nhead=kwargs.get("length_nhead", 4),
                    num_layers=kwargs.get("length_layers", 2),
                )
            elif length_head_type == "attn_reg":
                self.length_head = LengthBranchAttnReg(
                    self.ctc_encoder.out_channels,
                    max_length=kwargs.get("length_max", 25),
                    hidden=kwargs.get("length_hidden", 128),
                    nhead=kwargs.get("length_nhead", 4),
                    num_layers=kwargs.get("length_layers", 2),
                    temperature=kwargs.get("length_temperature", 5.0),
                )
            else:
                self.length_head = LengthBranch(
                    self.ctc_encoder.out_channels,
                    max_length=kwargs.get("length_max", 25),
                    hidden=kwargs.get("length_hidden", 128),
                )

        # unified rec+orientation: 0/180 head sharing the ctc encoder
        self.use_orient_head = kwargs.get("use_orient_head", False)
        if self.use_orient_head:
            self.orient_head = OrientBranch(
                self.ctc_encoder.out_channels,
                hidden=kwargs.get("orient_hidden", 128),
            )

        # optional auxiliary reconstruction branch; reads backbone.recon_feat
        # (set via base_model.py's back-reference, since this module only
        # ever sees the backbone's final pooled output otherwise)
        self.use_recon_head = kwargs.get("use_recon_head", False)
        if self.use_recon_head:
            self.recon_out_shape = tuple(
                kwargs.get("recon_out_shape", (3, 48, 320))
            )
            self.recon_in_channels = kwargs.get("recon_in_channels", 384)
            self.recon_head = ReconBranch(
                self.recon_in_channels, out_shape=self.recon_out_shape
            )
            self.backbone_ref = None  # set post-construction by base_model.py

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
        # eval mode: plain ctc_out unless a length-gated postprocess needs the
        # count prediction too -- gating a live decode requires the length
        # head's output to actually reach postprocess, which it never did
        # before (dropped here), so length-gating silently could not do
        # anything. Orientation is still read at deploy time via
        # head.orient_head directly, not through this return value.
        if not self.training:
            if self.use_length_head:
                out = {
                    "ctc": ctc_out,
                    "length": F.softmax(self.length_head(ctc_encoder), axis=1),
                }
                if getattr(self, "learn_count_beta", False):
                    out["count_beta"] = F.softplus(self.count_beta_raw)
                return out
            return ctc_out
        if self.use_length_head:
            # Detach: the length loss must never backprop into the shared
            # encoder/backbone. Without this, training on a length-skewed
            # distribution (or one without full length coverage) measurably
            # hurt CTC accuracy (93.25% -> 92.72%, one seed) even though the
            # length loss itself is never added to the CTC/NRTR objective --
            # the leak was purely through this shared feature tensor.
            # length_no_detach is an explicit opt-out for controlled
            # experiments that want to test whether letting this gradient
            # reach the encoder helps CTC learn to count; off by default.
            if getattr(self, "length_no_detach", False):
                head_out["length"] = self.length_head(ctc_encoder)
            else:
                head_out["length"] = self.length_head(ctc_encoder.detach())
            if getattr(self, "learn_count_beta", False):
                head_out["count_beta"] = F.softplus(self.count_beta_raw)
        if self.use_orient_head:
            head_out["orient"] = self.orient_head(ctc_encoder)
        if self.use_recon_head:
            recon_feat = getattr(self.backbone_ref, "recon_feat", None)
            if recon_feat is not None:
                head_out["recon"] = self.recon_head(recon_feat)
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, targets[1:])
            head_out["sar"] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), targets[1:])
            head_out["gtc"] = gtc_out
        return head_out
