# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TRIMMED for EditCTC: only the loss actually used by this repo's configs
# (MultiLossEditRefineUncertainty, which internally dispatches to
# CTCLoss/NRTRLoss/LengthLoss via rec_multi_loss.py) is registered here.
# All det/cls/e2e/kie/table/vqa/sr losses were dropped along with their
# source files -- none of them are reachable from this model's configs.

import copy
import paddle
import paddle.nn as nn

from .rec_multi_loss_editrefine_uncertainty import MultiLossEditRefineUncertainty
from .rec_multi_loss_editrefine_factorized import MultiLossEditRefineFactorized
from .rec_multi_loss_editrefine_token import MultiLossEditRefineToken


def build_loss(config):
    support_dict = [
        "MultiLossEditRefineUncertainty",
        "MultiLossEditRefineFactorized",
        "MultiLossEditRefineToken",
        "MultiLossEditRefineBG",
    ]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict)
    )
    if module_name == "MultiLossEditRefineBG":
        # The BG-NERD head and its loss ship as one file.  Import it here rather
        # than at module level: bg_nerd_all_in_one imports rec_multi_loss, so a
        # top-level import back from this file makes the two packages circular
        # and deadlocks whenever the head is imported first -- which is what
        # tools/eval.py and tools/infer_rec.py do (build_head before build_loss).
        from ..modeling.heads.bg_nerd_all_in_one import MultiLossEditRefineBG

        return MultiLossEditRefineBG(**config)
    module_class = eval(module_name)(**config)
    return module_class
