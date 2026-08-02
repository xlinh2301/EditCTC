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


def build_loss(config):
    support_dict = [
        "MultiLossEditRefineUncertainty",
    ]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
