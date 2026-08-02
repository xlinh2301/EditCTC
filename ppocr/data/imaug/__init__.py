# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
#
# TRIMMED for EditCTC: create_operators() resolves op class names via
# eval() in *this module's* namespace, so only the ops actually listed in
# the 5 configs' Train/Eval transforms need to be importable here:
#   DecodeImage, RecConAug, RecAug, MultiLabelEncode, RecResizeImg, KeepKeys
# (NRTRLabelEncode is referenced dynamically too, but only from *inside*
# label_ops.py's own MultiLabelEncode.__init__ via eval(gtc_encode), which
# resolves in label_ops.py's module namespace -- it does not need to be
# re-imported here.)
# All det/table/vqa/e2e/formula/latex augmentation pipelines (iaa_augment,
# make_border_map, east/sast/pg/table/ct/fce/drrg processors, latex/unimernet
# aug, etc.) were dropped along with their source files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .operators import DecodeImage, KeepKeys
from .rec_img_aug import RecAug, RecConAug, RecResizeImg
from .label_ops import MultiLabelEncode


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops
