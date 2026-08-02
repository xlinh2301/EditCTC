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
# TRIMMED for EditCTC: only the backbone actually used by this repo's
# configs (PPLCNetV4, model_type "rec") is registered here. The upstream
# file unconditionally imported every det/rec/cls/table backbone inside
# each model_type branch; only the "rec"/"cls" branch is kept, and within
# it only PPLCNetV4 -- all other backbones (and their source files) were
# dropped.

__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "rec" or model_type == "cls":
        from .rec_lcnetv4 import PPLCNetV4

        support_dict = [
            "PPLCNetV4",
        ]
    else:
        raise NotImplementedError(
            "EditCTC only supports model_type 'rec'/'cls' (PPLCNetV4); "
            "got model_type={!r}".format(model_type)
        )

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(
            model_type, support_dict
        )
    )
    module_class = eval(module_name)(**config)
    return module_class
