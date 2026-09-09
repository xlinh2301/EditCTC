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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.postprocess.rec_postprocess import NRTRLabelDecode
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


def _decode_id_sequence(decoder, ids, probs=None, remove_duplicate=False):
    """Decode one already-collapsed branch sequence for diagnostics."""
    ids = np.asarray(ids, dtype="int64").reshape([1, -1])
    if probs is None:
        probs = np.ones(ids.shape, dtype="float32")
    else:
        probs = np.asarray(probs, dtype="float32").reshape([1, -1])
    return decoder.decode(
        ids, probs, is_remove_duplicate=remove_duplicate
    )[0]


def _softmax_np(logits):
    logits = np.asarray(logits, dtype="float32")
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=-1, keepdims=True)


def _write_branch_debug(log_file, file_path, debug, ctc_decoder, nrtr_decoder, final):
    """Write a per-image branch audit record.

    CTC and NERD use the CTC vocabulary.  NRTR has four special tokens of its
    own, so it is decoded through NRTRLabelDecode.  All arrays are reduced to
    JSON scalars/lists here, keeping the model output unchanged for callers.
    """
    ctc_probs = np.asarray(debug["ctc_probs"])
    ctc_ids = ctc_probs.argmax(axis=2)
    ctc_conf = ctc_probs.max(axis=2)
    ctc_result = ctc_decoder.decode(
        ctc_ids, ctc_conf, is_remove_duplicate=True
    )[0]

    length_logits = debug.get("length_logits")
    lcb = None
    if length_logits is not None:
        length_probs = _softmax_np(length_logits)
        length_prob = length_probs[0]
        top_lengths = np.argsort(-length_prob)[:5]
        lcb = {
            "predicted_length": int(top_lengths[0]),
            "confidence": float(length_prob[top_lengths[0]]),
            "top5": [
                {"length": int(i), "probability": float(length_prob[i])}
                for i in top_lengths
            ],
        }

    nrtr = None
    if debug.get("nrtr_ids") is not None:
        nrtr_ids = np.asarray(debug["nrtr_ids"])
        nrtr_probs = np.asarray(debug["nrtr_probs"])
        nrtr_result = nrtr_decoder([nrtr_ids[:1], nrtr_probs[:1]])[0]
        nrtr = {
            "text": nrtr_result[0],
            "confidence": float(nrtr_result[1]),
            "ids": nrtr_ids[0].tolist(),
            "token_probabilities": nrtr_probs[0].tolist(),
        }

    seed_lens = int(np.asarray(debug["seed_lens"])[0])
    seed_ids = np.asarray(debug["seed_ids"])[0, :seed_lens]
    refined_ids = list(debug["refined_ids"][0])
    seed_result = _decode_id_sequence(ctc_decoder, seed_ids)
    refined_result = _decode_id_sequence(ctc_decoder, refined_ids)

    op_logits = np.asarray(debug["edit_op_logits"])[0, :seed_lens]
    tok_logits = np.asarray(debug["edit_tok_logits"])[0, :seed_lens]
    op_probs = _softmax_np(op_logits) if seed_lens else np.empty((0, 4))
    tok_probs = _softmax_np(tok_logits) if seed_lens else np.empty((0, tok_logits.shape[-1]))
    op_ids = np.asarray(debug["edit_op_ids"])[0, :seed_lens]
    tok_ids = np.asarray(debug["edit_tok_ids"])[0, :seed_lens]
    op_names = ["KEEP", "REPLACE", "DELETE", "INSERT_AFTER"]

    def char_at(idx):
        idx = int(idx)
        return "" if idx == 0 else ctc_decoder.character[idx]

    frame_trace = []
    frame_probs = ctc_probs[0]
    top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    for frame_id, frame_id_probs in enumerate(frame_probs):
        top_id = int(ctc_ids[0, frame_id])
        top_prob = float(ctc_conf[0, frame_id])
        second_prob = float(top2[frame_id].min())
        frame_trace.append(
            {
                "frame": frame_id,
                "top1_id": top_id,
                "top1_token": char_at(top_id),
                "top1_probability": top_prob,
                "top1_top2_margin": top_prob - second_prob,
            }
        )

    operations = []
    for pos in range(seed_lens):
        op_id = int(op_ids[pos])
        token_id = int(tok_ids[pos])
        operations.append(
            {
                "position": pos,
                "operation": op_names[op_id],
                "operation_confidence": float(op_probs[pos, op_id]),
                "seed_token_id": int(seed_ids[pos]),
                "seed_token": char_at(seed_ids[pos]),
                "predicted_token_id": token_id,
                "predicted_token": char_at(token_id),
                "token_confidence": float(tok_probs[pos, token_id]),
            }
        )

    final_text = final[0] if isinstance(final, (list, tuple)) and final else ""
    final_conf = (
        float(final[1])
        if isinstance(final, (list, tuple)) and len(final) > 1
        else None
    )
    record = {
        "file": file_path,
        "ctc": {
            "text": ctc_result[0],
            "confidence": float(ctc_result[1]),
            "frame_count": int(ctc_probs.shape[1]),
            "decoded_length": len(ctc_result[0]),
            "frame_trace": frame_trace,
        },
        "nrtr": nrtr,
        "lcb": lcb,
        "nerd": {
            "seed_text": seed_result[0],
            "seed_length": seed_lens,
            "refined_text": refined_result[0],
            "refined_length": len(refined_result[0]),
            "changed_positions": sum(op["operation"] != "KEEP" for op in operations),
            "operations": operations,
        },
        "final": {"text": final_text, "confidence": final_conf},
    }
    json.dump(record, log_file, ensure_ascii=False)
    log_file.write("\n")
    log_file.flush()


def main():
    global_config = config["Global"]
    if config["Architecture"].get("algorithm") in [
        "UniMERNet",
        "PP-FormulaNet-S",
        "PP-FormulaNet-L",
        "PP-FormulaNet_plus-S",
        "PP-FormulaNet_plus-M",
        "PP-FormulaNet_plus-L",
    ]:
        config["PostProcess"]["is_infer"] = True
    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] in [
            "MultiHead",
            "MultiHeadEditRefine",
            "MultiHeadEditRefineUncertainty",
        ]:  # multi head, including EditCTC custom heads
            out_channels_list = {}
            char_num = len(getattr(post_process_class, "character"))
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    if config["Architecture"].get("algorithm") in ["LaTeXOCR"]:
        config["Architecture"]["Backbone"]["is_predict"] = True
        config["Architecture"]["Backbone"]["is_export"] = True
        config["Architecture"]["Head"]["is_export"] = True

    model = build_model(config["Architecture"])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            if config["Architecture"]["algorithm"] == "SRN":
                op[op_name]["keep_keys"] = [
                    "image",
                    "encoder_word_pos",
                    "gsrm_word_pos",
                    "gsrm_slf_attn_bias1",
                    "gsrm_slf_attn_bias2",
                ]
            elif config["Architecture"]["algorithm"] == "SAR":
                op[op_name]["keep_keys"] = ["image", "valid_ratio"]
            elif config["Architecture"]["algorithm"] == "RobustScanner":
                op[op_name]["keep_keys"] = ["image", "valid_ratio", "word_positions"]
            else:
                op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    global_config["infer_mode"] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config["Global"].get(
        "save_res_path", "./output/rec/predicts_rec.txt"
    )
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    branch_log = None
    nrtr_decoder = None
    branch_debug_enabled = config["Architecture"].get(
        "branch_debug", False
    ) or config["Architecture"].get("Head", {}).get("branch_debug", False)
    if branch_debug_enabled:
        branch_log_path = config["Global"].get(
            "branch_log_path", save_res_path + ".branches.jsonl"
        )
        branch_log_dir = os.path.dirname(branch_log_path)
        if branch_log_dir and not os.path.exists(branch_log_dir):
            os.makedirs(branch_log_dir)
        branch_log = open(branch_log_path, "w")
        nrtr_decoder = NRTRLabelDecode(
            character_dict_path=global_config.get("character_dict_path"),
            use_space_char=global_config.get("use_space_char", True),
        )
        logger.info("branch debug log: {}".format(branch_log_path))

    model.eval()

    infer_imgs = config["Global"]["infer_img"]
    infer_list = config["Global"].get("infer_list", None)
    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(infer_imgs, infer_list=infer_list):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                if config["Architecture"]["algorithm"] in [
                    "UniMERNet",
                    "PP-FormulaNet-S",
                    "PP-FormulaNet-L",
                    "PP-FormulaNet_plus-S",
                    "PP-FormulaNet_plus-M",
                    "PP-FormulaNet_plus-L",
                ]:
                    data = {"image": img, "filename": file}
                else:
                    data = {"image": img}
            batch = transform(data, ops)
            if config["Architecture"]["algorithm"] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list),
                ]
            if config["Architecture"]["algorithm"] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if config["Architecture"]["algorithm"] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positions = np.expand_dims(batch[2], axis=0)
                img_metas = [
                    paddle.to_tensor(valid_ratio),
                    paddle.to_tensor(word_positions),
                ]
            if config["Architecture"]["algorithm"] == "CAN":
                image_mask = paddle.ones(
                    (np.expand_dims(batch[0], axis=0).shape), dtype="float32"
                )
                label = paddle.ones((1, 36), dtype="int64")
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            if config["Architecture"]["algorithm"] == "SRN":
                preds = model(images, others)
            elif config["Architecture"]["algorithm"] == "SAR":
                preds = model(images, img_metas)
            elif config["Architecture"]["algorithm"] == "RobustScanner":
                preds = model(images, img_metas)
            elif config["Architecture"]["algorithm"] == "CAN":
                preds = model([images, image_mask, label])
            else:
                preds = model(images)
            branch_debug = (
                preds.get("branch_debug") if isinstance(preds, dict) else None
            )
            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0], int):
                # for RFLearning CNT branch
                info = str(post_result[0])
            elif config["Architecture"]["algorithm"] in [
                "LaTeXOCR",
                "UniMERNet",
                "PP-FormulaNet-S",
                "PP-FormulaNet-L",
                "PP-FormulaNet_plus-S",
                "PP-FormulaNet_plus-M",
                "PP-FormulaNet_plus-L",
            ]:
                info = str(post_result[0])
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")
                if branch_debug is not None:
                    _write_branch_debug(
                        branch_log,
                        file,
                        branch_debug,
                        post_process_class,
                        nrtr_decoder,
                        post_result[0],
                    )
    if branch_log is not None:
        branch_log.close()
    logger.info("success!")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
