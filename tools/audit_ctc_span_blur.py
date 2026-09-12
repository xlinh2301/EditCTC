#!/usr/bin/env python3
"""Measure whether one-span blur actually changes frozen CTC predictions."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import paddle, yaml
from ppocr.data.imaug import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

def collapse(ids, chars):
    out=[]; prev=0
    for i in ids:
        i=int(i)
        if i!=0 and i!=prev:
            out.append(chars[i])
        prev=i
    return ''.join(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data-dir',required=True); ap.add_argument('--label-file',required=True); ap.add_argument('--span-index',required=True); ap.add_argument('--limit',type=int,default=256); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--min-span-fraction',type=float,default=0.06); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); g=cfg['Global']; post=build_post_process(cfg['PostProcess'],g); C=len(post.character); h=cfg['Architecture']['Head']; h['out_channels_list']={'CTCLabelDecode':C,'SARLabelDecode':C+2,'NRTRLabelDecode':C+3}; g['checkpoints']=args.checkpoint; g['pretrained_model']=None; model=build_model(cfg['Architecture']); load_model(cfg,model); model.eval()
    idx=json.load(open(args.span_index));
    ops_clean=create_operators([{'DecodeImage':{'img_mode':'BGR','channel_first':False}},{'RecResizeImg':{'image_shape':[3,48,320],'infer_mode':True}},{'KeepKeys':{'keep_keys':['image']}}],g)
    ops_blur=create_operators([{'DecodeImage':{'img_mode':'BGR','channel_first':False}},{'CTCSpanBlurAug':{'span_index_path':args.span_index,'prob':1.0,'min_kernel':15,'max_kernel':21,'sigma_min':6,'sigma_max':10,'expand':0.35,'min_span_fraction':args.min_span_fraction}},{'RecResizeImg':{'image_shape':[3,48,320],'infer_mode':True}},{'KeepKeys':{'keep_keys':['image']}}],g)
    rows=[]
    for l in open(args.label_file):
        if l.strip():
            n,gt=l.rstrip().split('\t',1); p=Path(args.data_dir)/n
            if p.is_file() and n in idx: rows.append((n,gt,p))
    random.seed(7); random.shuffle(rows); rows=rows[:args.limit]; stats={'rows':len(rows),'clean_wrong':0,'blur_wrong':0,'changed':0,'blur_wrong_from_clean_correct':0,'clean_correct':0}
    for off in range(0,len(rows),args.batch_size):
        br=rows[off:off+args.batch_size]; a=[]; b=[]
        for n,gt,p in br:
            x=transform({'img_path':str(p),'image':p.read_bytes()},ops_clean)[0]; y=transform({'img_path':str(p),'image':p.read_bytes()},ops_blur)[0]; a.append(x); b.append(y)
        with paddle.no_grad():
            def pred(arr):
                z=paddle.to_tensor(np.stack(arr)); f=model.backbone(z); f=model.neck(f) if model.use_neck else f; e=model.head.ctc_encoder(f); q=model.head.ctc_head(e); return q.argmax(axis=-1).numpy()
            ca=pred(a); cb=pred(b)
        for j,(_,gt,_) in enumerate(br):
            pc=collapse(ca[j],post.character); pb=collapse(cb[j],post.character); cc=pc==gt; bc=pb==gt; stats['clean_correct']+=cc; stats['clean_wrong']+=not cc; stats['blur_wrong']+=not bc; stats['changed']+=pc!=pb; stats['blur_wrong_from_clean_correct']+=(cc and not bc)
    stats['blur_induced_error_rate_on_clean_correct']=stats['blur_wrong_from_clean_correct']/max(stats['clean_correct'],1); print(json.dumps(stats))
if __name__=='__main__': main()
