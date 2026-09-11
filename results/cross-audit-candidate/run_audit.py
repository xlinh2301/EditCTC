#!/usr/bin/env python3
"""Read-only audit of the precomputed cross/indomain branch audits."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_eval_s1024')
OUT = Path(__file__).resolve().parent

def read_labels(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            name, text = line.rstrip('\n').split('\t')[:2]
            out[name] = text
    return out

def read_preds(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            p, text, *rest = line.rstrip('\n').split('\t')
            out[Path(p).name] = text
    return out

def edit_ops(src: str, tgt: str):
    # deterministic Levenshtein alignment, returning S/D/I/M counts
    n, m = len(src), len(tgt)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j] = min(d[i-1][j-1] + (src[i-1] != tgt[j-1]), d[i-1][j] + 1, d[i][j-1] + 1)
    i,j=n,m; c=Counter()
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (src[i-1] != tgt[j-1]):
            c['M' if src[i-1] == tgt[j-1] else 'S'] += 1; i-=1; j-=1
        elif i and d[i][j] == d[i-1][j] + 1:
            c['D'] += 1; i-=1
        else:
            c['I'] += 1; j-=1
    return c

def emitted_stats(trace):
    vals=[]; prev=None
    for fr in trace:
        tok=fr.get('top1_token','')
        if tok and tok != prev:
            vals.append((float(fr.get('top1_probability',0.0)), float(fr.get('top1_top2_margin',0.0))))
        prev=tok
    if not vals: return {'emitted':0, 'mean_prob':None, 'min_prob':None, 'mean_margin':None, 'min_margin':None}
    ps=[x[0] for x in vals]; ms=[x[1] for x in vals]
    return {'emitted':len(vals), 'mean_prob':sum(ps)/len(ps), 'min_prob':min(ps), 'mean_margin':sum(ms)/len(ms), 'min_margin':min(ms)}

def quant(v):
    v=sorted(x for x in v if x is not None)
    if not v: return {}
    def q(p): return v[min(len(v)-1, int(p*(len(v)-1)))]
    return {'n':len(v), 'mean':sum(v)/len(v), 'q10':q(.1), 'median':q(.5), 'q90':q(.9)}

def audit(split):
    p=BASE/split
    labels = read_labels(Path('/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Cross-data/crops/crossdata_label.txt')) if split=='crossdata' else read_labels(Path('/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain/label_backups/test_label_20260802_165337.txt'))
    preds=read_preds(p/'predictions.txt')
    rows=[]; missing=[]; mismatches=[]
    with (p/'branch_audit.jsonl').open() as f:
        for line in f:
            r=json.loads(line); name=Path(r['file']).name
            gt=labels.get(name)
            if gt is None: missing.append(name); continue
            if preds.get(name) and preds[name] != r['ctc']['text']: mismatches.append(name)
            r['_gt']=gt; rows.append(r)
    seq_cat=Counter(); op=Counter(); by_correct=defaultdict(list)
    ctc_conf={True:[],False:[]}; seed_len={True:[],False:[]}; frame_prob={True:[],False:[]}; frame_margin={True:[],False:[]}
    lengths=Counter(); low_margin=[]
    for r in rows:
        gt=r['_gt']; ctc=r['ctc']['text']; ok=ctc==gt
        e=edit_ops(ctc,gt)
        for k,v in e.items(): op[k]+=v
        if ok: cat='correct'
        elif e['S'] and not e['D'] and not e['I']: cat='substitution_only'
        elif e['D'] and e['I']: cat='mixed_length'
        elif e['D']: cat='deletion_only'
        elif e['I']: cat='insertion_only'
        else: cat='other'
        seq_cat[cat]+=1
        lengths[(len(gt),len(ctc))]+=1
        ctc_conf[ok].append(r['ctc'].get('confidence'))
        seed_len[ok].append(len(ctc))
        es=emitted_stats(r['ctc'].get('frame_trace',[]))
        frame_prob[ok].append(es['min_prob']); frame_margin[ok].append(es['min_margin'])
        lcb_ok=r['lcb'].get('predicted_length')==len(gt)
        by_correct['lcb_length_correct' if lcb_ok else 'lcb_length_wrong'].append(r)
        if not ok: low_margin.append({'file':r['file'],'gt':gt,'ctc':ctc,'ctc_conf':r['ctc'].get('confidence'),'min_emitted_margin':es['min_margin'],'edit':dict(e)})
    final_ok=sum(r['final']['text']==r['_gt'] for r in rows)
    nrtr_ok=sum(r['nrtr']['text']==r['_gt'] for r in rows)
    lcb_ok=sum(r['lcb'].get('predicted_length')==len(r['_gt']) for r in rows)
    out={'split':split,'n_rows':len(rows),'label_count':len(labels),'missing_rows':len(missing),'prediction_audit_mismatches':len(mismatches),
         'sequence':{'ctc_correct':sum(r['ctc']['text']==r['_gt'] for r in rows),'ctc_accuracy':sum(r['ctc']['text']==r['_gt'] for r in rows)/len(rows),
                     'nrtr_correct':nrtr_ok,'nrtr_accuracy':nrtr_ok/len(rows),'nerd_final_correct':final_ok,'nerd_final_accuracy':final_ok/len(rows),
                     'nerd_changed':sum(r['nerd']['changed_positions']>0 for r in rows),'nerd_helped':sum(r['ctc']['text']!=r['_gt'] and r['final']['text']==r['_gt'] for r in rows),
                     'nerd_hurt':sum(r['ctc']['text']==r['_gt'] and r['final']['text']!=r['_gt'] for r in rows),'lcb_length_correct':lcb_ok,'lcb_length_accuracy':lcb_ok/len(rows)},
         'natural_error_taxonomy':dict(seq_cat),'aligned_char_ops':dict(op),'length_pairs':{f'{a}->{b}':n for (a,b),n in sorted(lengths.items())},
         'confidence':{'ctc':{('correct' if k else 'wrong'):quant(v) for k,v in ctc_conf.items()},'min_emitted_prob':{('correct' if k else 'wrong'):quant(v) for k,v in frame_prob.items()},'min_emitted_margin':{('correct' if k else 'wrong'):quant(v) for k,v in frame_margin.items()}},
         'candidate_topk':{'available':False,'reason':'branch_audit frame_trace stores only top1 token/probability and top1-top2 margin; no top-3/top-5 token identities.'},
         'lowest_margin_wrong_examples':sorted(low_margin,key=lambda x: (x['min_emitted_margin'] if x['min_emitted_margin'] is not None else 1.0))[:20]}
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reports={s:audit(s) for s in ('crossdata','indomain_test')}
    (OUT/'audit.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False)+'\n')
    c=reports['crossdata']; i=reports['indomain_test']
    lines=['# Cross-data candidate/error audit','', 'Read-only audit of precomputed `branch_audit.jsonl`; no training or checkpoint changes.', '',
           '| Metric | Cross-data | Indomain test |', '|---|---:|---:|']
    keys=[('n_rows','Rows'),('sequence.ctc_correct','CTC exact'),('sequence.ctc_accuracy','CTC accuracy'),('sequence.nrtr_accuracy','NRTR accuracy'),('sequence.nerd_final_accuracy','NERD final accuracy'),('sequence.nerd_changed','NERD changed'),('sequence.nerd_helped','NERD helped'),('sequence.nerd_hurt','NERD hurt'),('sequence.lcb_length_accuracy','LCB length accuracy')]
    def get(d,k):
        for x in k.split('.'): d=d[x]
        return d
    for k,label in keys:
        a,b=get(c,k),get(i,k); fmt=lambda x:f'{x:.4f}' if isinstance(x,float) else str(x)
        lines.append(f'| {label} | {fmt(a)} | {fmt(b)} |')
    lines += ['', '## Natural error taxonomy', '', '| Category | Cross-data | Indomain test |', '|---|---:|---:|']
    cats=sorted(set(c['natural_error_taxonomy'])|set(i['natural_error_taxonomy']))
    for x in cats: lines.append(f"| {x} | {c['natural_error_taxonomy'].get(x,0)} | {i['natural_error_taxonomy'].get(x,0)} |")
    lines += ['', '## Aligned character operations', '', '| Operation | Cross-data | Indomain test |', '|---|---:|---:|']
    for x in sorted(set(c['aligned_char_ops'])|set(i['aligned_char_ops'])): lines.append(f"| {x} | {c['aligned_char_ops'].get(x,0)} | {i['aligned_char_ops'].get(x,0)} |")
    lines += ['', '## Interpretation', '',
              '- The cross split has substantially more natural CTC errors than the indomain split, so it is a stronger stress test for correction.',
              '- The precomputed NERD branch is KEEP-only on both splits; therefore its zero changed/helped/hurt is an inference behavior, not evidence that cross-data has no recoverable errors.',
              '- Top-3/Top-5 candidate coverage cannot be recomputed from these files because only frame top-1 identities are persisted. A fresh inference dump with full top-k is required for that specific E34/E35 oracle.',
              '- Confidence and margin summaries are provided to determine whether cross errors are low-confidence/ambiguous; these are the available posterior diagnostics without retraining.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cross':c['sequence'],'indomain':i['sequence'],'taxonomy_cross':c['natural_error_taxonomy'],'taxonomy_indomain':i['natural_error_taxonomy'],'out':str(OUT)},indent=2))
if __name__=='__main__': main()
