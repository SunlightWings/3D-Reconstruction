#!/usr/bin/env python3
"""Single resumable entry point for frozen dataset-wide Full/W4A4 inference."""
from __future__ import annotations
import gc,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np,torch
from prepare_co3d_scene import import_local_preprocessor
from vggt_inference_core import infer,load_variant
B=Path('/var/tmp/luli38se/quantsplat/disagreement_dataset_v1');MAN=B/'frozen_dataset_manifest.json';RUN=B/'run'; EXPECT='1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06'
P,_,S=import_local_preprocessor(Path('/home/utn/luli38se/cv/QuantVGGT'))
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def preprocess(paths):
 kw={'mode':'pad'}
 if 'target_size' in S.parameters:kw['target_size']=518
 x=P([str(p) for p in paths],**kw).detach().cpu().float().contiguous()
 if tuple(x.shape)!=(6,3,518,518) or not torch.isfinite(x).all():raise RuntimeError('bad frozen preprocessing output')
 return x
def semantic(x):return sha_bytes(x.numpy().astype('<f4',copy=False).tobytes(order='C'))
def idle():
 r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().split(',')
 return float(r[0])<=500 and float(r[1])<=20
def valid(npz,meta,variant,frames):
 try:
  q=json.loads(meta.read_text());z=np.load(npz,allow_pickle=False);ok=q['variant']==variant and q['frame_numbers']==frames and q['prediction_sha256']==sha_file(npz) and z['frame_numbers'].tolist()==frames and z['depth'].shape==(6,518,518,1) and z['world_points_from_depth'].shape==(6,518,518,3)
  z.close();return ok
 except Exception:return False
def save(d,variant,arr,meta):
 d.mkdir(parents=True,exist_ok=True);tmp=d/f'.{variant}.tmp.npz';np.savez(tmp,**arr);final=d/f'{variant}.npz';os.replace(tmp,final);meta['prediction_sha256']=sha_file(final);(d/f'{variant}_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
def main():
 if sha_file(MAN)!=EXPECT:raise RuntimeError('frozen manifest SHA mismatch')
 m=json.loads(MAN.read_text());groups=[(s,g) for s in m['scenes'] for g in s['groups']];RUN.mkdir(parents=True,exist_ok=True)
 for variant in ('full','w4a4'):
  if not idle():raise RuntimeError('GPU not idle; refusing inference')
  model=load_variant(variant,'cuda:0');complete=0
  for s,g in groups:
   d=RUN/'predictions'/s['category']/s['sequence']/g['group_id'];npz=d/f'{variant}.npz';met=d/f'{variant}_meta.json';frames=g['frame_numbers']
   if valid(npz,met,variant,frames):complete+=1;continue
   lookup={r['frame_number']:r['image_path'] for r in s['usable_frames']};x=preprocess([lookup[f] for f in frames]);ih=semantic(x)
   if variant=='w4a4':
    fm=json.loads((d/'full_meta.json').read_text());
    if fm['input_semantic_sha256']!=ih:raise RuntimeError('Full/W4 input semantic mismatch')
   arr,rt=infer(model,x,frames);save(d,variant,arr,{'category':s['category'],'sequence':s['sequence'],'group_id':g['group_id'],'frame_numbers':frames,'primary_frame_numbers':g['primary_frame_numbers'],'context_only_frame_numbers':g['context_only_frame_numbers'],'variant':variant,'input_semantic_sha256':ih,'frozen_manifest_sha256':EXPECT,'runtime':rt});complete+=1
   (RUN/'progress.json').write_text(json.dumps({variant:complete,'total':len(groups)},indent=2)+'\n')
  del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
