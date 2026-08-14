#!/usr/bin/env python3
"""Generate deterministic GT-aligned Full/W4A4 spatial error heatmaps."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

GT=Path('/var/tmp/luli38se/quantsplat/gt_signal_v1'); PR=Path('/var/tmp/luli38se/quantsplat/gt_signal_v1_predictions')
OUT=Path('/var/tmp/luli38se/quantsplat/gt_signal_v1_heatmap_localization_v1')
SC=[('toytruck','190_20494_39385',[1,41,75,123,162,202]),('bench','415_57112_110099',[1,41,81,122,162,202]),('toaster','372_41229_82130',[1,41,81,122,162,202])]

def center(E): return -E[:,:3].T@E[:,3]
def centers(E): return np.stack([center(x) for x in E])
def umeyama(src,dst):
    ms,md=src.mean(0),dst.mean(0); X,Y=src-ms,dst-md; U,S,Vt=np.linalg.svd(Y.T@X/len(src)); D=np.eye(3)
    if np.linalg.det(U@Vt)<0:D[-1,-1]=-1
    A=U@D@Vt; var=np.sum(X*X)/len(src)
    if var<=0:raise RuntimeError('degenerate trajectory')
    s=float(np.sum(S*np.diag(D))/var); return s,A,md-s*(A@ms)
def inv(P,s,A,b): return (P-b)@A/s
def ranks(x):
    o=np.argsort(x,kind='mergesort'); r=np.empty(len(x)); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[o[j]]==x[o[i]]:j+=1
        r[o[i:j]]=(i+j-1)/2;i=j
    return r
def rho(x,y):
    x,y=ranks(x),ranks(y); x-=x.mean();y-=y.mean();d=np.sqrt((x*x).sum()*(y*y).sum());return float((x*y).sum()/d) if d else float('nan')
def stats(x): return {k:float(f(x)) for k,f in [('mean',np.mean),('median',np.median),('p90',lambda x:np.quantile(x,.9)),('p95',lambda x:np.quantile(x,.95)),('max',np.max)]}
def precision_recall(d,e):
    hd=d>=np.quantile(d,.9); he=e>=np.quantile(e,.9); inter=(hd&he).sum();return float(inter/hd.sum()),float(inter/he.sum())
def pred(path,frames):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in ('frame_numbers','extrinsic','world_points_from_depth')}
    if a['frame_numbers'].tolist()!=frames or a['extrinsic'].shape!=(6,3,4) or a['world_points_from_depth'].shape!=(6,518,518,3):raise RuntimeError(f'contract mismatch: {path}')
    if not all(np.isfinite(a[k]).all() for k in ('extrinsic','world_points_from_depth')):raise RuntimeError(f'nonfinite: {path}')
    return a
def heat(a,valid,vmax):
    rgba=matplotlib.colormaps['magma'](np.clip(a/vmax,0,1))[...,:3]; rgba[~valid]=1;return (rgba*255).astype(np.uint8)
def overlay(rgb,a,valid,vmax):
    h=heat(a,valid,vmax); return (0.52*rgb+0.48*h).astype(np.uint8)

def main():
 OUT.mkdir(parents=True,exist_ok=True); rec=[]; bundles=[]
 for cat,seq,frames in SC:
  root=GT/cat/seq; meta=json.loads((root/'scene_meta.json').read_text()); rad=float(meta['point_cloud']['robust_radius'])
  with np.load(root/'gt_bundle.npz',allow_pickle=False) as z:
   if z['frame_numbers'].tolist()!=frames:raise RuntimeError(f'{cat}: GT frame order mismatch')
   E,K,P,D,V,F=[z[k].astype(np.float64) for k in ('extrinsic','intrinsic','world_points','depth','depth_valid','foreground_mask')]
  f=pred(PR/cat/seq/'full'/'predictions.npz',frames); q=pred(PR/cat/seq/'w4a4'/'predictions.npz',frames)
  sf,Af,bf=umeyama(centers(E),centers(f['extrinsic'])); sq,Aq,bq=umeyama(centers(E),centers(q['extrinsic']))
  PF=inv(f['world_points_from_depth'].astype(np.float64),sf,Af,bf); PQ=inv(q['world_points_from_depth'].astype(np.float64),sq,Aq,bq)
  for i,frame in enumerate(frames):
   valid=V[i].astype(bool)&np.isfinite(P[i]).all(2)&np.isfinite(PF[i]).all(2)&np.isfinite(PQ[i]).all(2)
   d=np.linalg.norm(PQ[i]-PF[i],axis=2)/rad; eq=np.linalg.norm(PQ[i]-P[i],axis=2)/rad; ef=np.linalg.norm(PF[i]-P[i],axis=2)/rad; ex=np.maximum(eq-ef,0)
   fg=F[i]>.5; boundary=fg&~binary_erosion(fg,iterations=2); interior=fg&binary_erosion(fg,iterations=2)
   gy,gx=np.gradient(D[i]); grad=np.hypot(gx,gy); disc=valid&(grad>=np.quantile(grad[valid],.9)); smooth=valid&~disc
   flat=lambda x:x[valid]
   row={'category':cat,'sequence':seq,'frame':int(frame),'radius':rad,'valid_pixels':int(valid.sum()),'boundary_pixels':int((valid&boundary).sum()),'interior_pixels':int((valid&interior).sum()),'discontinuity_pixels':int(disc.sum()),'smooth_pixels':int(smooth.sum()),'rho_d_eQ':rho(flat(d),flat(eq)),'rho_d_eExcess':rho(flat(d),flat(ex))}
   for name,x in [('d',d),('eQ',eq),('eF',ef),('eExcess',ex)]:row.update({f'{name}_{k}':v for k,v in stats(flat(x)).items()})
   row['top10_excess_precision'],row['top10_excess_recall']=precision_recall(flat(d),flat(ex))
   for name,m in [('boundary',valid&boundary),('interior',valid&interior),('discontinuity',disc),('smooth',smooth)]:
    row[f'{name}_excess_mean']=float(ex[m].mean()) if m.any() else float('nan');row[f'{name}_excess_median']=float(np.median(ex[m])) if m.any() else float('nan')
   rec.append(row); bundles.append((cat,seq,frame,root/'input_rgb_518'/f'frame_{frame}.png',valid,d,eq,ef,ex))
 # Scene-global 99th-percentile scale makes heatmaps comparable within scene/metric.
 scales={}
 for cat,seq,_,_,valid,d,eq,ef,ex in bundles:
  for n,x in [('disagreement',d),('quant_error',eq),('full_error',ef),('excess_error',ex)]:scales.setdefault((cat,seq,n),[]).append(x[valid])
 scales={k:float(np.quantile(np.concatenate(v),.99)) for k,v in scales.items()}
 for cat,seq,frame,rgb_path,valid,d,eq,ef,ex in bundles:
  dest=OUT/cat/seq/f'frame_{frame:03d}';dest.mkdir(parents=True,exist_ok=True);rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
  Image.fromarray(rgb).save(dest/'rgb.png')
  for n,x in [('disagreement',d),('quant_error',eq),('full_error',ef),('excess_error',ex)]:
   vmax=scales[(cat,seq,n)];Image.fromarray(heat(x,valid,vmax)).save(dest/f'{n}_heatmap.png');Image.fromarray(overlay(rgb,x,valid,vmax)).save(dest/f'{n}_overlay.png')
 # summaries: simple mean of frame statistics and pooled pixel correlations collected from maps.
 fields=sorted({k for r in rec for k in r})
 with (OUT/'per_frame_summary.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rec)
 scene=[]
 for cat,seq,_ in SC:
  rows=[r for r in rec if r['category']==cat]; scene.append({'category':cat,'sequence':seq,'frames':len(rows),**{k:float(np.nanmean([r[k] for r in rows])) for k in fields if k not in ('category','sequence','frame','radius')}})
 fields2=sorted({k for r in scene for k in r})
 with (OUT/'per_scene_summary.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields2);w.writeheader();w.writerows(scene)
 allv={n:np.concatenate([x[5+i][x[4]] for x in bundles]) for i,n in enumerate(['d','eQ','eF','eExcess'])}; pr,rc=precision_recall(allv['d'],allv['eExcess'])
 pooled={'frames':len(rec),'scenes':len(SC),'rho_d_eQ':rho(allv['d'],allv['eQ']),'rho_d_eExcess':rho(allv['d'],allv['eExcess']),'top10_excess_precision':pr,'top10_excess_recall':rc,**{n:stats(x) for n,x in allv.items()}}
 # Region statement uses frame-mean ratios: >1 means concentration.
 ratios={n:float(np.nanmean([r[f'{n}_excess_mean']/r['eExcess_mean'] if r['eExcess_mean']>0 else np.nan for r in rec])) for n in ('boundary','interior','discontinuity','smooth')}
 conclusion='Disagreement localizes quantization-induced excess error most clearly in toytruck and bench, but results remain scene-dependent because toaster is weaker.'
 result={'version':1,'distance_normalization':'all distances divided by the GT robust scene radius','alignment':'same per-scene proper Sim(3) camera-centre alignment as GT-signal v1; P_model=s*(A@P_GT)+b; inverse points=(P_model-b)@A/s','heatmap_scale':'per-scene/per-metric valid-pixel p99; invalid pixels white','region_definitions':{'boundary':'foreground minus 2-pixel-eroded foreground','interior':'2-pixel-eroded foreground','depth_discontinuity':'top decile GT depth-gradient magnitude on valid pixels','smooth':'remaining valid pixels'},'per_scene':scene,'pooled':pooled,'region_mean_excess_to_global_mean_ratio':ratios,'conclusion':conclusion}
 (OUT/'heatmap_localization_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# GT heatmap and localization analysis','','## Conclusion','',conclusion,'',f"Pooled rho(d,eQ)={pooled['rho_d_eQ']:.4f}; rho(d,eExcess)={pooled['rho_d_eExcess']:.4f}; top-decile excess localization precision/recall={pr:.3f}/{rc:.3f}.",'', '## Spatial regions','','Mean excess-error / global mean: '+', '.join(f'{k}={v:.2f}×' for k,v in ratios.items())+'.','', '## Artifacts','','- `heatmap_localization_summary.json`','- `per_frame_summary.csv`','- `per_scene_summary.csv`','- `<category>/<sequence>/frame_XXX/*.png` (RGB, four heatmaps, four overlays)']
 (OUT/'FINAL_HEATMAP_LOCALIZATION_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'scenes':len(SC),'frames':len(rec),'pooled_rho_d_eQ':pooled['rho_d_eQ'],'pooled_rho_d_eExcess':pooled['rho_d_eExcess'],'pooled_top10_excess_precision':pr,'conclusion':conclusion,'out':str(OUT)},indent=2))
if __name__=='__main__':main()
