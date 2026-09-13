"""Proven shared VGGT model loading and output conversion, extracted from pilot runner."""
from __future__ import annotations
import gc,sys,time
from pathlib import Path
import numpy as np,torch
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

Q = PROJECT_ROOT / "QuantVGGT"
BASE = Q / "VGGT-1B/model_tracker_fixed_e20.pt"

def load_variant(variant,device):
 sys.path.insert(0,str(Q))
 if variant=='full':
  from vggt.models.vggt import VGGT
  m=VGGT();s=torch.load(BASE,map_location='cpu');m.load_state_dict(s);del s;gc.collect();return m.eval().to(device)
 from w4a4_memory_optimized_loader import load_optimized_w4a4
 return load_optimized_w4a4(device)
def infer(model,images,frames,device='cuda:0'):
 sys.path.insert(0,str(Q));from vggt.utils.geometry import unproject_depth_map_to_point_map;from vggt.utils.pose_enc import pose_encoding_to_extri_intri
 torch.cuda.empty_cache();x=images.to(device);torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);t=time.perf_counter()
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  p=model(x);E,K=pose_encoding_to_extri_intri(p['pose_enc'],tuple(x.shape[-2:]))
 torch.cuda.synchronize(device); elapsed=time.perf_counter()-t
 arr={'frame_numbers':np.asarray(frames,dtype=np.int64),'depth':p['depth'][0].detach().float().cpu().numpy(),'intrinsic':K[0].detach().float().cpu().numpy(),'extrinsic':E[0].detach().float().cpu().numpy()}
 arr['world_points_from_depth']=np.ascontiguousarray(unproject_depth_map_to_point_map(arr['depth'],arr['extrinsic'],arr['intrinsic']).astype(np.float32))
 for k in ('depth','intrinsic','extrinsic','world_points_from_depth'):
  if not np.isfinite(arr[k]).all():raise RuntimeError(f'nonfinite {k}')
 return arr,{'seconds':elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved(device)/2**20}
