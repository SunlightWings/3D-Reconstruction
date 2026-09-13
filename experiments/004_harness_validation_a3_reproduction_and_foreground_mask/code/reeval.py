"""Re-score already-rendered sweep outputs under a chosen mask (content|foreground)."""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,"/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common
dv, dis = common.dv, common.dis
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")

def fgmask(rec):
    h,w=[int(v) for v in rec["image"]["size"]]
    nw,nh,left,top,_=dv.preprocess_geometry(w,h)
    tr={"resized_width":nw,"resized_height":nh,"padding":{"left":left,"top":top}}
    with Image.open(dv.CO3D_ROOT/rec["mask"]["path"]) as im: raw=np.asarray(im.convert("L"))
    d=np.ones_like(raw,dtype=np.float32)
    _a,_b,m=dis.resize_depth_and_masks(d,d>0,raw,tr)
    return (m>=0.5) & dv.content_mask_from_record(rec)

def score(tag, it, masktype):
    rows=[]
    for name in common.EXPENSIVE_SCENES:
        cat,seq=common.split_scene(name)
        rc=SWEEP/tag/cat/seq/"train"/f"ours_{it}"/"renders"
        if not rc.is_dir(): return None
        man=common.scene_manifest(cat,seq); recs=dv.load_annotations(cat,seq)
        held=man["heldout_frames"]
        gt=[common.scene_root(cat,seq)/"common"/"heldout_images"/f"frame{f:06d}.png" for f in held]
        masks=[(fgmask(recs[f]) if masktype=="foreground" else dv.content_mask_from_record(recs[f])) for f in held]
        rp=sorted(rc.glob("*.png"))
        if len(rp)!=len(gt): return None
        r=common.evaluate_renders(gt,rp,masks,with_ssim=True)
        rows.append({"scene":name,"psnr":r["mean_psnr"],"ssim":r.get("mean_ssim")})
    return {"mean_psnr":float(np.mean([x["psnr"] for x in rows])),
            "mean_ssim":float(np.mean([x["ssim"] for x in rows if x["ssim"] is not None])),
            "min_psnr":float(np.min([x["psnr"] for x in rows])),"per_scene":rows}

if __name__=="__main__":
    tag=sys.argv[1]; masktype=sys.argv[2]; its=[int(x) for x in sys.argv[3:]]
    for it in its:
        s=score(tag,it,masktype)
        if s is None: print(f"{tag} it={it} [{masktype}]: renders missing"); continue
        print(f"{tag} it={it} [{masktype}]: MEAN PSNR={s['mean_psnr']:.3f} dB  SSIM={s['mean_ssim']:.4f}  min={s['min_psnr']:.2f}")
        for r in s["per_scene"]: print(f"    {r['scene']:32s} {r['psnr']:6.2f}  {r['ssim']:.4f}")
        (SWEEP/"results").mkdir(parents=True,exist_ok=True)
        (SWEEP/"results"/f"{tag}_it{it}_{masktype}.json").write_text(json.dumps(s,indent=2))
