import sys, shutil
from pathlib import Path
import pycolmap
scene = sys.argv[1]
img = Path(f"/var/tmp/luli38se/vggt_official/examples/{scene}/images")
work = Path(f"/var/tmp/luli38se/llff_colmap/{scene}")
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
db = work / "database.db"
pycolmap.extract_features(db, img)
pycolmap.match_exhaustive(db)
recs = pycolmap.incremental_mapping(db, img, work / "sparse")
for i, r in recs.items():
    print(f"{scene} recon {i}: {r.num_reg_images()}/{len(list(img.glob('*.png')))} images, {r.num_points3D()} points")
    r.write_text(str(work / "sparse" / str(i)))
