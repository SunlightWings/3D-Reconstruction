#!/bin/bash
# Package the VGGT prediction outputs into per-category zips for manual download.
#
# One archive per CO3D category rather than a single ~56 GB file: each is
# independently extractable, so a failed transfer costs one category (~1.6 GB)
# instead of the whole set. Extracting them all into one directory reproduces the
# original tree exactly.
#
# .npz here is uncompressed (np.savez, not savez_compressed), so zip does buy
# something -- about 12% on float32 geometry.
set -u
SRC=/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/run/predictions
DST=/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/run/predictions_zips
mkdir -p "$DST"

cd "$SRC" || exit 1

# small metadata archive first, so it is available while the big ones build
zip -q -9 "$DST/_metadata.zip" README.md index.csv frozen_dataset_manifest.json
echo "[$(date +%H:%M)] _metadata.zip done"

total=$(find . -maxdepth 1 -mindepth 1 -type d | wc -l)
i=0
for d in */; do
  cat="${d%/}"
  i=$((i+1))
  out="$DST/${cat}.zip"
  if [ -f "$out" ]; then
    echo "[$(date +%H:%M)] ($i/$total) $cat: exists, skipping"
    continue
  fi
  # -1 = fastest deflate; the data is float32 so higher levels buy almost nothing
  zip -q -r -1 "$out.part" "$cat" && mv "$out.part" "$out"
  echo "[$(date +%H:%M)] ($i/$total) $cat -> $(du -h "$out" | cut -f1)"
done

echo "[$(date +%H:%M)] ALL ARCHIVES COMPLETE"
du -sh "$DST"
ls -1 "$DST" | wc -l
