#!/bin/bash
# Oracle-only 3DGS configuration sweep, one factor at a time.
# Metric is A3's: content-mask PSNR/SSIM on the 9 held-out views, 8-scene subset.
# Results land in /var/tmp/luli38se/quantsplat/oracle_sweep/results/<tag>.json and are
# written up as ongoing_logs.md entries #003 and #005-#013.
#
# Prep the non-oracle sources first:
#   python3 prep.py views12          # 12 input views
#   python3 prep.py maskedbg         # foreground-masked training images
#   python3 prep_depth.py            # GT inverse-depth maps for depth regularisation
#   python3 prep_bgshell.py          # GT foreground points + 100k random background points
#   python3 prep_views12_depth.py    # 12 views AND depth regularisation
#   python3 -c "import prep; prep.build('views24', n_views=24)"
set -u
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/var/tmp/luli38se/quantsplat/oracle_sweep/logs
mkdir -p "$LOG"

run () {
  tag=$1; src=$2; shift 2
  echo "=== START $tag ($(date +%H:%M)) ==="
  python3 -u "$SP/sweep.py" --tag "$tag" --iters 500 1000 3000 7000 --source "$src" \
      ${1+--extra "$@"} >> "$LOG/$tag.log" 2>&1
  echo "=== DONE $tag rc=$? ($(date +%H:%M)) ==="
}

# baseline / factor 2 (iterations): the checkpoints themselves are the sweep
run base_itersweep   oracle
# factor 1: densification schedule
run densify_off      oracle --densify_until_iter 0
run densify_3000     oracle --densify_until_iter 3000
# factor 3: opacity reset disabled (interval > iterations, so it never fires)
run no_opacity_reset oracle --opacity_reset_interval 100000
# factor 4: input view count
run views12          views12
run views24          views24
# factor 5: foreground-masked training images, constant black background
run maskedbg         maskedbg
# factor 6: depth regularisation against GT depth (foreground-only coverage)
run depthreg         depthreg -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
# extra: give the background an initialisation (NOT a pure GT-geometry oracle)
run bgshell          bgshell
# best combination found: 12 views + depth regularisation
run views12_depth    views12_depth -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01

echo "ALL CONFIGS COMPLETE"
