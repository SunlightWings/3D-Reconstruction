#!/usr/bin/env bash
set -euo pipefail

CORR="/var/tmp/poli22wo/full & w4a4 outputs"
DOWN="/var/tmp/poli22wo/quantsplat/downstream_validation_v1"
REPO="$HOME/Desktop/UTN/Semester 2/Computer Vision/Final Project/3D-Reconstruction/Clean_w3a3"
GS_PY="$HOME/miniconda3/envs/gs/bin/python"

cd "$REPO/gaussian-splatting"

for CAT_DIR in "$CORR"/*
do
    [ -d "$CAT_DIR" ] || continue

    CAT="$(basename "$CAT_DIR")"

    for SEQ_DIR in "$CAT_DIR"/*
    do
        [ -d "$SEQ_DIR" ] || continue

        SEQ="$(basename "$SEQ_DIR")"
        SCENE="$DOWN/$CAT/$SEQ"

        echo
        echo "=================================================="
        echo "$CAT/$SEQ"
        echo "=================================================="

        for ARM in full w4a4
        do
            SRC_PLY="$SEQ_DIR/$ARM/point_cloud.ply"

            MODEL="$SCENE/models/$ARM"

            DST_PLY="$MODEL/point_cloud/iteration_7000/point_cloud.ply"

            HELDOUT_SOURCE="$SCENE/sources/$ARM/heldout"

            CANONICAL="$SCENE/heldout/${ARM}_it7000/renders"

            if [ ! -f "$SRC_PLY" ]; then
                echo "[$ARM] MISSING PLY"
                continue
            fi

            if [ ! -d "$HELDOUT_SOURCE" ]; then
                echo "[$ARM] MISSING HELDOUT SOURCE"
                continue
            fi

            mkdir -p \
                "$MODEL/point_cloud/iteration_7000"

            cp -f \
                "$SRC_PLY" \
                "$DST_PLY"

            cat > "$MODEL/cfg_args" <<EOF
Namespace(sh_degree=3, source_path='$SCENE/sources/$ARM/train', model_path='$MODEL', images='images', depths='', resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=False)
EOF

            rm -rf \
                "$MODEL/train/ours_7000"

            echo "[$ARM] rendering..."

            "$GS_PY" render.py \
                -m "$MODEL" \
                -s "$HELDOUT_SOURCE" \
                --iteration 7000

            RENDER_DIR="$MODEL/train/ours_7000/renders"

            if [ ! -d "$RENDER_DIR" ]; then
                echo "[$ARM] ERROR: render folder missing"
                continue
            fi

            N=$(find "$RENDER_DIR" \
                -maxdepth 1 \
                -type f \
                -name '*.png' | wc -l)

            if [ "$N" -ne 9 ]; then
                echo "[$ARM] ERROR: expected 9 renders, got $N"
                continue
            fi

            rm -rf "$CANONICAL"
            mkdir -p "$CANONICAL"

            cp -f \
                "$RENDER_DIR"/*.png \
                "$CANONICAL/"

            echo "[$ARM] done: 9 renders"
        done
    done
done

echo
echo "ALL DONE"
