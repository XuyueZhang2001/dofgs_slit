#!/usr/bin/env bash
# Run the equal-area slit vs disk comparison on several source DOF-GS scenes,
# with the same settings as real_cake, then summarise all scenes.
#
# Per scene:
#   reference   slitsim_<scene>: the original 20 x 2 px slit (area 40 px^2 at the largest
#               defocus for the reference focus at the median inverse depth); it only
#               provides the aperture area and the reference focus and is not trained
#   slit (densmax+L_dir) n = 4, a = 0.4, max_angle densification + directional AiF loss 0.03
#               (code defaults)   -> slit_sim_dir0.03_densmax/slitsim_<scene>_n4_a0.40<sfx>
#   slit (orig) same data, no directional loss, mean densification
#                                 -> slit_sim/slitsim_<scene>_n4_a0.40<sfx>
#   disk        equal-area disk, one capture per pose -> slit_sim/disksim_<scene><sfx>
# Focus (--focus_mode): per_pose (default, <sfx> = _pf): every pose focuses at its own
# random depth (shared by its slit angles); training shares one F per pose and one K.
# global: one focus for all poses (the earlier datasets, <sfx> empty).
# Noise-free, Gaussian PSF (--psf gaussian). Finished steps are skipped, so the
# script can be restarted. The source model iteration is the latest one found in
# output/<model_version>/<model_folder>/<scene>/point_cloud (or --iteration).
#
# Usage (from the repository root, e.g. inside tmux):
#   bash tools/run_all_scenes.sh --gpu 1 --stage simulate          # all 10 real scenes, check K first
#   bash tools/run_all_scenes.sh --gpu 1 --stage train             # then train + evaluate
#   bash tools/run_all_scenes.sh --gpu 0 --stage all real_caps real_cisco   # a subset on another GPU
#   bash tools/run_all_scenes.sh --gpu 1 --focus_mode global        # the earlier global-focus setting
# Stages: simulate (datasets + K check), train (train + evaluate, needs the datasets),
#         all (both). Scenes whose simulation warns that K cannot be represented
#         (K < 101) are not trained.

set -u
cd "$(dirname "$0")/.." || exit 1

GPU=0
STAGE=all
ITER=""
FOCUS=per_pose
SCENES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --iteration) ITER="$2"; shift 2 ;;
        --focus_mode) FOCUS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) SCENES+=("$1"); shift ;;
    esac
done
case "$STAGE" in simulate|train|all) ;; *) echo "--stage must be simulate, train or all"; exit 1 ;; esac
case "$FOCUS" in per_pose) SFX=_pf ;; global) SFX="" ;; *) echo "--focus_mode must be per_pose or global"; exit 1 ;; esac
echo "focus mode: $FOCUS (dataset suffix '${SFX}')"
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(real_cake real_caps real_cisco real_coral real_cupcake real_cups real_daisy real_sausage real_seal real_tools)
fi

LOG_DIR=logs/design_sweep
mkdir -p "$LOG_DIR"
FAILED=()
K_WARN=()
SKIPPED=()

# latest trained iteration of the source DOF-GS model of a scene (empty if none)
source_iteration() {
    local cfg="$1" scene="$2" dir
    dir=$(python -c "from omegaconf import OmegaConf as O; c = O.load('$cfg'); \
print('./output/%s/%s/%s' % (c.get('model_version', 'model_dofgs'), c.model_folder, '$scene'))")
    ls -d "$dir"/point_cloud/iteration_* 2>/dev/null | sed 's/.*iteration_//' | sort -n | tail -1
}

# 0 if any simulation log of the scene warns that K is below the representable range
has_k_warning() {
    local scene="$1"
    grep -qs "WARN\] K =" "$LOG_DIR/slitsim_${scene}_reference.log" \
        "$LOG_DIR/slitsim_${scene}_n4_a0.40${SFX}_simulate.log" "$LOG_DIR/disksim_${scene}${SFX}_simulate.log"
}

for s in "${SCENES[@]}"; do
    echo
    echo "################################ $s ################################"
    cfg="config_files/config_${s}.yaml"
    if [ ! -f "$cfg" ]; then
        echo "[skip] $cfg not found"; SKIPPED+=("$s (no config)"); continue
    fi
    it="${ITER:-$(source_iteration "$cfg" "$s")}"
    if [ -z "$it" ]; then
        echo "[skip] no trained source model of $s found (download the pre-trained DOF-GS scene)"
        SKIPPED+=("$s (no source model)"); continue
    fi
    echo "source model iteration: $it"

    # ---------------- simulate ----------------
    if [ "$STAGE" != "train" ]; then
        if [ ! -f "data/slit_sim/slitsim_${s}/sim_gt.json" ]; then
            echo "--- reference dataset slitsim_${s} (20 x 2 px slit)"
            python tools/simulate_slit_dataset.py --config "$cfg" --scene "$s" --iteration "$it" \
                --n_angles 4 --max_blur_px 20 --width_ratio 0.1 --focus_mode global --new_scene "slitsim_${s}" \
                2>&1 | tee -a "$LOG_DIR/slitsim_${s}_reference.log"
            if [ "${PIPESTATUS[0]}" -ne 0 ]; then FAILED+=("$s (reference)"); continue; fi
        else
            echo "[skip] reference dataset slitsim_${s} exists"
        fi
        python tools/run_design_sweep.py --config "$cfg" --scene "$s" --iteration "$it" --gpu "$GPU" \
            --focus_mode "$FOCUS" --steps simulate || { FAILED+=("$s (simulate)"); continue; }
    fi

    if has_k_warning "$s"; then
        echo "[WARN] $s: K below the representable range (see $LOG_DIR/*${s}*simulate.log); not trained"
        K_WARN+=("$s"); continue
    fi
    [ "$STAGE" = "simulate" ] && continue

    # ---------------- train + evaluate ----------------
    if [ ! -f "data/slit_sim/slitsim_${s}_n4_a0.40${SFX}/sim_gt.json" ] || [ ! -f "data/slit_sim/disksim_${s}${SFX}/sim_gt.json" ]; then
        echo "[skip] datasets of $s missing; run --stage simulate first"; SKIPPED+=("$s (no datasets)"); continue
    fi
    # slit (densmax+L_dir: code defaults) + disk
    python tools/run_design_sweep.py --config "$cfg" --scene "$s" --iteration "$it" --gpu "$GPU" \
        --focus_mode "$FOCUS" || FAILED+=("$s (slit densmax+L_dir / disk)")
    # slit (original model)
    python tools/run_design_sweep.py --config "$cfg" --scene "$s" --iteration "$it" --gpu "$GPU" \
        --focus_mode "$FOCUS" --no_disk --dir_loss 0 --densify_stat mean \
        || FAILED+=("$s (slit original)")
done

echo
echo "================================ done ================================"
[ ${#SKIPPED[@]} -gt 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
[ ${#K_WARN[@]} -gt 0 ] && printf 'K below the representable range (not trained): %s\n' "${K_WARN[@]}"
[ ${#FAILED[@]} -gt 0 ] && printf 'failed: %s\n' "${FAILED[@]}"
if [ "$STAGE" != "simulate" ]; then
    python tools/summarize_scenes.py --suffix "$SFX" --scenes "${SCENES[@]}"
fi
