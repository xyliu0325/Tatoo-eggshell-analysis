#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

EXCEL_ROOT="${EXCEL_ROOT:-$PROJECT_ROOT/input_data/excel}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_ROOT/input_data/image}"
PRETRAINED_SWIN="${PRETRAINED_SWIN:-$PROJECT_ROOT/pretrained/swin_base_patch4_window7_224_22k.pth}"

OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/runs}"
STAGE1_OUT="$OUT_ROOT/stage1"
FM_FLAT="$OUT_ROOT/fm_flat"

if [ -n "${FIXED_SEED:-}" ]; then
    SEED="$FIXED_SEED"
else
    SEED=$(python -c "import secrets; print(secrets.randbelow(900000) + 100000)")
fi

STAGE2_OUT="$OUT_ROOT/stage2_seed_${SEED}"

N_TRIALS="${N_TRIALS:-100}"
N_SEEDS="${N_SEEDS:-3}"
FEATURE_VERSION="${FEATURE_VERSION:-v2}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
CV_FOLDS="${CV_FOLDS:-1}"

echo "=========================================="
echo "  egg cascade pipeline"
echo "=========================================="
echo "  PROJECT_ROOT     = $PROJECT_ROOT"
echo "  EXCEL_ROOT       = $EXCEL_ROOT"
echo "  IMAGE_ROOT       = $IMAGE_ROOT"
echo "  PRETRAINED_SWIN  = $PRETRAINED_SWIN"
echo "  OUT_ROOT         = $OUT_ROOT"
echo "  CV_FOLDS         = $CV_FOLDS"
echo "  SEED             = $SEED"
echo "------------------------------------------"

python -c "import xgboost, lightgbm, sklearn, skimage, optuna, tqdm, torch, torchvision, timm, PIL, pandas, openpyxl" \
  || { echo "Missing Python dependencies. Install them before running the pipeline (see README.md)."; exit 1; }

[ -d "$EXCEL_ROOT" ] || { echo "Error: EXCEL_ROOT does not exist: $EXCEL_ROOT"; exit 1; }
[ -d "$IMAGE_ROOT" ] || { echo "Error: IMAGE_ROOT does not exist: $IMAGE_ROOT"; exit 1; }
[ -f "$PRETRAINED_SWIN" ] || { echo "Error: PRETRAINED_SWIN does not exist: $PRETRAINED_SWIN"; exit 1; }

echo ""
echo "=========================================="
echo "  [1/3] Stage1: XGBoost (Dead/Alive)"
echo "=========================================="
python train_stage1_v2.py \
  --data_root "$EXCEL_ROOT" \
  --out_dir "$STAGE1_OUT" \
  --feature_version "$FEATURE_VERSION" \
  --mode random \
  --n_trials "$N_TRIALS" \
  --n_seeds "$N_SEEDS" \
  --final_split random

STAGE1_BUNDLE="$STAGE1_OUT/stage1_bundle.pkl"
[ -f "$STAGE1_BUNDLE" ] || { echo "Error: Stage 1 did not produce a model bundle."; exit 1; }
echo "Stage 1 complete: $STAGE1_BUNDLE"

echo ""
echo "=========================================="
echo "  [2/3] Stage2: Prepare the Female/Male image directories"
echo "=========================================="
rm -rf "$FM_FLAT"
mkdir -p "$FM_FLAT/Female" "$FM_FLAT/Male"
for batch_dir in "$IMAGE_ROOT"/*/; do
    b=$(basename "$batch_dir")
    for cls in Female Male; do
        if [ -d "$batch_dir/$cls" ]; then
            for f in "$batch_dir/$cls"/*.png; do
                [ -e "$f" ] || continue
                stem=$(basename "$f" .png)
                ln -sf "$f" "$FM_FLAT/$cls/${stem}__${b}.png"
            done
        fi
    done
done
N_F=$(ls "$FM_FLAT/Female" 2>/dev/null | wc -l)
N_M=$(ls "$FM_FLAT/Male" 2>/dev/null | wc -l)
echo "✓ Female=$N_F  Male=$N_M"
[ "$N_F" -gt 0 ] && [ "$N_M" -gt 0 ] || { echo "Error: no images found for one or both classes."; exit 1; }

echo ""
echo "=========================================="
echo "  [3/3] Stage2: Swin (Female/Male) via stage2_train/train_v2.py"
echo "=========================================="
mkdir -p "$STAGE2_OUT"
ABS_FM_FLAT="$(realpath "$FM_FLAT")"
ABS_PRETRAINED="$(realpath "$PRETRAINED_SWIN")"
ABS_STAGE2_OUT="$(realpath "$STAGE2_OUT")"

python stage2_train/train_v2.py \
  --data-path  "$ABS_FM_FLAT" \
  --weights    "$ABS_PRETRAINED" \
  --out_dir    "$ABS_STAGE2_OUT" \
  --num_classes 2 \
  --epochs      "$EPOCHS" \
  --batch-size  "$BATCH_SIZE" \
  --lr          "$LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --amp_dtype   "$AMP_DTYPE" \
  --cv_folds    "$CV_FOLDS" \
  --seed         "$SEED"

STAGE2_CKPT="$STAGE2_OUT/weights/best.pth"
[ -f "$STAGE2_CKPT" ] || STAGE2_CKPT="$STAGE2_OUT/weights/final_model.pth"
[ -f "$STAGE2_CKPT" ] || { echo "Error: Stage 2 did not produce a checkpoint."; exit 1; }
echo "Stage 2 complete: $STAGE2_CKPT"

echo ""
echo "=========================================="
echo "  [Smoke] Run a cascade prediction smoke test on one class directory"
echo "=========================================="
SMOKE_BATCH=$(ls "$EXCEL_ROOT" | head -1)
SMOKE_CLS=Male
if [ -d "$EXCEL_ROOT/$SMOKE_BATCH/$SMOKE_CLS" ] && [ -d "$IMAGE_ROOT/$SMOKE_BATCH/$SMOKE_CLS" ]; then
    python predict.py \
      --stage1_model "$STAGE1_BUNDLE" \
      --stage2_ckpt "$STAGE2_CKPT" \
      --excel_dir "$EXCEL_ROOT/$SMOKE_BATCH/$SMOKE_CLS" \
      --image_dir "$IMAGE_ROOT/$SMOKE_BATCH/$SMOKE_CLS" \
      --output_csv "$OUT_ROOT/smoke_${SMOKE_BATCH}_${SMOKE_CLS}.csv"
    echo "Smoke test results: $OUT_ROOT/smoke_${SMOKE_BATCH}_${SMOKE_CLS}.csv"
else
    echo "(Skipping the smoke test. Directory not found: $SMOKE_BATCH/$SMOKE_CLS)"
fi

echo ""
echo "=========================================="
echo "  Pipeline complete"
echo "    stage1 bundle: $STAGE1_BUNDLE"
echo "    stage2 ckpt:   $STAGE2_CKPT"
echo "    seed:          $SEED"
echo "    split mode:    single stratified 70/30"
echo "    stage2 output: $STAGE2_OUT"
echo "  Batch prediction:"
echo "    python predict.py \\"
echo "      --stage1_model $STAGE1_BUNDLE \\"
echo "      --stage2_ckpt $STAGE2_CKPT \\"
echo "      --excel_dir <xlsx_directory> --image_dir <png_directory> \\"
echo "      --output_csv <output_csv>"
echo "=========================================="
