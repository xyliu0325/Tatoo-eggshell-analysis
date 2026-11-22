"""Train and evaluate the Stage 1 dead/alive ensemble using four model families.

Export base-model, ensemble, and stacking metrics, prediction tables, ROC and PR curves,
confusion matrices, calibration curves, feature importance, and dye-dot contribution maps.
"""
import os, json, argparse, random, pickle, sys, warnings, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from PIL import Image
from scipy import ndimage
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_fscore_support, matthews_corrcoef,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from xgb_data import load_excel_features

warnings.filterwarnings("ignore", category=UserWarning)

CLASS_NAMES = ["Alive", "Dead"]
ENSEMBLE_WEIGHTS = {
    "xgb": 0.40,
    "lgb": 0.20,
    "rf": 0.20,
    "et": 0.20,
}
MODEL_LABELS = {
    "xgb": "XGBoost", "lgb": "LightGBM",
    "rf": "Random Forest", "et": "Extra Trees",
    "ens": "XGB-weighted Ensemble", "stk": "Stacking",
}


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "split"


def _feature_spots(feature_name):
    """Return (spot, fraction) pairs, or an empty list for a global statistic.

    Assign p19_* to spot 19. Split rgb_d19_* equally between spots 19 and 20.
    """
    m = re.match(r"^p(\d{2})_", feature_name)
    if m:
        return [(int(m.group(1)), 1.0)]
    m = re.match(r"^(?:rgb|hsv|lab)_d(\d{2})_", feature_name)
    if m:
        left = int(m.group(1))
        return [(left, 0.5), (left + 1, 0.5)]
    return []


def _feature_space(feature_name):
    if re.match(r"^p\d{2}_[RGB]$", feature_name):
        return "RGB"
    if re.match(r"^p\d{2}_[HSV]$", feature_name):
        return "HSV"
    if re.match(r"^p\d{2}_(?:L|a|b)$", feature_name):
        return "Lab"
    m = re.match(r"^(rgb|hsv|lab)_d\d{2}_", feature_name)
    return f"{m.group(1).upper()} difference" if m else "Global statistics"


def _mean_family_importance(clfs, n_features):
    vals = []
    for clf in clfs:
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None and len(imp) == n_features:
            imp = np.asarray(imp, dtype=float)
            vals.append(imp / max(imp.sum(), 1e-12))
    return np.mean(vals, axis=0) if vals else np.zeros(n_features, dtype=float)


def _predict_base_and_combined(models, stack_meta, X):
    probs = {}
    for tag in ("xgb", "lgb", "rf", "et"):
        probs[tag] = np.mean(
            [clf.predict_proba(X)[:, 1] for clf in models[tag]], axis=0)
    probs["ens"] = sum(
        ENSEMBLE_WEIGHTS[tag] * probs[tag]
        for tag in ("xgb", "lgb", "rf", "et"))
    if stack_meta is not None:
        meta_x = np.column_stack([probs[t] for t in ("xgb", "lgb", "rf", "et")])
        probs["stk"] = stack_meta.predict_proba(meta_x)[:, 1]
    return probs


def _metric_row(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0)
    has_both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if has_both else np.nan,
        "average_precision": float(average_precision_score(y_true, prob)) if has_both else np.nan,
        "dead_precision": float(precision[0]),
        "dead_recall_sensitivity": float(recall[0]),
        "specificity": float(tn / max(tn + fp, 1)),
        "dead_f1": float(f1[0]),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tn_alive_correct": int(tn),
        "fp_alive_as_dead": int(fp),
        "fn_dead_as_alive": int(fn),
        "tp_dead_correct": int(tp),
    }


def _write_highlighted_prediction_excel(
        df, path, true_col="true_label", predicted_cols=None,
        final_correct_col="final_correct"):
    """Export predictions and highlight incorrect model cells and final prediction rows in yellow."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    predicted_cols = predicted_cols or []
    df.to_excel(path, index=False, sheet_name="Predictions")
    wb = load_workbook(path)
    ws = wb["Predictions"]
    headers = {cell.value: cell.column for cell in ws[1]}
    yellow = PatternFill(fill_type="solid", fgColor="FFF2CC")
    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    true_idx = headers.get(true_col)
    final_correct_idx = headers.get(final_correct_col)
    pred_indices = [headers[c] for c in predicted_cols if c in headers]
    for row in range(2, ws.max_row + 1):
        true_label = ws.cell(row, true_idx).value if true_idx else None
        final_wrong = (
            final_correct_idx is not None
            and ws.cell(row, final_correct_idx).value in (False, 0, "False", "FALSE"))
        if final_wrong:
            for col in range(1, ws.max_column + 1):
                ws.cell(row, col).fill = yellow
        else:
            for col in pred_indices:
                if ws.cell(row, col).value != true_label:
                    ws.cell(row, col).fill = yellow

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col in range(1, ws.max_column + 1):
        values = [str(ws.cell(r, col).value or "") for r in range(1, min(ws.max_row, 200) + 1)]
        ws.column_dimensions[get_column_letter(col)].width = min(
            max(max(map(len, values)) + 2, 10), 34)

    legend = wb.create_sheet("Legend")
    legend["A1"] = "Yellow row/cell"
    legend["A1"].fill = yellow
    legend["B1"] = "Incorrect prediction; a full yellow row means the final selected model was wrong."
    legend.column_dimensions["A"].width = 20
    legend.column_dimensions["B"].width = 90
    wb.save(path)


def _save_confusion_plot(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    cutoff = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cutoff else "black",
                    fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_spot_heatmap(values, title, path):
    arr = np.asarray(values, dtype=float).reshape(1, 20)
    fig, ax = plt.subplots(figsize=(13, 2.8))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(20), [str(i) for i in range(1, 21)])
    ax.set_yticks([0], ["Contribution"])
    ax.set_xlabel("Dye spot")
    ax.set_title(title)
    for j, v in enumerate(arr[0]):
        ax.text(j, 0, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v > np.nanmax(arr) * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Normalized contribution (%)",
                 fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _spot_tables(feature_names, importance):
    spot = np.zeros(20, dtype=float)
    spaces = ["RGB", "HSV", "Lab", "RGB difference", "HSV difference", "LAB difference"]
    by_space = {s: np.zeros(20, dtype=float) for s in spaces}
    global_importance = 0.0
    for name, value in zip(feature_names, importance):
        targets = _feature_spots(name)
        if not targets:
            global_importance += float(value)
            continue
        space = _feature_space(name)
        for s, frac in targets:
            spot[s - 1] += float(value) * frac
            if space in by_space:
                by_space[space][s - 1] += float(value) * frac
    pct = spot / max(spot.sum(), 1e-12) * 100.0
    df_spot = pd.DataFrame({
        "spot": np.arange(1, 21),
        "attributable_importance": spot,
        "normalized_contribution_percent": pct,
    })
    df_space = pd.DataFrame({"spot": np.arange(1, 21), **by_space})
    return df_spot, df_space, global_importance


def _aggregate_feature_values_to_spots(feature_names, feature_values):
    """Map feature contributions to the 20 dye-dot positions."""
    values = np.asarray(feature_values, dtype=float)
    out = np.zeros(values.shape[:-1] + (20,), dtype=float)
    for col, name in enumerate(feature_names):
        for spot, fraction in _feature_spots(name):
            out[..., spot - 1] += values[..., col] * fraction
    return out


def _find_original_image(image_root, batch, stem):
    if not image_root:
        return None
    patterns = [
        os.path.join(image_root, str(batch), "*", f"{stem}.png"),
        os.path.join(image_root, str(batch), "*", f"{stem}.jpg"),
        os.path.join(image_root, str(batch), "*", f"{stem}.jpeg"),
        os.path.join(image_root, str(batch), f"{stem}.png"),
    ]
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(pattern))
    return sorted(set(hits))[0] if hits else None


def _fit_regular_grid_axis(indices, values, n_positions,
                           fallback_step, fallback_offset):
    """Fit an evenly spaced one-dimensional grid to reliable points and reject outliers."""
    indices = np.asarray(indices, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(indices) & np.isfinite(values)
    indices, values = indices[finite], values[finite]

    if len(indices) >= 2 and len(np.unique(indices)) >= 2:
        slope, intercept = np.polyfit(indices, values, 1)
        residual = values - (intercept + slope * indices)
        median_residual = np.median(residual)
        mad = np.median(np.abs(residual - median_residual))
        tolerance = max(2.5 * 1.4826 * mad, 0.12 * abs(slope), 2.0)
        keep = np.abs(residual - median_residual) <= tolerance
        if keep.sum() >= 2 and len(np.unique(indices[keep])) >= 2:
            slope, intercept = np.polyfit(indices[keep], values[keep], 1)
    elif len(indices) == 1:
        slope = float(fallback_step)
        intercept = float(values[0] - slope * indices[0])
    else:
        slope = float(fallback_step)
        intercept = float(fallback_offset)

    if (not np.isfinite(slope)
            or slope < 0.35 * fallback_step
            or slope > 1.80 * fallback_step):
        slope = float(fallback_step)
        if len(indices):
            intercept = float(np.median(values - slope * indices))
        else:
            intercept = float(fallback_offset)
    return intercept + slope * np.arange(n_positions), float(slope)


def _auto_detect_grid_spots(image_path, rows=4, cols=5):
    """Detect dye dots and fit an evenly spaced rows-by-cols grid.

    Use individual detections to estimate the grid origin and spacing in both directions.
    Interpolate missed points on the same grid to avoid displaced or overlapping positions.
    """
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    height, width = rgb.shape[:2]
    edge = max(2, int(round(min(width, height) * 0.015)))
    border = np.concatenate([
        rgb[:edge].reshape(-1, 3), rgb[-edge:].reshape(-1, 3),
        rgb[:, :edge].reshape(-1, 3), rgb[:, -edge:].reshape(-1, 3),
    ], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)

    records = []
    spot = 1
    for r in range(rows):
        y0, y1 = int(round(r * height / rows)), int(round((r + 1) * height / rows))
        for c in range(cols):
            x0, x1 = int(round(c * width / cols)), int(round((c + 1) * width / cols))
            cell = distance[y0:y1, x0:x1]
            cell_h, cell_w = cell.shape
            high = float(np.percentile(cell, 95))
            low = float(np.percentile(cell, 35))
            threshold = max(3.0, low + 0.18 * max(high - low, 0.0))
            mask = cell > threshold
            structure = np.ones((3, 3), dtype=bool)
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
            mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
            mask = ndimage.binary_fill_holes(mask)
            labels, n_labels = ndimage.label(mask)

            detected = False
            if n_labels:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                label_id = int(np.argmax(sizes))
                area = int(sizes[label_id])
                min_area = 0.025 * cell_h * cell_w
                max_area = 0.82 * cell_h * cell_w
                if min_area <= area <= max_area:
                    cy, cx = ndimage.center_of_mass(labels == label_id)
                    radius = np.sqrt(area / np.pi)
                    margin_x, margin_y = 0.08 * cell_w, 0.08 * cell_h
                    if (margin_x <= cx <= cell_w - margin_x
                            and margin_y <= cy <= cell_h - margin_y):
                        detected = True
                        records.append({
                            "spot": spot, "x": x0 + float(cx), "y": y0 + float(cy),
                            "radius": float(np.clip(
                                radius, 0.12 * min(cell_w, cell_h),
                                0.46 * min(cell_w, cell_h))),
                            "detected": True,
                            "source": "raw_foreground_component",
                        })
            if not detected:
                records.append({
                    "spot": spot, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                    "radius": 0.34 * min(cell_w, cell_h),
                    "detected": False,
                    "source": "raw_detection_failed",
                })
            spot += 1

    raw = pd.DataFrame(records)
    raw["grid_row"] = (raw["spot"] - 1) // cols
    raw["grid_col"] = (raw["spot"] - 1) % cols
    raw["raw_x"] = raw["x"]
    raw["raw_y"] = raw["y"]
    raw["raw_radius"] = raw["radius"]
    reliable = raw[raw["detected"]].copy()

    col_centers = reliable.groupby("grid_col", as_index=False)["raw_x"].median()
    row_centers = reliable.groupby("grid_row", as_index=False)["raw_y"].median()
    fallback_dx, fallback_dy = width / cols, height / rows
    grid_x, dx = _fit_regular_grid_axis(
        col_centers["grid_col"], col_centers["raw_x"], cols,
        fallback_dx, 0.5 * fallback_dx)
    grid_y, dy = _fit_regular_grid_axis(
        row_centers["grid_row"], row_centers["raw_y"], rows,
        fallback_dy, 0.5 * fallback_dy)

    if grid_x[0] < 0:
        grid_x -= grid_x[0]
    if grid_x[-1] > width - 1:
        grid_x -= grid_x[-1] - (width - 1)
    if grid_y[0] < 0:
        grid_y -= grid_y[0]
    if grid_y[-1] > height - 1:
        grid_y -= grid_y[-1] - (height - 1)

    detected_radii = reliable["raw_radius"].to_numpy(dtype=float)
    if len(detected_radii):
        radius = float(np.median(detected_radii))
    else:
        radius = 0.34 * min(dx, dy)
    radius = float(np.clip(radius, 0.12 * min(dx, dy), 0.42 * min(dx, dy)))

    raw["x"] = raw["grid_col"].map(dict(enumerate(grid_x)))
    raw["y"] = raw["grid_row"].map(dict(enumerate(grid_y)))
    raw["radius"] = radius
    raw["source"] = np.where(
        raw["detected"],
        "regularized_equal_grid_from_detection",
        "grid_interpolated_missing_detection")
    raw["grid_dx"] = dx
    raw["grid_dy"] = dy
    return raw[[
        "spot", "grid_row", "grid_col", "x", "y", "radius",
        "detected", "raw_x", "raw_y", "raw_radius",
        "grid_dx", "grid_dy", "source",
    ]]


def _load_spot_coordinates(coord_path, width, height, rows=4, cols=5,
                           image_path=None):
    """Load the spot coordinates or detect their centers. Fall back to a regular grid if needed.

    Treat x, y, and radius values at or below 1.5 as normalized coordinates and larger values as pixels.
    """
    if coord_path:
        if coord_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(coord_path)
        else:
            df = pd.read_csv(coord_path)
        required = {"spot", "x", "y"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"spot coordinate file must contain {sorted(required)}; got {list(df.columns)}")
        df = df.sort_values("spot").copy()
        if df["spot"].tolist() != list(range(1, 21)):
            raise ValueError("spot coordinate file must contain spots 1..20 exactly once")
        radius_missing = "radius" not in df
        if radius_missing:
            df["radius"] = 0.35 * min(width / max(cols, 1), height / max(rows, 1))
        normalized_xy = df[["x", "y"]].abs().to_numpy().max() <= 1.5
        if normalized_xy:
            df["x"] *= width
            df["y"] *= height
        if not radius_missing and df["radius"].abs().max() <= 1.5:
            df["radius"] *= min(width, height)
        df["source"] = "coordinate_file"
        return df[["spot", "x", "y", "radius", "source"]]

    if rows * cols != 20:
        raise ValueError(f"spot_rows * spot_cols must equal 20; got {rows} * {cols}")
    if image_path and os.path.isfile(image_path):
        try:
            return _auto_detect_grid_spots(image_path, rows=rows, cols=cols)
        except Exception as exc:
            print(f"  Could not locate dye dots automatically. Using a regular grid: {exc}")
    cell_w, cell_h = width / cols, height / rows
    radius = 0.34 * min(cell_w, cell_h)
    records = []
    spot = 1
    for r in range(rows):
        for c in range(cols):
            records.append({
                "spot": spot,
                "x": (c + 0.5) * cell_w,
                "y": (r + 0.5) * cell_h,
                "radius": radius,
                "source": f"fallback_grid_{rows}x{cols}_row_major",
            })
            spot += 1
    return pd.DataFrame(records)


def _gaussian_spot_map(width, height, spot_values, coords):
    yy, xx = np.mgrid[0:height, 0:width]
    heat = np.zeros((height, width), dtype=float)
    for row, value in zip(coords.itertuples(index=False), spot_values):
        sigma = max(float(row.radius) / 1.8, 1.0)
        heat += float(value) * np.exp(
            -((xx - float(row.x)) ** 2 + (yy - float(row.y)) ** 2)
            / (2.0 * sigma ** 2))
    return heat


def _save_attribution_overlay(image_path, spot_values, coords, out_path,
                              title, signed=False, alpha=0.74):
    image = Image.open(image_path).convert("RGB")
    base = np.asarray(image, dtype=np.float32) / 255.0
    height, width = base.shape[:2]
    heat = _gaussian_spot_map(width, height, spot_values, coords)
    if signed:
        vmax = max(float(np.max(np.abs(heat))), 1e-12)
        norm = mpl_colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        rgba = plt.get_cmap("coolwarm")(norm(heat))
        strength = np.clip(np.abs(heat) / vmax, 0, 1)[..., None]
    else:
        heat = np.maximum(heat, 0)
        vmax = max(float(np.max(heat)), 1e-12)
        norm = mpl_colors.PowerNorm(gamma=2.2, vmin=0.0, vmax=vmax)
        rgba = plt.get_cmap("jet")(norm(heat))
        strength = np.ones_like(heat)[..., None]
    blend_alpha = 0.58 if signed else alpha
    overlay = (base * (1 - blend_alpha * strength)
               + rgba[..., :3] * (blend_alpha * strength))
    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(overlay)
    for row in coords.itertuples(index=False):
        ax.text(row.x, row.y, str(int(row.spot)), ha="center", va="center",
                fontsize=7, color="white",
                bbox={"boxstyle": "circle,pad=0.15", "fc": "black", "alpha": 0.55,
                      "ec": "none"})
    ax.set_title(title)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm" if signed else "jet")
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02,
                 label="Signed SHAP contribution to Dead"
                 if signed else "Absolute SHAP importance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return heat


def _save_shap_overlay_outputs(shap_matrix, feat_names, X_val, y_val, probs_final,
                               stems_val, batches_val, image_root, coord_path,
                               figures_dir, results_dir, rows, cols, max_samples,
                               final_threshold=0.5, final_model_label="Final model"):
    """Save global and per-sample SHAP dye-dot overlays with their underlying values."""
    image_paths = [
        _find_original_image(image_root, batch, stem)
        for batch, stem in zip(batches_val, stems_val)
    ]
    valid = [i for i, path in enumerate(image_paths) if path and os.path.isfile(path)]
    if not valid:
        print(f"  Skipping SHAP overlays: no matching PNG or JPG files in IMAGE_ROOT={image_root!r}")
        return

    denom = np.sum(np.abs(shap_matrix), axis=1, keepdims=True)
    shap_norm = shap_matrix / np.maximum(denom, 1e-12)
    signed_spot = _aggregate_feature_values_to_spots(feat_names, shap_norm)
    absolute_spot = _aggregate_feature_values_to_spots(
        feat_names, np.abs(shap_norm))
    global_abs = absolute_spot.mean(axis=0)
    global_pct = global_abs / max(global_abs.sum(), 1e-12) * 100.0

    ref_i = valid[0]
    ref_img = Image.open(image_paths[ref_i]).convert("RGB")
    width, height = ref_img.size
    coords = _load_spot_coordinates(
        coord_path, width, height, rows=rows, cols=cols,
        image_path=image_paths[ref_i])
    source_counts = coords["source"].value_counts().to_dict()
    print(f"  Dye-dot coordinate sources: {source_counts}")
    if any(str(s).startswith("fallback") for s in coords["source"]):
        print("  Some points could not be located automatically and use grid centers. "
              "Review spot_coordinates_used_for_overlays.xlsx before using the figures in publications.")
    overlay_dir = os.path.join(figures_dir, "shap_attribution_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    for old_path in glob.glob(os.path.join(
            overlay_dir, "*_signed_shap_overlay.png")):
        os.remove(old_path)
    coords.to_excel(os.path.join(
        results_dir, "spot_coordinates_used_for_overlays.xlsx"), index=False)

    _save_attribution_overlay(
        image_paths[ref_i], global_pct, coords,
        os.path.join(overlay_dir, "global_absolute_shap_overlay.png"),
        "Global absolute SHAP dye-spot attribution", signed=False)

    final_predictions = (np.asarray(probs_final) >= final_threshold).astype(int)
    table = {
        "stem": stems_val, "batch": batches_val,
        "true_label": [CLASS_NAMES[int(v)] for v in y_val],
        "final_model": np.repeat(final_model_label, len(y_val)),
        "dead_probability": probs_final,
        "final_threshold": np.repeat(final_threshold, len(y_val)),
        "final_predicted_label": [CLASS_NAMES[int(v)] for v in final_predictions],
        "final_correct": final_predictions == y_val,
        "image_path": image_paths,
    }
    for s in range(20):
        table[f"spot{s+1:02d}_signed_shap"] = signed_spot[:, s]
        table[f"spot{s+1:02d}_absolute_shap"] = absolute_spot[:, s]
    shap_sample_df = pd.DataFrame(table)
    _write_highlighted_prediction_excel(
        shap_sample_df,
        os.path.join(results_dir, "sample_level_shap_spot_values.xlsx"),
        predicted_cols=["final_predicted_label"])
    global_spot_df = pd.DataFrame({
        "spot": np.arange(1, 21),
        "mean_absolute_shap": global_abs,
        "normalized_contribution_percent": global_pct,
    })
    global_spot_df.to_excel(os.path.join(
        results_dir, "global_shap_spot_contribution.xlsx"), index=False)

    selected = valid[:max_samples] if max_samples > 0 else valid
    index_rows = []
    for i in selected:
        current = Image.open(image_paths[i]).convert("RGB")
        cw, ch = current.size
        current_coords = _load_spot_coordinates(
            coord_path, cw, ch, rows=rows, cols=cols,
            image_path=image_paths[i])
        safe_stem = _safe_name(stems_val[i])
        pred_label = CLASS_NAMES[int(probs_final[i] >= final_threshold)]
        absolute_name = f"{safe_stem}_absolute_shap_overlay.png"
        _save_attribution_overlay(
            image_paths[i], absolute_spot[i], current_coords,
            os.path.join(overlay_dir, absolute_name),
            f"{stems_val[i]} | absolute SHAP attribution",
            signed=False)
        index_rows.append({
            "stem": stems_val[i], "batch": batches_val[i],
            "true_label": CLASS_NAMES[int(y_val[i])],
            "final_model": final_model_label,
            "final_threshold": float(final_threshold),
            "final_predicted_label": pred_label,
            "final_correct": bool(final_predictions[i] == y_val[i]),
            "dead_probability": float(probs_final[i]),
            "source_image": image_paths[i],
            "absolute_overlay": absolute_name,
        })
    _write_highlighted_prediction_excel(
        pd.DataFrame(index_rows),
        os.path.join(results_dir, "shap_overlay_file_index.xlsx"),
        predicted_cols=["final_predicted_label"])
    print(f"  SHAP image overlays saved to: {overlay_dir}")


def _grouped_permutation_importance(run, X_val, y_val, feat_names, repeats, seed):
    """Permute final-model features by dye dot and measure the drop in ROC AUC."""
    if len(np.unique(y_val)) < 2 or repeats <= 0:
        return pd.DataFrame(columns=["spot", "auc_drop_mean", "auc_drop_std"])
    final_tag = run["final_tag"]
    base_prob = _predict_base_and_combined(run["models"], run["stack_meta"], X_val)[final_tag]
    baseline = roc_auc_score(y_val, base_prob)
    rng = np.random.default_rng(seed)
    groups = {s: [] for s in range(1, 21)}
    for col, name in enumerate(feat_names):
        for spot, _ in _feature_spots(name):
            groups[spot].append(col)
    rows = []
    for spot in range(1, 21):
        drops = []
        cols = sorted(set(groups[spot]))
        for _ in range(repeats):
            xp = X_val.copy()
            order = rng.permutation(len(xp))
            xp[:, cols] = xp[order][:, cols]
            pp = _predict_base_and_combined(
                run["models"], run["stack_meta"], xp)[final_tag]
            drops.append(baseline - roc_auc_score(y_val, pp))
        rows.append({
            "spot": spot, "n_grouped_features": len(cols),
            "auc_drop_mean": float(np.mean(drops)),
            "auc_drop_std": float(np.std(drops)),
        })
    return pd.DataFrame(rows)


def random_split_like_client(y, val_rate=0.3, seed=999):
    random.seed(seed)
    train_idx, val_idx = [], []
    for cls in sorted(set(y.tolist())):
        cls_pos = np.where(y == cls)[0].tolist()
        cls_pos.sort()
        n_val = int(len(cls_pos) * val_rate)
        val_pick = set(random.sample(cls_pos, k=n_val))
        for i in cls_pos:
            (val_idx if i in val_pick else train_idx).append(i)
    return np.array(train_idx), np.array(val_idx)


def cross_batch_split(batch_ids, test_batch):
    tr = np.where(batch_ids != test_batch)[0]
    va = np.where(batch_ids == test_batch)[0]
    return tr, va


def fit_xgb(X_tr, y_tr, X_va, y_va, params, seed=0):
    pos_weight = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=pos_weight,
        random_state=seed,
        early_stopping_rounds=50,
        **params,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return clf


def fit_lgb(X_tr, y_tr, X_va, y_va, seed=0):
    clf = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        n_estimators=1500, learning_rate=0.02,
        num_leaves=15, max_depth=-1,
        min_child_samples=5,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1,
        is_unbalance=True,
        random_state=seed, verbose=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


def fit_rf(X_tr, y_tr, seed=0):
    clf = RandomForestClassifier(
        n_estimators=800, max_depth=None,
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def fit_et(X_tr, y_tr, seed=0):
    clf = ExtraTreesClassifier(
        n_estimators=800, max_depth=None,
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def best_threshold(y_true, prob, metric="acc"):
    """Find the threshold between 0.05 and 0.95 that maximizes the selected metric."""
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if metric == "acc":
            v = accuracy_score(y_true, pred)
        elif metric == "f1_dead":
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            v = 2 * prec * rec / max(prec + rec, 1e-9)
        else:
            raise ValueError(metric)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def optuna_search_xgb(X_tr, y_tr, X_va, y_va, n_trials=40):
    """Search with Optuna when installed and use default parameters otherwise."""
    try:
        import optuna
    except ImportError:
        print("Optuna is unavailable. Using the default hyperparameters.")
        return {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.03,
                "min_child_weight": 3, "gamma": 0.0,
                "subsample": 0.85, "colsample_bytree": 0.85,
                "reg_alpha": 0.1, "reg_lambda": 1.0}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        clf = fit_xgb(X_tr, y_tr, X_va, y_va, params)
        prob = clf.predict_proba(X_va)[:, 1]
        try:
            return roc_auc_score(y_va, prob)
        except ValueError:
            return 0.0

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
    pbar = tqdm(total=n_trials, desc="optuna", leave=False)
    def _cb(study_, trial_):
        pbar.set_postfix(best=f"{study_.best_value:.4f}")
        pbar.update(1)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=int(os.environ.get("OPTUNA_NJOBS", "1")),
                   show_progress_bar=False, callbacks=[_cb])
    pbar.close()
    print(f"  optuna best AUC = {study.best_value:.4f}, params = {study.best_params}")
    return study.best_params


def _optional_shap_importance(
        run, X_val, y_val, feat_names, stems_val, batches_val,
        image_root, coord_path, results_dir, figures_dir,
        spot_rows=4, spot_cols=5, overlay_max_samples=40):
    """Compute SHAP values for four tree-model families and export spot contributions and image overlays."""
    try:
        import shap
    except ImportError:
        print("  SHAP is unavailable. To enable SHAP plots, run python -m pip install shap.")
        return

    shap_cols = {}
    shap_matrices = {}
    for tag in ("xgb", "lgb", "rf", "et"):
        family = []
        try:
            for clf in run["models"][tag]:
                values = shap.TreeExplainer(clf).shap_values(X_val)
                if isinstance(values, list):
                    values = values[-1]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[:, :, -1]
                if values.shape == X_val.shape:
                    family.append(values)
            if family:
                shap_matrices[tag] = np.mean(family, axis=0)
                shap_cols[tag] = np.mean(
                    np.abs(shap_matrices[tag]), axis=0)
        except Exception as exc:
            print(f"  Skipping SHAP for {MODEL_LABELS[tag]}: {exc}")

    if not shap_cols:
        return
    family_tags = list(shap_matrices)
    normalized_matrices = []
    for tag in family_tags:
        values = shap_matrices[tag]
        normalized_matrices.append(
            values / np.maximum(
                np.sum(np.abs(values), axis=1, keepdims=True), 1e-12))
    combined_matrix = np.mean(normalized_matrices, axis=0)
    shap_matrices["combined_mean"] = combined_matrix
    shap_cols["combined_mean"] = np.mean(np.abs(combined_matrix), axis=0)
    shap_df = pd.DataFrame({"feature": feat_names, **shap_cols})

    shap_spot_sheets = {}
    for tag, imp in shap_cols.items():
        label = MODEL_LABELS.get(tag, "Combined SHAP")
        order = np.argsort(imp)[-30:]
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(np.asarray(feat_names)[order], imp[order], color="#7A5195")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"Top 30 SHAP features — {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"shap_top30_{tag}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        spot_df, _, global_imp = _spot_tables(feat_names, imp)
        spot_df["global_statistics_importance_excluded"] = global_imp
        shap_spot_sheets[tag] = spot_df
        _save_spot_heatmap(
            spot_df["normalized_contribution_percent"].to_numpy(),
            f"Dye-spot SHAP contribution — {label}",
            os.path.join(figures_dir, f"shap_spot_heatmap_{tag}.png"))

    shap_df.to_excel(os.path.join(
        results_dir, "shap_feature_importance.xlsx"), index=False)
    with pd.ExcelWriter(os.path.join(
            results_dir, "shap_spot_importance_all_models.xlsx")) as writer:
        for tag, df in shap_spot_sheets.items():
            df.to_excel(writer, sheet_name=tag, index=False)

    final_prob = run["probabilities"][run["final_tag"]]
    _save_shap_overlay_outputs(
        combined_matrix, feat_names, X_val, y_val, final_prob,
        np.asarray(stems_val), np.asarray(batches_val),
        image_root=image_root, coord_path=coord_path,
        figures_dir=figures_dir, results_dir=results_dir,
        rows=spot_rows, cols=spot_cols,
        max_samples=overlay_max_samples,
        final_threshold=run["best_thresh_final"],
        final_model_label=MODEL_LABELS[run["final_tag"]])


def generate_split_report(run, X, y, batch_ids, stems, feat_names, va_idx,
                          out_dir, split_key, perm_repeats=10, do_shap=True,
                          seed=999, image_root=None, spot_coord_file=None,
                          spot_rows=4, spot_cols=5,
                          shap_overlay_max_samples=40):
    """Save validation figures and data under figures/<split> and results/<split>."""
    split_key = _safe_name(split_key)
    figures_dir = os.path.join(out_dir, "figures", split_key)
    results_dir = os.path.join(out_dir, "results", split_key)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    for deprecated_pattern in (
            "feature_importance_top30_*.png", "spot_heatmap_*.png",
            "feature_importance_all_models.xlsx",
            "spot_importance_all_models.xlsx",
            "spot_feature_space_importance.xlsx",
            "shap_spot_importance_*.xlsx",
            "boosting_iteration_curves.xlsx",
            "stacking_model_weights.xlsx",
            "model_evaluation.xlsx", "model_curve_data.xlsx",
            "shap_importance_data.xlsx", "shap_overlay_data.xlsx"):
        for old_path in glob.glob(os.path.join(figures_dir, deprecated_pattern)):
            os.remove(old_path)
        for old_path in glob.glob(os.path.join(results_dir, deprecated_pattern)):
            os.remove(old_path)

    y_val = y[va_idx]
    X_val = X[va_idx]
    probs = run["probabilities"]
    tags = [t for t in ("xgb", "lgb", "rf", "et", "ens", "stk") if t in probs]

    metric_rows, pred_table = [], {
        "stem": stems[va_idx],
        "batch": batch_ids[va_idx],
        "true_label_id": y_val,
        "true_label": [CLASS_NAMES[int(v)] for v in y_val],
    }
    confusion_sheets = {}
    for tag in tags:
        threshold = run["results"][tag]["best_thresh"]
        pred = (probs[tag] >= threshold).astype(int)
        row = {"model": MODEL_LABELS[tag], "tag": tag}
        row.update(_metric_row(y_val, probs[tag], threshold))
        metric_rows.append(row)
        pred_table[f"{tag}_dead_probability"] = probs[tag]
        pred_table[f"{tag}_threshold"] = np.repeat(threshold, len(y_val))
        pred_table[f"{tag}_predicted_label"] = [CLASS_NAMES[int(v)] for v in pred]
        cm = confusion_matrix(y_val, pred, labels=[0, 1])
        confusion_sheets[tag] = pd.DataFrame(
            cm, index=["True Alive", "True Dead"],
            columns=["Pred Alive", "Pred Dead"])
        _save_confusion_plot(
            cm, f"{MODEL_LABELS[tag]} (threshold={threshold:.3f})",
            os.path.join(figures_dir, f"confusion_matrix_{tag}.png"))

    final_tag = run["final_tag"]
    final_threshold = run["best_thresh_final"]
    final_pred = (probs[final_tag] >= final_threshold).astype(int)
    pred_table["final_model"] = np.repeat(MODEL_LABELS[final_tag], len(y_val))
    pred_table["final_dead_probability"] = probs[final_tag]
    pred_table["final_threshold"] = np.repeat(final_threshold, len(y_val))
    pred_table["final_predicted_label"] = [
        CLASS_NAMES[int(v)] for v in final_pred]
    pred_table["final_correct"] = final_pred == y_val

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(pred_table)
    metrics_df.to_excel(os.path.join(results_dir, "metrics_all_models.xlsx"),
                        index=False)
    with pd.ExcelWriter(os.path.join(
            results_dir, "confusion_matrices.xlsx")) as writer:
        for tag, df in confusion_sheets.items():
            df.to_excel(writer, sheet_name=tag)
    prediction_columns = [
        f"{tag}_predicted_label" for tag in tags
    ] + ["final_predicted_label"]
    _write_highlighted_prediction_excel(
        predictions_df,
        os.path.join(results_dir, "predictions_all_models.xlsx"),
        predicted_cols=prediction_columns)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(
            axes, ("roc_auc", "average_precision", "dead_recall_sensitivity"),
            ("ROC-AUC", "Average precision", "Dead recall")):
        ax.bar(metrics_df["model"], metrics_df[col], color="#4C78A8")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(metrics_df[col]):
            if np.isfinite(v):
                ax.text(i, min(v + 0.025, 1.02), f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "model_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    ncols = 3
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        ax.hist(probs[tag][y_val == 0], bins=12, alpha=0.65,
                label="Alive", color="#4C78A8")
        ax.hist(probs[tag][y_val == 1], bins=12, alpha=0.65,
                label="Dead", color="#E45756")
        ax.axvline(run["results"][tag]["best_thresh"], color="black",
                   linestyle="--", linewidth=1.2, label="threshold")
        ax.set(xlabel="Predicted Dead probability", ylabel="Count",
               title=MODEL_LABELS[tag], xlim=(0, 1))
        ax.legend(fontsize=8)
    for ax in axes.flat[len(tags):]:
        ax.axis("off")
    fig.suptitle("Validation probability distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "probability_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    boosting_curves = {}
    try:
        xgb_eval = run["models"]["xgb"][0].evals_result()
        for dataset_name, metrics in xgb_eval.items():
            for metric_name, values in metrics.items():
                boosting_curves[f"XGBoost_{dataset_name}_{metric_name}"] = values
    except Exception:
        pass
    try:
        lgb_eval = run["models"]["lgb"][0].evals_result_
        for dataset_name, metrics in lgb_eval.items():
            for metric_name, values in metrics.items():
                boosting_curves[f"LightGBM_{dataset_name}_{metric_name}"] = values
    except Exception:
        pass
    curve_df = None
    if boosting_curves:
        max_len = max(len(v) for v in boosting_curves.values())
        curve_df = pd.DataFrame({
            name: pd.Series(values) for name, values in boosting_curves.items()})
        curve_df.insert(0, "iteration", np.arange(1, max_len + 1))
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, values in boosting_curves.items():
            ax.plot(np.arange(1, len(values) + 1), values, label=name)
        ax.set_xlabel("Boosting iteration / number of trees")
        ax.set_ylabel("Evaluation metric")
        ax.set_title("XGBoost and LightGBM validation curves")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "boosting_iteration_curves.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    roc_data, pr_data = {}, {}
    fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5))
    fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5.5))
    if len(np.unique(y_val)) == 2:
        for tag in tags:
            fpr, tpr, roc_thr = roc_curve(y_val, probs[tag])
            precision, recall, pr_thr = precision_recall_curve(y_val, probs[tag])
            auc_v = roc_auc_score(y_val, probs[tag])
            ap_v = average_precision_score(y_val, probs[tag])
            ax_roc.plot(fpr, tpr, lw=2, label=f"{MODEL_LABELS[tag]} ({auc_v:.3f})")
            ax_pr.step(recall, precision, where="post", lw=2,
                       label=f"{MODEL_LABELS[tag]} ({ap_v:.3f})")
            roc_data[tag] = pd.DataFrame(
                {"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
            pr_data[tag] = pd.DataFrame({
                "precision": precision, "recall": recall,
                "threshold": np.r_[pr_thr, np.nan]})
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC curves — all Stage1 models", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision–recall curves — all Stage1 models",
              xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(fontsize=8); ax_pr.legend(fontsize=8)
    fig_roc.tight_layout(); fig_pr.tight_layout()
    fig_roc.savefig(os.path.join(figures_dir, "roc_all_models.png"),
                    dpi=300, bbox_inches="tight")
    fig_pr.savefig(os.path.join(figures_dir, "pr_all_models.png"),
                   dpi=300, bbox_inches="tight")
    plt.close(fig_roc); plt.close(fig_pr)

    calibration_data = {}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if len(np.unique(y_val)) == 2:
        for tag in tags:
            frac, mean_pred = calibration_curve(
                y_val, probs[tag], n_bins=min(10, max(3, len(y_val) // 10)),
                strategy="quantile")
            calibration_data[tag] = pd.DataFrame(
                {"mean_predicted_probability": mean_pred,
                 "observed_dead_fraction": frac})
            ax.plot(mean_pred, frac, marker="o", label=MODEL_LABELS[tag])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Mean predicted Dead probability",
           ylabel="Observed Dead fraction", title="Calibration curves",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "calibration_all_models.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    if curve_df is not None:
        curve_df.to_excel(os.path.join(
            results_dir, "boosting_iteration_curves.xlsx"), index=False)
    with pd.ExcelWriter(os.path.join(
            results_dir, "roc_curve_data.xlsx")) as writer:
        if roc_data:
            for tag, df in roc_data.items():
                df.to_excel(writer, sheet_name=tag, index=False)
        else:
            pd.DataFrame({
                "note": ["ROC unavailable: validation set has one class"]
            }).to_excel(writer, sheet_name="unavailable", index=False)
    with pd.ExcelWriter(os.path.join(
            results_dir, "pr_curve_data.xlsx")) as writer:
        if pr_data:
            for tag, df in pr_data.items():
                df.to_excel(writer, sheet_name=tag, index=False)
        else:
            pd.DataFrame({
                "note": ["PR unavailable: validation set has one class"]
            }).to_excel(writer, sheet_name="unavailable", index=False)
    with pd.ExcelWriter(os.path.join(
            results_dir, "calibration_data.xlsx")) as writer:
        if calibration_data:
            for tag, df in calibration_data.items():
                df.to_excel(writer, sheet_name=tag, index=False)
        else:
            pd.DataFrame({
                "note": ["Calibration unavailable: validation set has one class"]
            }).to_excel(writer, sheet_name="unavailable", index=False)

    perm_df = _grouped_permutation_importance(
        run, X_val, y_val, feat_names, perm_repeats, seed)
    perm_df.to_excel(os.path.join(
        results_dir, "grouped_permutation_spot_importance.xlsx"), index=False)
    if not perm_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(perm_df["spot"], perm_df["auc_drop_mean"],
               yerr=perm_df["auc_drop_std"], capsize=3, color="#54A24B")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(1, 21))
        ax.set_xlabel("Dye spot")
        ax.set_ylabel("ROC-AUC decrease after grouped permutation")
        ax.set_title(f"Final {MODEL_LABELS[run['final_tag']]}: dye-spot permutation importance")
        fig.tight_layout()
        fig.savefig(os.path.join(
            figures_dir, "grouped_permutation_spot_importance.png"),
            dpi=300, bbox_inches="tight")
        plt.close(fig)

    if run["stack_meta"] is not None:
        coef = run["stack_meta"].coef_[0]
        pd.DataFrame({
            "base_model": [MODEL_LABELS[t] for t in ("xgb", "lgb", "rf", "et")],
            "stacking_coefficient": coef,
        }).to_excel(os.path.join(
            results_dir, "stacking_model_weights.xlsx"), index=False)

    if do_shap:
        _optional_shap_importance(
            run, X_val, y_val, feat_names,
            stems[va_idx], batch_ids[va_idx],
            image_root, spot_coord_file,
            results_dir, figures_dir,
            spot_rows=spot_rows, spot_cols=spot_cols,
            overlay_max_samples=shap_overlay_max_samples)

    summary = {
        "split": split_key,
        "validation_samples": int(len(va_idx)),
        "alive": int(np.sum(y_val == 0)),
        "dead": int(np.sum(y_val == 1)),
        "final_model": MODEL_LABELS[run["final_tag"]],
        "final_threshold": float(run["best_thresh_final"]),
        "figures_dir": figures_dir,
        "results_dir": results_dir,
        "image_root_for_overlays": image_root,
        "spot_coordinate_file": spot_coord_file,
        "spot_grid_fallback": f"{spot_rows}x{spot_cols} row-major",
        "spot_attribution_rule":
            "single-spot features -> that spot; adjacent differences -> 50% to each endpoint; global statistics kept separate",
    }
    with open(os.path.join(results_dir, "report_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Stage 1 figures saved to: {figures_dir}")
    print(f"  Stage 1 data saved to: {results_dir}")


def evaluate_one_split(X, y, tr_idx, va_idx, feat_names, name,
                       n_trials, prefer_metric, n_seeds=3, use_stack=True):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    print(f"  train n={len(tr_idx)} (Alive={int(np.sum(y_tr==0))} Dead={int(np.sum(y_tr==1))})  "
          f"val n={len(va_idx)} (Alive={int(np.sum(y_va==0))} Dead={int(np.sum(y_va==1))})")

    seeds = list(range(n_seeds))
    print(f"  Searching XGBoost hyperparameters (n_trials={n_trials})...")
    best_params = optuna_search_xgb(X_tr, y_tr, X_va, y_va, n_trials=n_trials)

    xgb_clfs = [fit_xgb(X_tr, y_tr, X_va, y_va, best_params, seed=s)
                for s in tqdm(seeds, desc=f"  XGB×{n_seeds}", leave=False)]
    p_xgb = np.mean([c.predict_proba(X_va)[:, 1] for c in xgb_clfs], axis=0)

    lgb_clfs = [fit_lgb(X_tr, y_tr, X_va, y_va, seed=s)
                for s in tqdm(seeds, desc=f"  LGB×{n_seeds}", leave=False)]
    p_lgb = np.mean([c.predict_proba(X_va)[:, 1] for c in lgb_clfs], axis=0)

    rf_clfs = [fit_rf(X_tr, y_tr, seed=s)
               for s in tqdm(seeds, desc=f"  RF×{n_seeds}", leave=False)]
    p_rf = np.mean([c.predict_proba(X_va)[:, 1] for c in rf_clfs], axis=0)

    et_clfs = [fit_et(X_tr, y_tr, seed=s)
               for s in tqdm(seeds, desc=f"  ET×{n_seeds}", leave=False)]
    p_et = np.mean([c.predict_proba(X_va)[:, 1] for c in et_clfs], axis=0)

    p_ens = (
        ENSEMBLE_WEIGHTS["xgb"] * p_xgb
        + ENSEMBLE_WEIGHTS["lgb"] * p_lgb
        + ENSEMBLE_WEIGHTS["rf"] * p_rf
        + ENSEMBLE_WEIGHTS["et"] * p_et
    )

    p_stack = None
    stack_meta = None
    if use_stack and len(tr_idx) >= 30:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros((len(y_tr), 4))
        folds = list(skf.split(X_tr, y_tr))
        for fi, (i_in, i_out) in enumerate(tqdm(folds, desc="  stack OOF", leave=False)):
            Xa, ya = X_tr[i_in], y_tr[i_in]
            Xb, yb = X_tr[i_out], y_tr[i_out]
            try:
                cx = fit_xgb(Xa, ya, Xb, yb, best_params, seed=0)
                cl = fit_lgb(Xa, ya, Xb, yb, seed=0)
                cr = fit_rf(Xa, ya, seed=0)
                ce = fit_et(Xa, ya, seed=0)
                oof[i_out, 0] = cx.predict_proba(Xb)[:, 1]
                oof[i_out, 1] = cl.predict_proba(Xb)[:, 1]
                oof[i_out, 2] = cr.predict_proba(Xb)[:, 1]
                oof[i_out, 3] = ce.predict_proba(Xb)[:, 1]
            except Exception as e:
                print(f"    Fold {fi} failed: {e}")
        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        meta.fit(oof, y_tr)
        val_stack_in = np.column_stack([p_xgb, p_lgb, p_rf, p_et])
        p_stack = meta.predict_proba(val_stack_in)[:, 1]
        stack_meta = meta
        print(f"  Stacking meta-model coefficients: xgb={meta.coef_[0,0]:.2f} "
              f"lgb={meta.coef_[0,1]:.2f} rf={meta.coef_[0,2]:.2f} et={meta.coef_[0,3]:.2f}")

    results = {}
    rows = [("xgb", p_xgb), ("lgb", p_lgb), ("rf", p_rf), ("et", p_et), ("ens", p_ens)]
    if p_stack is not None:
        rows.append(("stk", p_stack))
    for tag, p in rows:
        try:
            auc = roc_auc_score(y_va, p)
        except ValueError:
            auc = float("nan")
        t, _ = best_threshold(y_va, p, metric=prefer_metric)
        pred = (p >= t).astype(int)
        acc = accuracy_score(y_va, pred)
        results[tag] = {"auc": float(auc), "best_thresh": float(t),
                        "acc_at_best": float(acc),
                        "acc_at_0.5": float(accuracy_score(y_va, (p >= 0.5).astype(int)))}
    print(f"  {'model':<6s} {'AUC':>7s} {'acc@0.5':>9s} {'best_t':>8s} {'acc@best':>9s}")
    tags_show = ["xgb", "lgb", "rf", "et", "ens"] + (["stk"] if p_stack is not None else [])
    for tag in tags_show:
        r = results[tag]
        print(f"  {tag:<6s} {r['auc']:>7.4f} {r['acc_at_0.5']*100:>8.2f}% "
              f"{r['best_thresh']:>8.3f} {r['acc_at_best']*100:>8.2f}%")

    final_tag = "stk" if (p_stack is not None and
                          results["stk"]["acc_at_best"] >= results["ens"]["acc_at_best"]) else "ens"
    p_final = p_stack if final_tag == "stk" else p_ens
    final_t = results[final_tag]["best_thresh"]
    pred_final = (p_final >= final_t).astype(int)
    cm = confusion_matrix(y_va, pred_final, labels=[0, 1])
    print(f"\n  Final confusion matrix ({final_tag} @ threshold={final_t:.3f}):")
    print(f"             Alive  Dead")
    print(f"    Alive    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"    Dead     {cm[1,0]:5d}  {cm[1,1]:5d}")

    bundle_models = {
        "xgb": xgb_clfs, "lgb": lgb_clfs, "rf": rf_clfs, "et": et_clfs,
    }
    return {
        "results": results,
        "models": bundle_models,
        "stack_meta": stack_meta,
        "probabilities": {tag: p for tag, p in rows},
        "val_idx": np.asarray(va_idx),
        "final_tag": final_tag,
        "best_xgb_params": best_params,
        "best_thresh_final": final_t,
        "val_acc_final": results[final_tag]["acc_at_best"],
        "confusion_matrix_final": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Root directory containing Batch-X/{Female,Male,Dead}/*.xlsx")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--mode", choices=["random", "cross_batch_all", "both"], default="both",
                    help="random: one random split. cross_batch_all: hold out each batch in turn. both: run both modes.")
    ap.add_argument("--val_rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--n_trials", type=int, default=40,
                    help="Number of hyperparameter trials when Optuna is installed")
    ap.add_argument("--prefer_metric", choices=["acc", "f1_dead"], default="acc",
                    help="Threshold selection metric. Use acc for accuracy or f1_dead to balance precision and recall for dead eggs.")
    ap.add_argument("--final_split", choices=["random", "Batch-3", "Batch-4", "Batch-5"],
                    default="random",
                    help="Training split to use for the exported model")
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="Number of seeds to average for each base model")
    ap.add_argument("--no_stack", action="store_true",
                    help="Disable the stacking meta-model and use a weighted ensemble "
                         "(XGB=0.40, LGB/RF/ET=0.20 each)")
    ap.add_argument("--no_shap", action="store_true",
                    help="Skip SHAP explanations. Also skipped automatically if SHAP is unavailable.")
    ap.add_argument("--perm_repeats", type=int, default=10,
                    help="Number of repeats for grouped dye-dot permutation importance")
    ap.add_argument("--image_root", default="",
                    help="Color-difference image root containing Batch-X/{Female,Male,Dead}/*.png. "
                         "If omitted, look for an image directory alongside data_root.")
    ap.add_argument("--spot_coord_file", default="",
                    help="Optional CSV or XLSX file with spot, x, y, and radius columns. "
                         "If omitted, use the grid defined by spot_rows and spot_cols.")
    ap.add_argument("--spot_rows", type=int, default=4,
                    help="Grid rows when the coordinate file is omitted (default: 4)")
    ap.add_argument("--spot_cols", type=int, default=5,
                    help="Grid columns when the coordinate file is omitted (default: 5). The grid must contain 20 positions.")
    ap.add_argument("--shap_overlay_max_samples", type=int, default=40,
                    help="Maximum validation samples for SHAP overlays. Use 0 or less for all samples.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.spot_rows * args.spot_cols != 20 and not args.spot_coord_file:
        raise ValueError("Without --spot_coord_file, --spot_rows * --spot_cols must equal 20")
    image_root = args.image_root.strip()
    if not image_root:
        image_root = os.path.join(os.path.dirname(os.path.abspath(args.data_root)), "image")
    if not os.path.isdir(image_root):
        print(f"Color-difference image directory not found. Only SHAP plots without overlays will be generated: {image_root}")
        image_root = None
    spot_coord_file = args.spot_coord_file.strip() or None
    if not spot_coord_file:
        data_parent = os.path.dirname(os.path.abspath(args.data_root))
        project_parent = os.path.dirname(data_parent)
        candidates = [
            os.path.join(data_parent, "spot_coordinates.xlsx"),
            os.path.join(data_parent, "spot_coordinates.csv"),
            os.path.join(project_parent, "spot_coordinates.xlsx"),
            os.path.join(project_parent, "spot_coordinates.csv"),
        ]
        spot_coord_file = next((p for p in candidates if os.path.isfile(p)), None)
        if spot_coord_file:
            print(f"Found a dye-dot coordinate file: {spot_coord_file}")
    if spot_coord_file and not os.path.isfile(spot_coord_file):
        raise FileNotFoundError(f"spot coordinate file not found: {spot_coord_file}")

    X, y, batch_ids, stems, feat_names = load_excel_features(
        args.data_root, task="da", version=args.feature_version)
    print(f"Feature version: {args.feature_version}  Dimensions: {X.shape[1]}")
    print(f"Total samples: {len(X)}  Alive={int(np.sum(y==0))}  Dead={int(np.sum(y==1))}")
    for b in sorted(set(batch_ids.tolist())):
        m = batch_ids == b
        print(f"  {b}: Alive={int(np.sum((y==0)&m))}  Dead={int(np.sum((y==1)&m))}")

    all_runs = {}

    if args.mode in ("random", "both"):
        tr_idx, va_idx = random_split_like_client(y, args.val_rate, args.seed)
        all_runs["random"] = evaluate_one_split(
            X, y, tr_idx, va_idx, feat_names,
            f"Split: random val_rate={args.val_rate} seed={args.seed}",
            args.n_trials, args.prefer_metric,
            n_seeds=args.n_seeds, use_stack=not args.no_stack)
        generate_split_report(
            all_runs["random"], X, y, batch_ids, stems, feat_names, va_idx,
            args.out_dir, "random", perm_repeats=args.perm_repeats,
            do_shap=not args.no_shap, seed=args.seed,
            image_root=image_root, spot_coord_file=spot_coord_file,
            spot_rows=args.spot_rows, spot_cols=args.spot_cols,
            shap_overlay_max_samples=args.shap_overlay_max_samples)

    if args.mode in ("cross_batch_all", "both"):
        for b in sorted(set(batch_ids.tolist())):
            tr_idx, va_idx = cross_batch_split(batch_ids, b)
            all_runs[f"cb_{b}"] = evaluate_one_split(
                X, y, tr_idx, va_idx, feat_names,
                f"Split: cross_batch  val={b}",
                args.n_trials, args.prefer_metric,
            n_seeds=args.n_seeds, use_stack=not args.no_stack)
            generate_split_report(
                all_runs[f"cb_{b}"], X, y, batch_ids, stems, feat_names, va_idx,
                args.out_dir, f"cross_batch_{b}",
                perm_repeats=args.perm_repeats,
                do_shap=not args.no_shap, seed=args.seed,
                image_root=image_root, spot_coord_file=spot_coord_file,
                spot_rows=args.spot_rows, spot_cols=args.spot_cols,
                shap_overlay_max_samples=args.shap_overlay_max_samples)

    print(f"\n{'='*70}\nSummary (final_tag @ best_thresh)\n{'='*70}")
    for name, run in all_runs.items():
        r = run["results"][run["final_tag"]]
        print(f"  {name:<14s}  [{run['final_tag']}] AUC={r['auc']:.4f}  "
              f"acc@best={r['acc_at_best']*100:.2f}%  thresh={r['best_thresh']:.3f}")
    cb_runs = [v for k, v in all_runs.items() if k.startswith("cb_")]
    if cb_runs:
        avg_acc = np.mean([r["results"][r["final_tag"]]["acc_at_best"] for r in cb_runs])
        avg_auc = np.mean([r["results"][r["final_tag"]]["auc"] for r in cb_runs])
        print(f"\n  Cross-batch average: AUC={avg_auc:.4f}  acc={avg_acc*100:.2f}% (estimated performance on unseen batches)")

    if args.final_split == "random":
        run = all_runs["random"]
        source = "random split"
    else:
        run = all_runs[f"cb_{args.final_split}"]
        source = f"cross_batch held-out {args.final_split}"

    bundle = run["models"]
    thresh = run["best_thresh_final"]
    final_tag = run["final_tag"]
    stack_meta = run.get("stack_meta") if final_tag == "stk" else None

    bundle_path = os.path.join(args.out_dir, "stage1_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump({
            "models": bundle,
            "stack_meta": stack_meta,
            "final_tag": final_tag,
            "threshold": thresh,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "feature_version": args.feature_version,
            "feature_names": feat_names,
            "class_names": CLASS_NAMES,
            "source": source,
        }, f)

    cfg = {
        **vars(args),
        "feature_dim": int(X.shape[1]),
        "class_names": CLASS_NAMES,
        "summary": {
            name: {"final_tag": run["final_tag"],
                   "auc": run["results"][run["final_tag"]]["auc"],
                   "acc_at_best": run["results"][run["final_tag"]]["acc_at_best"],
                   "best_thresh": run["results"][run["final_tag"]]["best_thresh"],
                   "confusion_matrix": run["confusion_matrix_final"]}
            for name, run in all_runs.items()
        },
        "saved_model_source": source,
        "saved_threshold": thresh,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("\nSaved outputs:")
    print(f"  Ensemble model: {bundle_path} (source: {source})")
    print(f"  Inference threshold: {thresh:.3f}")
    print(f"  Configuration: {os.path.join(args.out_dir, 'train_config.json')}")


if __name__ == "__main__":
    main()
