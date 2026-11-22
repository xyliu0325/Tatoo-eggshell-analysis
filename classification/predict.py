"""Run Stage 1 on Excel RGB data and Stage 2 on PNG color-difference maps.

Provide an XLSX workbook and a PNG with the same filename stem for each sample.
Usage:
    Single sample: --excel <egg_N.xlsx> --image <egg_N.png>
    Batch: --excel_dir <xlsx_directory> --image_dir <png_directory>. Unpaired files are reported.
"""
import os, argparse, glob, csv, pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import xgboost as xgb

from xgb_data import load_features_for_paths
from stage2_io import build_stage2_transform, load_stage2


def load_stage1(model_path):
    """Load a Stage 1 model in one of two formats:
    - .json: a single XGBoost model using v1 features.
    - .pkl: an ensemble bundle with its threshold and feature version.
    Return a dictionary with a prediction function, threshold, and model metadata.
    """
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            b = pickle.load(f)
        models = b["models"]
        threshold = float(b["threshold"])
        feature_version = b.get("feature_version", "v1")
        stack_meta = b.get("stack_meta")
        final_tag = b.get("final_tag", "ens")
        ensemble_weights = b.get("ensemble_weights")
        if ensemble_weights is None:
            ensemble_weights = {
                "xgb": 0.25, "lgb": 0.25, "rf": 0.25, "et": 0.25}

        def _family_avg(clfs_or_clf, X):
            if isinstance(clfs_or_clf, list):
                return np.mean([c.predict_proba(X)[:, 1] for c in clfs_or_clf], axis=0)
            return clfs_or_clf.predict_proba(X)[:, 1]

        def predict_proba_dead(xlsx_paths):
            X = load_features_for_paths(xlsx_paths, version=feature_version)
            family_probs = {fam: _family_avg(clfs, X) for fam, clfs in models.items()}
            if final_tag == "stk" and stack_meta is not None:
                cols = [family_probs[k] for k in ["xgb", "lgb", "rf", "et"] if k in family_probs]
                stack_in = np.column_stack(cols)
                return stack_meta.predict_proba(stack_in)[:, 1]
            required = ("xgb", "lgb", "rf", "et")
            missing = [k for k in required if k not in family_probs]
            if missing:
                raise KeyError(f"Stage 1 bundle is missing base models: {missing}")
            return sum(
                float(ensemble_weights[k]) * family_probs[k]
                for k in required)
        return {"predict_proba_dead": predict_proba_dead, "threshold": threshold,
                "feature_version": feature_version,
                "kind": f"ensemble({final_tag})",
                "ensemble_weights": ensemble_weights,
                "final_tag": final_tag}
    else:
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)

        def predict_proba_dead(xlsx_paths):
            X = load_features_for_paths(xlsx_paths, version="v1")
            return clf.predict_proba(X)[:, 1]
        return {"predict_proba_dead": predict_proba_dead, "threshold": 0.5,
                "feature_version": "v1", "kind": "xgb_only"}


@torch.no_grad()
def st_predict_batch(model, img_paths, transform, device, batch_size=16):
    """Run Stage 2 on image batches and return an (N, 2) probability array."""
    probs = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        xs = [transform(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(xs).to(device)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 2))


def pair_inputs(excel_dir, image_dir):
    """Pair XLSX and PNG files by stem and report any unpaired files."""
    xlsx_map = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(excel_dir, "*.xlsx"))}
    png_map = {os.path.splitext(os.path.basename(p))[0]: p
               for p in sorted(glob.glob(os.path.join(image_dir, "*.png")))}
    common = sorted(set(xlsx_map) & set(png_map))
    only_xlsx = sorted(set(xlsx_map) - set(png_map))
    only_png = sorted(set(png_map) - set(xlsx_map))
    return (common,
            [xlsx_map[s] for s in common],
            [png_map[s] for s in common],
            {"only_xlsx": only_xlsx, "only_png": only_png})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_model", required=True,
                    help="Stage 1 model: model.json (single XGBoost model) or stage1_bundle.pkl (ensemble)")
    ap.add_argument("--stage2_ckpt", required=True,
                    help="Trained Stage 2 checkpoint, such as final_model.pth (PyTorch state_dict)")
    g_single = ap.add_argument_group("Single-sample input")
    g_single.add_argument("--excel", help="Path to one XLSX workbook")
    g_single.add_argument("--image", help="Path to one PNG image")
    g_batch = ap.add_argument_group("Batch input")
    g_batch.add_argument("--excel_dir", help="Directory containing XLSX workbooks")
    g_batch.add_argument("--image_dir", help="Directory containing PNG images")
    ap.add_argument("--dead_threshold", type=float, default=None,
                    help="Stage 1 threshold. Defaults to the bundle threshold or 0.5 for a single XGBoost model.")
    ap.add_argument("--st_batch_size", type=int, default=16)
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    if args.excel and args.image:
        stems = [os.path.splitext(os.path.basename(args.excel))[0]]
        xlsx_paths = [args.excel]
        png_paths = [args.image]
        missing = {"only_xlsx": [], "only_png": []}
    elif args.excel_dir and args.image_dir:
        stems, xlsx_paths, png_paths, missing = pair_inputs(args.excel_dir, args.image_dir)
        if missing["only_xlsx"]:
            print(f"Warning: {len(missing['only_xlsx'])} XLSX workbooks have no matching PNG: "
                  f"{missing['only_xlsx'][:5]}{'...' if len(missing['only_xlsx'])>5 else ''}")
        if missing["only_png"]:
            print(f"Warning: {len(missing['only_png'])} PNG images have no matching XLSX: "
                  f"{missing['only_png'][:5]}{'...' if len(missing['only_png'])>5 else ''}")
    else:
        ap.error("Provide --excel and --image, or --excel_dir and --image_dir.")

    if not stems:
        print("No paired samples to predict.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1 = load_stage1(args.stage1_model)
    stage2, s2_cls = load_stage2(args.stage2_ckpt, device)
    transform = build_stage2_transform(224)

    threshold = args.dead_threshold if args.dead_threshold is not None else stage1["threshold"]
    print(f"Predicting {len(stems)} samples. Stage 1: {stage1['kind']} "
          f"(features={stage1['feature_version']})  thresh={threshold:.3f}")
    if stage1.get("final_tag") == "ens":
        weights_text = ", ".join(
            f"{k}={v:.2f}" for k, v in stage1["ensemble_weights"].items())
        print(f"stage1 ensemble weights: {weights_text}")

    p_dead = stage1["predict_proba_dead"](xlsx_paths)
    is_dead = p_dead >= threshold
    alive_idx = np.where(~is_dead)[0]

    rows = [None] * len(stems)
    for i, dead in enumerate(is_dead):
        rows[i] = {
            "stem": stems[i],
            "xlsx": xlsx_paths[i],
            "png": png_paths[i],
            "p_dead": float(p_dead[i]),
            "p_alive": float(1 - p_dead[i]),
            "p_female": None,
            "p_male": None,
            "label": "Dead" if dead else None,
            "confidence": float(p_dead[i]) if dead else None,
        }

    if len(alive_idx) > 0:
        alive_pngs = [png_paths[i] for i in alive_idx]
        prob2 = st_predict_batch(stage2, alive_pngs, transform, device, args.st_batch_size)
        f_idx = s2_cls.index("Female")
        m_idx = s2_cls.index("Male")
        for k, i in enumerate(alive_idx):
            p_f = float(prob2[k, f_idx])
            p_m = float(prob2[k, m_idx])
            rows[i]["p_female"] = p_f
            rows[i]["p_male"] = p_m
            if p_m >= p_f:
                rows[i]["label"] = "Male"
                rows[i]["confidence"] = p_m
            else:
                rows[i]["label"] = "Female"
                rows[i]["confidence"] = p_f

    for r in rows:
        extra = ""
        if r["label"] != "Dead":
            extra = f"  p_F={r['p_female']:.3f} p_M={r['p_male']:.3f}"
        print(f"  {r['stem']:>12s}  ->  {r['label']:<6s}  "
              f"(conf={r['confidence']:.3f}, p_dead={r['p_dead']:.3f}){extra}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "label", "confidence",
                                              "p_dead", "p_alive", "p_female", "p_male",
                                              "xlsx", "png"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nPredictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
