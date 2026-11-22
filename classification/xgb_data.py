"""Read 20 RGB points from each XLSX workbook under Batch-X/{Female,Male,Dead} and optionally build additional features."""
import os, glob
import numpy as np
import pandas as pd

from features import build_X_v2


BATCHES_DEFAULT = ("Batch-3", "Batch-4", "Batch-5")

FEATURE_NAMES_V1 = []
for k in range(1, 21):
    for ch in ["R", "G", "B"]:
        FEATURE_NAMES_V1.append(f"p{k:02d}_{ch}")


def _read_rgb20(xlsx_path):
    df = pd.read_excel(xlsx_path)
    rgb = df.iloc[:, -3:].to_numpy(dtype=np.float32)
    return rgb if rgb.shape[0] == 20 else None


def load_excel_features(data_root, task, batches=BATCHES_DEFAULT, version="v1"):
    """
    task='da': Dead=1, Female∪Male=0
    task='fm': Female=0, Male=1. Ignore Dead.
    version='v1': 60 raw RGB features for compatibility with earlier models.
    version='v2': About 430 features from RGB, HSV, Lab, channel statistics, and differences between points.
    Return X, y, batch_ids, stems, and feat_names.
    """
    if task == "da":
        cls_map = {"Female": 0, "Male": 0, "Dead": 1}
    elif task == "fm":
        cls_map = {"Female": 0, "Male": 1}
    else:
        raise ValueError(f"task must be 'da' or 'fm', got {task}")

    rgbs, y, batch_ids, stems = [], [], [], []
    for b in batches:
        for cls_name, cls_idx in cls_map.items():
            folder = os.path.join(data_root, b, cls_name)
            if not os.path.isdir(folder):
                continue
            for p in sorted(glob.glob(os.path.join(folder, "*.xlsx"))):
                rgb = _read_rgb20(p)
                if rgb is None:
                    print(f"  Skipping {p}: expected 20 rows.")
                    continue
                rgbs.append(rgb)
                y.append(cls_idx)
                batch_ids.append(b)
                stems.append(os.path.splitext(os.path.basename(p))[0])

    if version == "v1":
        X = np.array([r.flatten() for r in rgbs], dtype=np.float32)
        feat_names = list(FEATURE_NAMES_V1)
    elif version == "v2":
        X, feat_names = build_X_v2(rgbs)
    else:
        raise ValueError(f"version must be 'v1' or 'v2', got {version}")

    return (X, np.array(y, dtype=np.int64),
            np.array(batch_ids), np.array(stems), feat_names)


def load_features_for_paths(xlsx_paths, version="v1"):
    """Load an (N, F) feature matrix for prediction using the same feature version as training."""
    rgbs = []
    for p in xlsx_paths:
        rgb = _read_rgb20(p)
        if rgb is None:
            raise ValueError(f"{p}: expected 20 rows.")
        rgbs.append(rgb)
    if version == "v1":
        return np.array([r.flatten() for r in rgbs], dtype=np.float32)
    elif version == "v2":
        X, _ = build_X_v2(rgbs)
        return X
    else:
        raise ValueError(f"version must be 'v1' or 'v2', got {version}")


FEATURE_NAMES = FEATURE_NAMES_V1
