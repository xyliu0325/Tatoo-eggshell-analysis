"""Build features from RGB, HSV, Lab, channel statistics, and differences between adjacent points.
Feature engineering is especially important for the 303-sample dataset.
"""
import numpy as np
from skimage.color import rgb2hsv, rgb2lab


def _per_channel_stats(arr20x3, prefix):
    """Return per-channel statistics for the 20 RGB points in an image."""
    out = {}
    ch_names = list("RGB") if prefix == "rgb" else list(prefix.upper())
    for ci, cname in enumerate(ch_names):
        v = arr20x3[:, ci]
        out[f"{prefix}_{cname}_mean"] = float(np.mean(v))
        out[f"{prefix}_{cname}_std"]  = float(np.std(v))
        out[f"{prefix}_{cname}_min"]  = float(np.min(v))
        out[f"{prefix}_{cname}_max"]  = float(np.max(v))
        out[f"{prefix}_{cname}_p25"]  = float(np.percentile(v, 25))
        out[f"{prefix}_{cname}_p50"]  = float(np.percentile(v, 50))
        out[f"{prefix}_{cname}_p75"]  = float(np.percentile(v, 75))
        out[f"{prefix}_{cname}_range"] = float(np.ptp(v))
    return out


def _pairwise_diffs(arr20x3, prefix):
    """Return 57 channel differences from the 19 adjacent point pairs."""
    out = {}
    ch_names = list("RGB") if prefix == "rgb" else list(prefix.upper())
    diffs = np.diff(arr20x3, axis=0)
    for k in range(diffs.shape[0]):
        for ci, cname in enumerate(ch_names):
            out[f"{prefix}_d{k+1:02d}_{cname}"] = float(diffs[k, ci])
    return out


def build_features_from_rgb20(rgb20):
    """
    Accepts a (20, 3) array with RGB values in [0, 255] or [0, 1] and normalizes as needed.
    Returns an ordered feature dictionary and a list of feature names.
    """
    rgb = np.asarray(rgb20, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb01 = rgb / 255.0
    else:
        rgb01 = rgb.copy()
        rgb = rgb01 * 255.0

    rgb_img = rgb01.reshape(1, -1, 3)
    hsv = rgb2hsv(rgb_img).reshape(-1, 3)
    lab = rgb2lab(rgb_img).reshape(-1, 3)

    feat = {}

    for k in range(20):
        for ci, c in enumerate("RGB"):
            feat[f"p{k+1:02d}_{c}"] = float(rgb[k, ci])
    for k in range(20):
        for ci, c in enumerate("HSV"):
            feat[f"p{k+1:02d}_{c}"] = float(hsv[k, ci])
    for k in range(20):
        for ci, c in enumerate("Ll" if False else ["L", "a", "b"]):
            feat[f"p{k+1:02d}_{c}"] = float(lab[k, ci])

    feat.update(_per_channel_stats(rgb, "rgb"))
    feat.update(_per_channel_stats(hsv, "hsv"))
    feat.update(_per_channel_stats(lab, "lab"))

    feat.update(_pairwise_diffs(rgb, "rgb"))
    feat.update(_pairwise_diffs(hsv, "hsv"))
    feat.update(_pairwise_diffs(lab, "lab"))

    return feat


def build_X_v2(rgb20_list):
    """Convert a list of (20, 3) RGB arrays into a feature matrix and feature names."""
    feats = [build_features_from_rgb20(r) for r in rgb20_list]
    feat_names = list(feats[0].keys())
    X = np.array([[f[name] for name in feat_names] for f in feats], dtype=np.float32)
    return X, feat_names
