"""Load Stage 2 checkpoints from the original train-edited-by-gy.py workflow.

The original workflow uses these conventions:
  - Model: swin_base_patch4_window7_224_in22k(num_classes=2) from model.py.
  - Class labels in alphabetical order: ["Female", "Male"].
  - Image preprocessing: Resize(256), CenterCrop(224), ToTensor, and ImageNet normalization.
  - Checkpoint: a PyTorch state_dict saved with torch.save(model.state_dict(), ...).

The bundled stage2_train/model.py provides the required
swin_base_patch4_window7_224_in22k model constructor.
"""
import os, sys
import torch
from torchvision import transforms


CLIENT_CLASS_NAMES = ["Female", "Male"]
STAGE2_TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_train")


def build_stage2_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_stage2(ckpt_path, device, num_classes=2):
    if STAGE2_TRAIN_DIR not in sys.path:
        sys.path.insert(0, STAGE2_TRAIN_DIR)
    try:
        from model import swin_base_patch4_window7_224_in22k as create_model
    except ImportError as e:
        raise ImportError(
            f"Could not import swin_base_patch4_window7_224_in22k from stage2_train/model.py. "
            f"Check {STAGE2_TRAIN_DIR}/model.py.\nOriginal error: {e}")

    net = create_model(num_classes=num_classes)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Stage 2] Missing weights: {len(missing)}. First entries: {missing[:3]}")
    if unexpected:
        print(f"[Stage 2] Unexpected weights: {len(unexpected)}. First entries: {unexpected[:3]}")
    net.to(device).eval()
    return net, list(CLIENT_CLASS_NAMES)
