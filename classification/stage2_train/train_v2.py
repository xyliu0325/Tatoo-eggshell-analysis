"""Train the Stage 2 Swin classifier.
    1. Resize the whole image proportionally and center it on a black 224 x 224 canvas. No random augmentation is used.
    2. Use automatic mixed precision, with BF16 preferred on supported GPUs.
    3. Apply cosine learning rate decay with a linear warmup.
    4. Save the best checkpoint by validation accuracy and the final checkpoint, plus ROC and PR outputs.
    5. Export confusion matrices and per-egg labels, probabilities, and correctness.

Input and output directories are configurable.
The script extends the original train-edited-by-gy.py workflow.

Set --cv_folds to K (K >= 2) to train K models with StratifiedKFold.
Report the mean and standard deviation of validation accuracy, AUC, and AP. Save the most accurate fold as best.pth.
"""
import os
import csv
import shutil
import argparse
import secrets

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve
from openpyxl.styles import PatternFill
from PIL import Image

from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as create_model
from utilsnew import (
    read_split_data_seeded,
    train_one_epoch_amp,
    evaluate,
    plot_metrics_to,
    save_roc_curve,
    save_pr_curve,
    set_global_seed,
    seed_worker,
)


def str2bool(v):
    """Parse explicit Boolean command-line values.
    Using type=bool would treat any nonempty string, including 'False', as true."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('yes', 'true', 't', 'y', '1'):
        return True
    if s in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


class LetterboxResize:
    """Resize the whole image proportionally and center it on a square black canvas."""

    def __init__(self, size=224, fill=(0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = min(self.size / width, self.size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def build_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    spatial_transform = LetterboxResize(size=img_size, fill=(0, 0, 0))
    train_t = transforms.Compose([
        spatial_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_t = transforms.Compose([
        spatial_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_t, val_t


def build_scheduler(optimizer, epochs, warmup_epochs, base_lr):
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 1e-2)
    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs,
                                eta_min=base_lr * 1e-2)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def collect_all_data(root, supported=('.jpg', '.JPG', '.png', '.PNG')):
    classes = sorted([c for c in os.listdir(root)
                      if os.path.isdir(os.path.join(root, c))])
    class_indices = {k: v for v, k in enumerate(classes)}
    all_p, all_l = [], []
    for cla in classes:
        cla_path = os.path.join(root, cla)
        imgs = sorted([os.path.join(cla_path, i) for i in os.listdir(cla_path)
                       if os.path.splitext(i)[-1] in supported])
        for p in imgs:
            all_p.append(p)
            all_l.append(class_indices[cla])
    return all_p, all_l, class_indices


def _class_names_from_paths(paths, labels):
    """Recover the meaning of each class ID from the Female and Male directory names."""
    mapping = {}
    for path, label in zip(paths, labels):
        mapping[int(label)] = os.path.basename(os.path.dirname(path))
    return mapping


def _prediction_dataframe(paths, labels, probs, preds, split_name, fold_name):
    """Build a per-egg prediction table. In binary classification, probs is the probability of class 1."""
    class_names = _class_names_from_paths(paths, labels)
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    probs = np.asarray(probs, dtype=float)
    correct = labels == preds
    return pd.DataFrame({
        "egg_name": [os.path.basename(p) for p in paths],
        "image_path": paths,
        "dataset_split": split_name,
        "fold": fold_name,
        "true_label_id": labels,
        "true_label": [class_names.get(int(v), str(v)) for v in labels],
        "predicted_label_id": preds,
        "predicted_label": [class_names.get(int(v), str(v)) for v in preds],
        "class0_name": class_names.get(0, "class0"),
        "class0_probability": 1.0 - probs,
        "class1_name": class_names.get(1, "class1"),
        "class1_probability": probs,
        "prediction_confidence": np.maximum(1.0 - probs, probs),
        "correct": correct,
        "status": np.where(correct, "Correct", "Incorrect"),
    })


def _write_prediction_excel(df, path):
    """Save per-egg predictions and highlight incorrect rows in yellow."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False, sheet_name="Predictions")
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb["Predictions"]
    status_col = {
        cell.value: cell.column for cell in ws[1]
    }.get("status")
    yellow = PatternFill(fill_type="solid", fgColor="FFF2CC")
    if status_col is not None:
        for row in range(2, ws.max_row + 1):
            if ws.cell(row=row, column=status_col).value == "Incorrect":
                for col in range(1, ws.max_column + 1):
                    ws.cell(row=row, column=col).fill = yellow
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column in ws.columns:
        letter = column[0].column_letter
        max_len = max(len(str(cell.value or "")) for cell in column)
        ws.column_dimensions[letter].width = min(max(max_len + 2, 11), 55)
    wb.save(path)


def _write_split_csv(train_p, train_l, val_p, val_l, out_path):
    """Save the train and validation assignments for this run."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for split_name, paths, labels in (
            ("train", train_p, train_l), ("val", val_p, val_l)):
        for path, label in zip(paths, labels):
            rows.append({
                "split": split_name,
                "label_id": int(label),
                "label": os.path.basename(os.path.dirname(path)),
                "egg_name": os.path.basename(path),
                "image_path": path,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_and_print_pr_curve_data(labels, probs, out_dir, prefix="validation"):
    """Export and print the data used to plot the precision-recall curve."""
    os.makedirs(out_dir, exist_ok=True)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    threshold_col = np.empty_like(precision, dtype=float)
    threshold_col[:] = np.nan
    threshold_col[:-1] = thresholds
    df = pd.DataFrame({
        "threshold": threshold_col,
        "precision": precision,
        "recall": recall,
    })
    csv_path = os.path.join(out_dir, f"{prefix}_pr_curve_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n{prefix} PR curve data:")
    print(df.to_string(index=False))
    print(f"{prefix} PR curve data saved to: {csv_path}")
    return df, csv_path


def save_confusion_outputs(labels, preds, class_names, out_dir,
                           prefix="validation"):
    """Export a confusion matrix plot and the raw and row-normalized counts."""
    os.makedirs(out_dir, exist_ok=True)
    label_ids = sorted(class_names)
    if not label_ids:
        label_ids = sorted(set(labels) | set(preds))
    names = [class_names.get(i, str(i)) for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    matrix_path = os.path.join(out_dir, f"{prefix}_confusion_matrix_data.xlsx")
    with pd.ExcelWriter(matrix_path) as writer:
        pd.DataFrame(
            cm, index=[f"True {n}" for n in names],
            columns=[f"Pred {n}" for n in names],
        ).to_excel(writer, sheet_name="Counts")
        pd.DataFrame(
            cm_norm, index=[f"True {n}" for n in names],
            columns=[f"Pred {n}" for n in names],
        ).to_excel(writer, sheet_name="Row_Normalized")

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                      vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="Row-normalized proportion")
    ax.set(
        xticks=np.arange(len(names)), yticks=np.arange(len(names)),
        xticklabels=names, yticklabels=names,
        xlabel="Predicted label", ylabel="True label",
        title=f"{prefix.replace('_', ' ').title()} confusion matrix")
    threshold = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > threshold else "black",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(
        out_dir, f"{prefix}_confusion_matrix.png"),
        dpi=300, bbox_inches="tight")
    plt.close(fig)
    return cm


def build_model(args, device):
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights:
        assert os.path.exists(args.weights), f"weights file not exist: {args.weights}"
        sd = torch.load(args.weights, map_location=device)
        sd = sd.get("model", sd)
        for k in list(sd.keys()):
            if "head" in k:
                del sd[k]
        print(model.load_state_dict(sd, strict=False))
    if args.freeze_layers:
        for name, p in model.named_parameters():
            if "head" not in name:
                p.requires_grad_(False)
    return model


def train_and_eval_split(train_p, train_l, val_p, val_l, args, device,
                          weights_dir, fold_logs_dir, tag=""):
    """Train and evaluate one train/validation split.

    Use tag as a checkpoint suffix. An empty tag denotes a single split, and "_fold2" denotes the second fold.
    Write this fold's logs under fold_logs_dir.
    Return metrics, model paths, training history, and per-egg validation predictions.
    """
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(fold_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(fold_logs_dir, "tensorboard"))

    csv_path = os.path.join(fold_logs_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    train_t, val_t = build_transforms(img_size=224)
    train_ds = MyDataSet(train_p, train_l, transform=train_t)
    val_ds = MyDataSet(val_p, val_l, transform=val_t)

    bs = args.batch_size
    nw = min([os.cpu_count() or 1, max(bs, 1), 8])
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, pin_memory=True,
        num_workers=nw, collate_fn=train_ds.collate_fn, drop_last=False,
        generator=g, worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, pin_memory=True,
        num_workers=nw, collate_fn=val_ds.collate_fn,
        worker_init_fn=seed_worker)

    model = build_model(args, device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.lr)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "none": None}[args.amp_dtype]
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None

    history = {k: [] for k in ("epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "lr")}
    best_acc, best_epoch = -1.0, -1
    best_ckpt = os.path.join(weights_dir, f"best{tag}.pth")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_amp(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, amp_dtype=amp_dtype)
        val_loss, val_acc = evaluate(model, val_loader, device, epoch)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(("epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"),
                        (epoch, train_loss, train_acc, val_loss, val_acc, cur_lr)):
            history[k].append(v)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, cur_lr])
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        tb_writer.add_scalar('Learning Rate', cur_lr, epoch)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> new best val_acc = {val_acc:.4f} @ epoch {epoch}")

    plot_metrics_to(history, os.path.join(fold_logs_dir, "training_metrics.png"))

    print(f"\nReloading best (epoch {best_epoch}, val_acc {best_acc:.4f}) for ROC/PR ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, _, labels_all, probs_all, preds_all = evaluate(
        model, val_loader, device, epoch=best_epoch, return_outputs=True)
    roc_auc = save_roc_curve(
        labels_all, probs_all,
        save_path=os.path.join(fold_logs_dir, "roc_curve.png"),
        excel_path=os.path.join(fold_logs_dir, "roc_curve_data.xlsx"))
    ap = save_pr_curve(
        labels_all, probs_all,
        save_path=os.path.join(fold_logs_dir, "pr_curve.png"),
        )
    save_and_print_pr_curve_data(
        labels_all, probs_all, fold_logs_dir, prefix="validation")

    fold_name = tag.lstrip("_") if tag else "single_split"
    val_predictions = _prediction_dataframe(
        val_p, labels_all, probs_all, preds_all,
        split_name="validation", fold_name=fold_name)
    _write_prediction_excel(
        val_predictions,
        os.path.join(fold_logs_dir, "validation_egg_predictions.xlsx"))
    class_names = _class_names_from_paths(
        list(train_p) + list(val_p), list(train_l) + list(val_l))
    save_confusion_outputs(
        labels_all, preds_all, class_names, fold_logs_dir,
        prefix="validation")

    all_egg_predictions = None
    if args.cv_folds <= 1:
        train_eval_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=bs, shuffle=False, pin_memory=True,
            num_workers=nw, collate_fn=train_ds.collate_fn,
            worker_init_fn=seed_worker)
        _, _, train_labels, train_probs, train_preds = evaluate(
            model, train_eval_loader, device, epoch=best_epoch,
            return_outputs=True)
        train_predictions = _prediction_dataframe(
            train_p, train_labels, train_probs, train_preds,
            split_name="training", fold_name=fold_name)
        all_egg_predictions = pd.concat(
            [train_predictions, val_predictions], ignore_index=True)
        _write_prediction_excel(
            all_egg_predictions,
            os.path.join(fold_logs_dir, "all_egg_predictions.xlsx"))

    with open(os.path.join(fold_logs_dir, "summary.txt"), "w") as f:
        f.write(f"best_epoch     = {best_epoch}\n")
        f.write(f"best_val_acc   = {best_acc:.4f}\n")
        f.write(f"val_auc        = {roc_auc:.4f}\n")
        f.write(f"val_avg_prec   = {ap:.4f}\n")
        f.write(f"best_ckpt      = {best_ckpt}\n")

    return dict(best_acc=best_acc, best_epoch=best_epoch,
                val_auc=roc_auc, val_ap=ap,
                best_ckpt=best_ckpt, history=history,
                val_predictions=val_predictions,
                all_egg_predictions=all_egg_predictions)


def collect_stage2_data(root, classes=("Female", "Male"), supported=(".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")):
    """
    Read Stage 2 images. The batch layout is:

        root/Batch-3/Female/*.png
        root/Batch-3/Male/*.png
        root/Batch-4/Female/*.png
        root/Batch-4/Male/*.png
        ...

    The flat layout is also supported:

        root/Female/*.png
        root/Male/*.png

    Merge all batches into one dataset, using only Female and Male images.
    Assign class IDs in the order specified by classes.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Stage2 data root not found: {root}")

    class_indices = {name: idx for idx, name in enumerate(classes)}
    all_p, all_l = [], []

    batch_dirs = sorted(
        [os.path.join(root, name) for name in os.listdir(root)
         if os.path.isdir(os.path.join(root, name)) and name.lower().startswith("batch-")],
        key=lambda p: p.lower(),
    )

    if batch_dirs:
        for batch_dir in batch_dirs:
            for cls_name in classes:
                cls_dir = os.path.join(batch_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    ext = os.path.splitext(fname)[1]
                    if ext in supported:
                        all_p.append(os.path.join(cls_dir, fname))
                        all_l.append(class_indices[cls_name])
    else:
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1]
                if ext in supported:
                    all_p.append(os.path.join(cls_dir, fname))
                    all_l.append(class_indices[cls_name])

    if not all_p:
        expected = " / ".join(f"Batch-*/{c}/*.png" for c in classes)
        raise RuntimeError(
            f"No Stage2 images found under: {root}\n"
            f"Expected structure like: {expected}")

    print(f"Stage2 data root: {root}")
    print(f"Merged batches: {len(batch_dirs) if batch_dirs else 0}")
    for cls_name, cls_idx in class_indices.items():
        count = sum(1 for y in all_l if y == cls_idx)
        print(f"  {cls_name}: {count} images")
    print(f"  Total: {len(all_p)} images")

    return all_p, all_l, class_indices


def read_stage2_split_data(root, val_rate=0.3, seed=None, json_out=None,
                           split_csv=None):
    """
    Merge batches before making a stratified 70/30 split.
    Use only images from the Female and Male directories.
    """
    all_p, all_l, class_indices = collect_stage2_data(root)

    from sklearn.model_selection import train_test_split
    if not (0.0 < val_rate < 1.0):
        raise ValueError(f"val_rate must be between 0 and 1, got {val_rate}")

    train_p, val_p, train_l, val_l = train_test_split(
        all_p, all_l, test_size=val_rate, random_state=seed, stratify=all_l
    )

    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            import json
            json.dump({str(v): k for k, v in class_indices.items()},
                      f, ensure_ascii=False, indent=2)

    print(f"Seed: {seed}")
    print(f"Train: {len(train_p)} | Validation: {len(val_p)}")
    print("Validation examples:")
    for p in val_p[:10]:
        print(f"  {os.path.basename(p)}")
    if split_csv:
        _write_split_csv(train_p, train_l, val_p, val_l, split_csv)
        print(f"Train/validation split saved to: {split_csv}")
    return train_p, train_l, val_p, val_l


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights_dir = os.path.join(args.out_dir, "weights")
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if args.seed is None:
        args.seed = secrets.randbelow(900000) + 100000
        print(f"Random seed generated for this run: {args.seed}")
    else:
        print(f"Using user-provided seed: {args.seed}")
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.cv_folds <= 1:
        train_p, train_l, val_p, val_l = read_stage2_split_data(
            args.data_path, val_rate=args.val_rate, seed=args.seed,
            json_out=os.path.join(logs_dir, "class_indices.json"),
            split_csv=os.path.join(logs_dir, "train_val_split.csv"))
        result = train_and_eval_split(
            train_p, train_l, val_p, val_l, args, device,
            weights_dir, logs_dir, tag="")
        final_path = os.path.join(weights_dir, "final_model.pth")
        if os.path.exists(result['best_ckpt']):
            shutil.copy2(result['best_ckpt'], final_path)
        print(f"\nDone. best_val_acc = {result['best_acc']:.4f}, "
              f"AUC = {result['val_auc']:.4f}")
        print(f"Artifacts: {args.out_dir}")
        return

    print(f"\n========== {args.cv_folds}-fold StratifiedKFold ==========")
    all_p, all_l, class_indices = collect_stage2_data(args.data_path)
    print(f"total samples: {len(all_p)}, classes: {class_indices}")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    results = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(all_p, all_l)):
        fold_tag = f"_fold{k+1}"
        fold_logs = os.path.join(logs_dir, f"fold{k+1}")
        print(f"\n========== Fold {k+1}/{args.cv_folds} "
              f"(train={len(tr_idx)}, val={len(va_idx)}) ==========")
        set_global_seed(args.seed + k, deterministic=args.deterministic)
        tr_p = [all_p[i] for i in tr_idx]
        tr_l = [all_l[i] for i in tr_idx]
        va_p = [all_p[i] for i in va_idx]
        va_l = [all_l[i] for i in va_idx]
        _write_split_csv(
            tr_p, tr_l, va_p, va_l,
            os.path.join(fold_logs, "train_val_split.csv"))
        res = train_and_eval_split(
            tr_p, tr_l, va_p, va_l, args, device,
            weights_dir, fold_logs, tag=fold_tag)
        results.append(res)

    oof_predictions = pd.concat(
        [r["val_predictions"] for r in results], ignore_index=True)
    oof_predictions = oof_predictions.sort_values(
        ["egg_name", "image_path"]).reset_index(drop=True)
    _write_prediction_excel(
        oof_predictions,
        os.path.join(logs_dir, "cv_all_egg_oof_predictions.xlsx"))
    save_confusion_outputs(
        oof_predictions["true_label_id"].to_numpy(),
        oof_predictions["predicted_label_id"].to_numpy(),
        {v: k for k, v in class_indices.items()},
        logs_dir, prefix="cv_oof")

    accs = np.array([r['best_acc'] for r in results])
    aucs = np.array([r['val_auc'] for r in results])
    aps = np.array([r['val_ap'] for r in results])
    cv_summary_path = os.path.join(logs_dir, "cv_summary.txt")
    with open(cv_summary_path, "w") as f:
        f.write(f"cv_folds       = {args.cv_folds}\n")
        f.write(f"seed           = {args.seed}\n")
        f.write(f"total_samples  = {len(all_p)}\n")
        f.write(f"classes        = {class_indices}\n")
        f.write(f"---\n")
        for k, r in enumerate(results):
            f.write(f"fold{k+1}: best_acc={r['best_acc']:.4f}  "
                    f"AUC={r['val_auc']:.4f}  AP={r['val_ap']:.4f}  "
                    f"best_epoch={r['best_epoch']}\n")
        f.write(f"---\n")
        f.write(f"mean_val_acc   = {accs.mean():.4f} +/- {accs.std():.4f}\n")
        f.write(f"mean_val_auc   = {aucs.mean():.4f} +/- {aucs.std():.4f}\n")
        f.write(f"mean_val_ap    = {aps.mean():.4f} +/- {aps.std():.4f}\n")

    best_k = int(np.argmax(accs))
    src = results[best_k]['best_ckpt']
    dst = os.path.join(weights_dir, "best.pth")
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)
    print(f"\n========== CV done ==========")
    print(f"mean_val_acc = {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"mean_val_auc = {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"mean_val_ap  = {aps.mean():.4f} +/- {aps.std():.4f}")
    print(f"best fold = fold{best_k+1} (acc {accs[best_k]:.4f}) -> {dst}")
    print(f"Full summary: {cv_summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--val_rate', type=float, default=0.3,
                        help='Validation fraction for a single split (cv_folds <= 1)')
    parser.add_argument('--cv_folds', type=int, default=1,
                        help='Use stratified K-fold cross-validation for values of 2 or more')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the split. A new seed is generated for each run when omitted.')
    parser.add_argument('--amp_dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16', 'none'])
    parser.add_argument('--data-path', type=str, required=True,
                        help='Image root containing Female and Male directories, with optional Batch-* subdirectories')
    parser.add_argument('--weights', type=str, default='',
                        help='Path to the pretrained swin_base_patch4_window7_224_22k.pth checkpoint')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for weights and logs')
    parser.add_argument('--freeze-layers', type=str2bool, nargs='?', const=True,
                        default=False,
                        help='True or False. Using --freeze-layers without a value sets it to True.')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True,
                        default=True,
                        help='Use deterministic seeding and cuDNN operations for reproducibility')
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()
    main(opt)
