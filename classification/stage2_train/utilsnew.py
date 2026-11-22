import os
import sys
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score)
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np


def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(999)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def plot_metrics(history):
    """Plot and save training metrics."""
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Training Accuracy')
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if 'lr' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['epoch'], history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()

    plt.tight_layout()
    out_path = './logs/training_metrics.png'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_metrics_to(history, out_path):
    """Plot and save training metrics in the specified directory."""
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Training Accuracy')
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

    if 'lr' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['epoch'], history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs'); plt.ylabel('Learning Rate'); plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def train_one_epoch_amp(model, optimizer, data_loader, device, epoch,
                        scaler=None, amp_dtype=None):
    """Train for one epoch with optional automatic mixed precision.

    Use torch.bfloat16 without a gradient scaler on hardware with native BF16 support.
    Use a gradient scaler with torch.float16.
    Use FP32 when amp_dtype is None.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    use_amp = amp_dtype is not None and device.type == 'cuda'
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                pred = model(images)
                loss = loss_function(pred, labels)
        else:
            pred = model(images)
            loss = loss_function(pred, labels)

        bsz = images.shape[0]
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss.detach() * bsz

        if use_amp and amp_dtype == torch.float16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / sample_num, accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / sample_num, accu_num.item() / sample_num


def set_global_seed(seed: int, deterministic: bool = True):
    """Seed random, NumPy, PyTorch, and CUDA.

    Deterministic mode disables cuDNN benchmarking and chooses reproducible kernels.
    Some convolution kernels can run more slowly in this mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed each DataLoader worker's NumPy and random generators from PyTorch
    so multiprocessing data augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_split_data_seeded(root: str, val_rate: float = 0.3, seed: int = 999,
                            json_out: str = './class_indices.json'):
    """Split the dataset using the given seed and save the class mapping to the specified path."""
    rng = random.Random(seed)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    classes = sorted([c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))])
    class_indices = {k: v for v, k in enumerate(classes)}

    os.makedirs(os.path.dirname(json_out) or '.', exist_ok=True)
    with open(json_out, 'w') as jf:
        json.dump({v: k for k, v in class_indices.items()}, jf, indent=4)

    train_p, train_l, val_p, val_l, per_class = [], [], [], [], []
    supported = ('.jpg', '.JPG', '.png', '.PNG')
    for cla in classes:
        cla_path = os.path.join(root, cla)
        imgs = sorted([os.path.join(cla_path, i) for i in os.listdir(cla_path)
                       if os.path.splitext(i)[-1] in supported])
        per_class.append(len(imgs))
        idx = class_indices[cla]
        val_pick = set(rng.sample(imgs, k=int(len(imgs) * val_rate)))
        for p in imgs:
            if p in val_pick:
                val_p.append(p); val_l.append(idx)
            else:
                train_p.append(p); train_l.append(idx)

    print("{} images were found in the dataset.".format(sum(per_class)))
    print("{} images for training.".format(len(train_p)))
    print("{} images for validation.".format(len(val_p)))
    assert len(train_p) > 0 and len(val_p) > 0
    return train_p, train_l, val_p, val_l


def save_roc_curve(true_labels, probs,
                   save_path='./logs/roc_curve.png',
                   excel_path='./logs/roc_curve_data.xlsx'):
    """
    Plot and save the ROC curve, and export the curve data to Excel or CSV.

    Args:
            true_labels: Ground-truth binary labels as a list or array.
            probs: Model probabilities as a list or array.
            save_path: Path for the ROC curve image.
            excel_path: Path for the ROC curve workbook (.xlsx).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Threshold': thresholds
    })

    try:
        df.to_excel(excel_path, index=False)
        print(f"ROC data saved to Excel: {excel_path}")
    except ModuleNotFoundError:
        csv_path = excel_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"openpyxl is unavailable. ROC data saved as CSV: {csv_path}")

    print(f"ROC curve saved to: {save_path}")
    return roc_auc

def save_pr_curve(true_labels, probs, save_path='./logs/pr_curve.png'):
    """Plot and save the precision-recall curve and return average precision.

    Use average_precision_score for AP. Taking the arithmetic mean of the
    precision values ignores the uneven spacing between recall points.
    """
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    average_precision = float(average_precision_score(true_labels, probs))

    plt.figure()
    plt.step(recall, precision, where='post',
             label=f'PR curve (AP={average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return average_precision


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, return_outputs=False):
    """Evaluate the model and optionally return its predictions."""
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    all_labels = []
    all_preds = []
    all_probs = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        bsz = images.shape[0]
        sample_num += bsz

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.max(outputs, dim=1)[1]
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        loss = loss_function(outputs, labels)
        accu_loss += loss * bsz
        accu_num += torch.eq(preds, labels).sum()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / sample_num, accu_num.item() / sample_num
        )

    val_loss = accu_loss.item() / sample_num
    val_acc = accu_num.item() / sample_num

    if return_outputs:
        return val_loss, val_acc, all_labels, all_probs, all_preds
    else:
        return val_loss, val_acc
