import os
import time
import random
from dataclasses import dataclass

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from models.vision_transformer import VisionTransformer
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# ImageNet normalization params
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_cv(image_rgb: np.ndarray, img_size: int = 224) -> np.ndarray:
    """Resize + [0,1] + normalize using ImageNet stats. Input expects RGB image (H, W, C).
    Returns Numpy float32 array in (H, W, C).
    """
    image = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def to_tensor_nchw(image_hwc: np.ndarray) -> torch.Tensor:
    """Convert HWC float32 in [-?, ?] to NCHW torch tensor with batch dim=1 on current device."""
    x = torch.from_numpy(image_hwc)  # (H, W, C)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().float()  # (1, C, H, W)
    return x


# -----------------------------
# Dataset using torchvision ImageFolder
# -----------------------------
class ImageFolderWithCVTransforms(ImageFolder):
    def __init__(self, root: str, img_size: int = 224):
        self.img_size = img_size
        # We'll use a callable transform that uses OpenCV then converts to tensor
        def opencv_transform(path: str):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr = preprocess_cv(rgb, img_size=self.img_size)
            tensor = to_tensor_nchw(arr)  # (1, C, H, W)
            return tensor.squeeze(0)      # (C, H, W)

        super().__init__(root=root, loader=lambda p: opencv_transform(p), target_transform=None)

    # Override to return (tensor, label)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)  # (C, H, W) tensor float32
        return sample, target


# -----------------------------
# Micron‑BERT loader
# -----------------------------

def load_micronbert_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Instantiate the model defined in models/ using the saved args, then load state_dict.
    Returns (model, args_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint must contain 'args' and 'model' keys.")

    args = ckpt["args"]

    # Fallback to dict in case args is a Namespace-like with attributes
    def getattr_fallback(o, k, default=None):
        try:
            return getattr(o, k)
        except Exception:
            return o.get(k, default) if isinstance(o, dict) else default

    model_name = getattr_fallback(args, "model_name", None)
    if model_name is None:
        raise ValueError("'model_name' not found in checkpoint args.")

    # The original micron_bert.py imports models.mae as mae_dict and constructs by name
    import importlib
    mae_dict = importlib.import_module("models.mae")

    model = mae_dict.__dict__[model_name](
        has_decoder=getattr_fallback(args, "has_decoder", False),
        aux_cls=getattr_fallback(args, "aux_cls", False),
        img_size=getattr_fallback(args, "img_size", 224),
        att_loss=getattr_fallback(args, "att_loss", False),
        diag_att=getattr_fallback(args, "diag_att", False),
        # DINO params (some models use them)
        enable_dino=getattr_fallback(args, "enable_dino", False),
        out_dim=getattr_fallback(args, "out_dim", 0),
        local_crops_number=getattr_fallback(args, "local_crops_number", 0),
        warmup_teacher_temp=getattr_fallback(args, "warmup_teacher_temp", 0.04),
        teacher_temp=getattr_fallback(args, "teacher_temp", 0.04),
        warmup_teacher_temp_epochs=getattr_fallback(args, "warmup_teacher_temp_epochs", 0),
        epochs=getattr_fallback(args, "epochs", 0),
    )

    # Remove 'module.' if present (from DataParallel)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys in state_dict:", unexpected)
    if len(missing) > 0:
        print("[WARN] Missing keys when loading state_dict:", missing)

    model.to(device)
    return model, args


# -----------------------------
# Binary classifier head wrapper
# -----------------------------
class BinaryHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.fc(self.dropout(x))


class MicronBERTBinary(nn.Module):
    """Attempts to use `extract_features` if available; otherwise uses forward() output.
    A projection (pooling) is applied if the base returns a sequence.
    """
    def __init__(self, base_model: nn.Module, feature_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.feature_dim = feature_dim
        self.head = None

        # Try to infer feature dimension by running a dummy pass if not provided
        if self.feature_dim is None:
            self.feature_dim = self._infer_feat_dim()
        self.head = BinaryHead(self.feature_dim, dropout=dropout)

    def _infer_feat_dim(self) -> int:
        self.base.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=next(self.base.parameters()).device)
            feats = self._extract_feats(dummy)
            if feats.ndim == 3:  # (B, N, C) sequence -> mean pool
                feats = feats.mean(dim=1)
            elif feats.ndim == 4:  # (B, C, H, W) -> global avg pool
                feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            return feats.shape[-1]

    def _extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer method named extract_features
        if hasattr(self.base, "extract_features"):
            return self.base.extract_features(x)
        # Else try forward (some models return logits; we must hook earlier)
        out = self.base(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_feats(x)
        if feats.ndim == 3:
            feats = feats.mean(dim=1)
        elif feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.head(feats)
        return logits


# -----------------------------
# Training / Evaluation helpers
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def run_one_epoch(model, loader, optimizer, device, scaler=None, train=True):
    total_loss, total_correct, total_count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    model.train(mode=train)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


# -----------------------------
# Main
# -----------------------------

def main():
    # print(os.getcwd())
    # Usar un diccionario de configuración en vez de argparse
    config = {
        "train_dir": "../../CASME/CASME-II-Binary-splitted/train",
        "val_dir": "../../CASME/CASME-II-Binary-splitted/validation",
        "test_dir": "../../CASME/CASME-II-Binary-splitted/test",  # o "data/test" si se desea usar test

        "checkpoint": "checkpoints/CASME2-is224-p8-b16-ep200.pth",
        "img_size": 224,

        "epochs": 10,
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.1,

        "freeze_base": False,
        "amp": False,

        "num_workers": 4,
        "seed": 42,
        "out_dir": "checkpoints_binary_10_epoch"
    }

    set_seed(config["seed"])
    os.makedirs(config["out_dir"], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = ImageFolderWithCVTransforms(config["train_dir"], img_size=config["img_size"])
    val_ds   = ImageFolderWithCVTransforms(config["val_dir"],   img_size=config["img_size"])
    test_ds  = ImageFolderWithCVTransforms(config["test_dir"],  img_size=config["img_size"]) if config["test_dir"] else None

    # Class mapping
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    print("Classes:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True) if test_ds else None

    # Base model (sin checkpoint)
    base_model = VisionTransformer(
                img_size=(224, 224),
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
                num_classes=0).to(device)

    base_args = None  # si tu código lo necesita
    

    # Wrap for binary classification
    clf = MicronBERTBinary(base_model, feature_dim=None, dropout=config["dropout"])

    if config["freeze_base"]:
        for p in clf.base.parameters():
            p.requires_grad = False
        print("Frozen base model parameters. Training only the head.")

    clf.to(device)

    # === SummaryWriter para TensorBoard ===
    writer = SummaryWriter(log_dir=f"runs/micron_bert_binary_{config['epochs']}_epochs")

    # Optimizer & scaler
    trainable_params = [p for p in clf.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    best_val_acc = 0.0
    best_path = os.path.join(config["out_dir"], 'best_model.pth')


    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        tr_loss, tr_acc = run_one_epoch(clf, train_loader, optimizer, device, scaler=scaler, train=True)
        va_loss, va_acc = run_one_epoch(clf, val_loader,   optimizer, device, scaler=None,  train=False)
        elapsed = time.time() - start

        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} | {elapsed:.1f}s")

        # --- Log a TensorBoard ---
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", va_acc, epoch)

        # Save best
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'epoch': epoch,
                'state_dict': clf.state_dict(),
                'base_checkpoint': config["checkpoint"],
                'idx_to_class': idx_to_class,
                'img_size': config["img_size"],
            }, best_path)
            print(f"[+] Saved best to {best_path} (val_acc={best_val_acc:.4f})")

    # Final test
    if test_loader is not None:
        # Load best
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location='cpu')
            clf.load_state_dict(ckpt['state_dict'])
        te_loss, te_acc = run_one_epoch(clf, test_loader, optimizer, device, scaler=None, train=False)
        print(f"TEST | loss {te_loss:.4f} acc {te_acc:.4f}")

        # --- Log test metrics también ---
        writer.add_scalar("Loss/test", te_loss)
        writer.add_scalar("Accuracy/test", te_acc)

    # cerrar writer
    writer.close()


if __name__ == '__main__':
    main()