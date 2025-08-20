#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Micron‑BERT Pairs Binary — entrenamiento e inferencia con pares (onset, apex)
para clasificación binaria: MicroExpresión (1) vs No‑Microexpresión (0).

Este script incluye TODO en un solo archivo:
- Dataset que empareja automáticamente *onset/apex* dentro de cada clase.
- Fine‑tuning de Micron‑BERT (encoder compartido) con cabeza binaria.
- Métricas: accuracy, precision, recall, F1, ROC‑AUC (val/test).
- Logging en CSV + checkpoints (best + last).
- Inferencia por carpeta con pares.

Estructura esperada para TRAIN/VAL/TEST (ImageFolder-like):

    data/
      train/
        micro/      # contiene archivos *_onset.* y *_apex.* del mismo prefijo
        nomicro/
      val/
        micro/
        nomicro/
      test/ (opcional)
        micro/
        nomicro/

Convención de nombres de archivos:
  <id>_onset.jpg/png/...  y  <id>_apex.jpg/png/...

Ejemplos de uso:

  # Entrenamiento con base congelada (solo cabeza)
  python micronbert_pairs_binary.py train \
    --train_dir data/train --val_dir data/val --test_dir data/test \
    --checkpoint checkpoints/CASME2-is224-p8-b16-ep200.pth \
    --out_dir runs/pairs_bin --freeze_base --epochs 20 --bs 16 --lr 1e-4 --amp

  # Fine‑tuning end‑to‑end (descongelado)
  python micronbert_pairs_binary.py train \
    --train_dir data/train --val_dir data/val \
    --checkpoint checkpoints/CASME2-is224-p8-b16-ep200.pth \
    --out_dir runs/pairs_e2e --epochs 10 --bs 8 --lr 1e-5 --amp

  # Inferencia sobre una carpeta con pares (sin subcarpetas de clase)
  # Estructura: input_dir/  contiene  <id>_onset.png  y  <id>_apex.png
  python micronbert_pairs_binary.py infer \
    --input_dir samples_pairs \
    --checkpoint_best runs/pairs_bin/best_model.pth \
    --output_csv preds.csv
"""

import os
import re
import csv
import sys
import math
import time
import json
import glob
import copy
import argparse
from dataclasses import dataclass

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# -----------------------------
# ImageNet normalization params
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_cv(image_rgb: np.ndarray, img_size: int = 224) -> np.ndarray:
    image = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor_nchw(image_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(image_hwc)  # (H, W, C)
    x = x.permute(2, 0, 1).contiguous().float()  # (C, H, W)
    return x


# -----------------------------
# Utilidades para encontrar pares onset/apex
# -----------------------------
PAIR_ONSET = re.compile(r"^(?P<id>.+?)_onset\.[^.]+$")
PAIR_APEX  = re.compile(r"^(?P<id>.+?)_apex\.[^.]+$")


def collect_pairs_in_dir(dir_path: str) -> list:
    """Devuelve lista de tuplas (onset_path, apex_path) emparejadas por prefijo <id>.
    Busca archivos *_onset.* y *_apex.* dentro de dir_path (no recursivo).
    """
    files = os.listdir(dir_path)
    onset_map = {}
    apex_map = {}
    for fname in files:
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        m_on = PAIR_ONSET.match(fname)
        m_ap = PAIR_APEX.match(fname)
        if m_on:
            onset_map[m_on.group('id')] = fpath
        elif m_ap:
            apex_map[m_ap.group('id')] = fpath
    ids = sorted(set(onset_map.keys()) & set(apex_map.keys()))
    pairs = [(onset_map[i], apex_map[i]) for i in ids]
    return pairs


# -----------------------------
# Dataset de pares por clase (ImageFolder-like)
# -----------------------------
class PairsByClassDataset(Dataset):
    def __init__(self, root: str, img_size: int = 224, class_names=("micro", "nomicro")):
        super().__init__()
        self.root = root
        self.img_size = img_size
        # Mapear clases a índices
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []  # lista de (onset_path, apex_path, label)

        for cls in class_names:
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                raise FileNotFoundError(f"No existe carpeta de clase: {cdir}")
            pairs = collect_pairs_in_dir(cdir)
            self.samples.extend([(on, ap, self.class_to_idx[cls]) for on, ap in pairs])

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron pares onset/apex en {root}. Verifica nombres *_onset.* y *_apex.*")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        onset_path, apex_path, y = self.samples[idx]
        onset_rgb = read_rgb(onset_path)
        apex_rgb  = read_rgb(apex_path)
        onset_arr = preprocess_cv(onset_rgb, self.img_size)
        apex_arr  = preprocess_cv(apex_rgb,  self.img_size)
        onset_t = to_tensor_nchw(onset_arr)  # (C,H,W)
        apex_t  = to_tensor_nchw(apex_arr)
        return onset_t, apex_t, y


# -----------------------------
# Carga de Micron‑BERT desde checkpoint
# -----------------------------

def getattr_fallback(o, k, default=None):
    try:
        return getattr(o, k)
    except Exception:
        return o.get(k, default) if isinstance(o, dict) else default


def load_micronbert_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in ckpt or "model" not in ckpt:
        raise ValueError("El checkpoint debe contener claves 'args' y 'model'.")

    args = ckpt["args"]
    model_name = getattr_fallback(args, "model_name", None)
    if model_name is None:
        raise ValueError("'model_name' no encontrado en args del checkpoint.")

    import importlib
    mae_dict = importlib.import_module("models.mae")

    model = mae_dict.__dict__[model_name](
        has_decoder=getattr_fallback(args, "has_decoder", False),
        aux_cls=getattr_fallback(args, "aux_cls", False),
        img_size=getattr_fallback(args, "img_size", 224),
        att_loss=getattr_fallback(args, "att_loss", False),
        diag_att=getattr_fallback(args, "diag_att", False),
        enable_dino=getattr_fallback(args, "enable_dino", False),
        out_dim=getattr_fallback(args, "out_dim", 0),
        local_crops_number=getattr_fallback(args, "local_crops_number", 0),
        warmup_teacher_temp=getattr_fallback(args, "warmup_teacher_temp", 0.04),
        teacher_temp=getattr_fallback(args, "teacher_temp", 0.04),
        warmup_teacher_temp_epochs=getattr_fallback(args, "warmup_teacher_temp_epochs", 0),
        epochs=getattr_fallback(args, "epochs", 0),
    )

    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Faltan claves al cargar state_dict:", missing)
    if unexpected:
        print("[WARN] Claves inesperadas en state_dict:", unexpected)

    model.to(device)
    return model, args


# -----------------------------
# Modelo binario con dos entradas (onset, apex)
# -----------------------------
class PairwiseBinaryHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.mlp(x)


class MicronBERTPairsBinary(nn.Module):
    """Dos imágenes (onset, apex) pasan por el mismo encoder.
    Se extraen embeddings, se combinan (cat + diff + |diff|) y se clasifican.
    """
    def __init__(self, base_model: nn.Module, feature_dim: int = None, dropout: float = 0.1, hidden: int = 512):
        super().__init__()
        self.base = base_model
        self.feature_dim = feature_dim or self._infer_feat_dim()
        # combinación: [on, ap, ap-on, |ap-on|] => 4 * feat_dim
        combined_dim = 4 * self.feature_dim
        self.head = PairwiseBinaryHead(combined_dim, hidden=hidden, dropout=dropout)

    def _extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base, "extract_features"):
            z = self.base.extract_features(x)
        else:
            z = self.base(x)
        # Pooling si la salida es secuencia (B, N, C) o mapa (B, C, H, W)
        if z.ndim == 3:
            z = z.mean(dim=1)
        elif z.ndim == 4:
            z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        return z

    def _infer_feat_dim(self) -> int:
        self.base.eval()
        with torch.no_grad():
            device = next(self.base.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            z = self._extract_feats(dummy)
            return int(z.shape[-1])

    def forward(self, onset: torch.Tensor, apex: torch.Tensor) -> torch.Tensor:
        z_on = self._extract_feats(onset)
        z_ap = self._extract_feats(apex)
        diff = z_ap - z_on
        adiff = torch.abs(diff)
        feats = torch.cat([z_on, z_ap, diff, adiff], dim=1)
        logits = self.head(feats)
        return logits


# -----------------------------
# Entrenamiento / evaluación
# -----------------------------
@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_prec: float
    val_rec: float
    val_f1: float
    val_auc: float


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def eval_full(model, loader, device) -> dict:
    model.eval()
    ys = []
    probs = []
    loss_total = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for on, ap, y in loader:
            on = on.to(device, non_blocking=True)
            ap = ap.to(device, non_blocking=True)
            y = torch.tensor(y, dtype=torch.long, device=device)
            logits = model(on, ap)
            loss = ce(logits, y)
            p = torch.softmax(logits, dim=1)[:, 1]  # prob clase 1 (micro)
            probs.append(p.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            loss_total += float(loss.item()) * y.size(0)
            n += y.size(0)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = (y_pred == y_true).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'loss': loss_total / max(n, 1),
        'acc': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
    }


def run_train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets
    train_ds = PairsByClassDataset(args.train_dir, img_size=args.img_size, class_names=(args.pos_class, args.neg_class))
    val_ds   = PairsByClassDataset(args.val_dir,   img_size=args.img_size, class_names=(args.pos_class, args.neg_class))
    test_ds  = PairsByClassDataset(args.test_dir,  img_size=args.img_size, class_names=(args.pos_class, args.neg_class)) if args.test_dir else None

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True) if test_ds else None

    # Base model y wrapper
    base, base_args = load_micronbert_from_checkpoint(args.checkpoint, device)
    model = MicronBERTPairsBinary(base, feature_dim=None, dropout=args.dropout, hidden=args.hidden)
    model.to(device)

    if args.freeze_base:
        for p in model.base.parameters():
            p.requires_grad = False
        print("[INFO] Base congelada — sólo se entrena la cabeza binaria.")

    # Optimizador + scheduler + AMP
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    last_path = os.path.join(args.out_dir, 'last_model.pth')
    log_csv = os.path.join(args.out_dir, 'log.csv')

    # CSV logging header
    if not os.path.exists(log_csv):
        with open(log_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","val_precision","val_recall","val_f1","val_auc","lr"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_correct = 0
        count = 0

        for on, ap, y in train_loader:
            on = on.to(device, non_blocking=True)
            ap = ap.to(device, non_blocking=True)
            y = torch.tensor(y, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(on, ap)
                    loss = ce(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(on, ap)
                loss = ce(logits, y)
                loss.backward()
                optimizer.step()

            run_loss += float(loss.item()) * y.size(0)
            run_correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.size(0)

        train_loss = run_loss / max(count, 1)
        train_acc  = run_correct / max(count, 1)

        # Validación completa con métricas
        metrics_val = eval_full(model, val_loader, device)

        # Scheduler con métrica val_acc
        scheduler.step(metrics_val['acc'])

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {metrics_val['loss']:.4f} acc {metrics_val['acc']:.4f} f1 {metrics_val['f1']:.4f} auc {metrics_val['auc']:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.1f}s")

        # Guardado de 'last'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'base_checkpoint': args.checkpoint,
            'img_size': args.img_size,
            'class_names': [args.pos_class, args.neg_class],
        }, last_path)

        # Guardado de 'best' por val_acc
        if metrics_val['acc'] >= best_acc:
            best_acc = metrics_val['acc']
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'base_checkpoint': args.checkpoint,
                'img_size': args.img_size,
                'class_names': [args.pos_class, args.neg_class],
            }, best_path)
            print(f"[+] Guardado BEST -> {best_path} (val_acc={best_acc:.4f})")

        # Log CSV
        with open(log_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{metrics_val['loss']:.6f}", f"{metrics_val['acc']:.6f}",
                        f"{metrics_val['precision']:.6f}", f"{metrics_val['recall']:.6f}", f"{metrics_val['f1']:.6f}", f"{metrics_val['auc']:.6f}", f"{lr_now:.6e}"])

    # Evaluación final en test
    if test_loader is not None:
        # cargar best
        if os.path.isfile(best_path):
            ck = torch.load(best_path, map_location='cpu')
            model.load_state_dict(ck['state_dict'])
        mte = eval_full(model, test_loader, device)
        print(f"TEST | loss {mte['loss']:.4f} acc {mte['acc']:.4f} prec {mte['precision']:.4f} rec {mte['recall']:.4f} f1 {mte['f1']:.4f} auc {mte['auc']:.4f}")


# -----------------------------
# Inferencia sobre carpeta con pares (sin clases)
# -----------------------------

def load_trained_model(best_path: str, device: torch.device):
    ck = torch.load(best_path, map_location='cpu')
    base_ckpt = ck.get('base_checkpoint', None)
    if base_ckpt is None:
        raise ValueError("El checkpoint entrenado no contiene 'base_checkpoint'.")
    base, _ = load_micronbert_from_checkpoint(base_ckpt, device)
    model = MicronBERTPairsBinary(base)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    model.eval()
    img_size = ck.get('img_size', 224)
    class_names = ck.get('class_names', ["micro", "nomicro"])  # índice 0 -> micro, 1 -> nomicro
    return model, img_size, class_names


def infer_dir_pairs(input_dir: str, model: nn.Module, img_size: int, device: torch.device):
    pairs = collect_pairs_in_dir(input_dir)
    results = []
    with torch.no_grad():
        for on_path, ap_path in pairs:
            on = to_tensor_nchw(preprocess_cv(read_rgb(on_path), img_size)).unsqueeze(0).to(device)
            ap = to_tensor_nchw(preprocess_cv(read_rgb(ap_path),  img_size)).unsqueeze(0).to(device)
            logits = model(on, ap)
            prob_micro = torch.softmax(logits, dim=1)[:, 1].item()
            pred = int(prob_micro >= 0.5)
            results.append({
                'id': os.path.basename(on_path).replace('_onset', '').rsplit('.', 1)[0],
                'onset': on_path,
                'apex': ap_path,
                'prob_micro': prob_micro,
                'pred_label': pred,
            })
    return results


def run_infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, img_size, class_names = load_trained_model(args.checkpoint_best, device)
    results = infer_dir_pairs(args.input_dir, model, img_size, device)

    # Guardar CSV
    out_csv = args.output_csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id','onset','apex','prob_micro','pred_label'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Guardado CSV de predicciones en: {out_csv}")


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Micron‑BERT pairs (train/infer)")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Entrenamiento con pares onset/apex')
    pt.add_argument('--train_dir', type=str, required=True)
    pt.add_argument('--val_dir', type=str, required=True)
    pt.add_argument('--test_dir', type=str, default=None)
    pt.add_argument('--checkpoint', type=str, required=True, help='Checkpoint base de Micron‑BERT (.pth)')
    pt.add_argument('--out_dir', type=str, default='runs/pairs_binary')

    pt.add_argument('--pos_class', type=str, default='micro')
    pt.add_argument('--neg_class', type=str, default='nomicro')

    pt.add_argument('--img_size', type=int, default=224)
    pt.add_argument('--epochs', type=int, default=10)
    pt.add_argument('--bs', type=int, default=16)
    pt.add_argument('--lr', type=float, default=1e-4)
    pt.add_argument('--weight_decay', type=float, default=1e-4)
    pt.add_argument('--dropout', type=float, default=0.1)
    pt.add_argument('--hidden', type=int, default=512)
    pt.add_argument('--freeze_base', action='store_true')
    pt.add_argument('--amp', action='store_true')
    pt.add_argument('--num_workers', type=int, default=4)
    pt.add_argument('--seed', type=int, default=42)

    # Infer
    pi = sub.add_parser('infer', help='Inferencia sobre carpeta con pares (sin clases)')
    pi.add_argument('--input_dir', type=str, required=True)
    pi.add_argument('--checkpoint_best', type=str, required=True, help='Checkpoint BEST guardado por train')
    pi.add_argument('--output_csv', type=str, default='preds.csv')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == 'train':
        run_train(args)
    elif args.cmd == 'infer':
        run_infer(args)
    else:
        raise ValueError('Comando no soportado')


if __name__ == '__main__':
    main()
