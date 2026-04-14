"""
scripts/deeplearning/impl2.py

Implementation 2 — Single-Output U-Net (elevation only, 4-block encoder).

Usage (from repo root):
    python scripts/deeplearning/impl2.py [--train_dir ...] [--val_dir ...] [--epochs 50]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from data.dataset import MCTEDDataset
from scripts.deeplearning.unet import UNet, masked_mae, compute_metrics
from scripts.deeplearning.trainer import run, save_results, save_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",   default="data/train")
    p.add_argument("--val_dir",     default="data/val")
    p.add_argument("--out_dir",     default="outputs/deeplearning/impl2")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--cache",       action="store_true", help="Load dataset into RAM before training")
    return p.parse_args()


def loss_fn(pred, batch, device):
    return masked_mae(pred, batch["dem"].to(device), batch["valid"].to(device))


def metric_fn(pred, batch, device):
    m = compute_metrics(pred, batch["dem"].to(device), batch["valid"].to(device))
    return {"val_mae": m["mae"], "val_rmse": m["rmse"], "val_delta1": m["delta1"]}


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[impl2] device: {device} | args: {vars(args)}")

    train_ds = MCTEDDataset(args.train_dir, augment=True,  compute_auxiliary=False, cache=args.cache)
    val_ds   = MCTEDDataset(args.val_dir,   augment=False, compute_auxiliary=False, cache=args.cache)
    print(f"[impl2] train: {len(train_ds):,}  val: {len(val_ds):,}")

    model    = UNet(in_channels=1, out_channels=1, num_blocks=4, base_ch=32).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[impl2] parameters: {n_params:,}")

    history, best_rmse, best_epoch = run(
        model, train_ds, val_ds, loss_fn, metric_fn,
        primary_metric="val_rmse", args=args,
    )

    save_results(
        os.path.join(args.out_dir, "results.json"),
        {
            "implementation": "impl2_single_output_unet",
            "config":         vars(args),
            "model":          {"num_blocks": 4, "out_channels": 1, "n_params": n_params},
            "best_epoch":     best_epoch,
            "best_val_rmse":  best_rmse,
            "history":        history,
        },
    )

    save_curves(
        {"impl2": history},
        args.out_dir,
        filename="training_curves.png",
        title="Implementation 2 — Single-Output U-Net",
        series=[
            ("train_loss", 0, 0, "MAE (m)", "-"),
            ("val_loss",   0, 0, "MAE (m)", "--"),
            ("val_rmse",   0, 1, "RMSE (m)", "-"),
            ("val_delta1", 0, 2, "Delta-1", "-"),
        ],
    )


if __name__ == "__main__":
    main()