"""
scripts/deeplearning/impl3.py

Implementation 3 — Multi-Output U-Net (elevation + slope + roughness).
Trains uniform (1:1:1) and elev-emphasised (2:1:1) weight configs in sequence.

Usage (from repo root):
    python scripts/deeplearning/impl3.py [--weights both|uniform|elev_emph] [...]
"""

import argparse
import os
import shutil
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from data.dataset import MCTEDDataset
from scripts.deeplearning.unet import UNet, multi_task_loss, compute_metrics
from scripts.deeplearning.trainer import run, save_results, save_curves

WEIGHT_CONFIGS = {
    "uniform":   (1.0, 1.0, 1.0),
    "elev_emph": (2.0, 1.0, 1.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",   default="data/train")
    p.add_argument("--val_dir",     default="data/val")
    p.add_argument("--out_dir",     default="outputs/deeplearning/impl3")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--cache",       action="store_true", help="Load dataset into RAM before training")
    p.add_argument("--weights",     choices=["both", "uniform", "elev_emph"], default="both")
    return p.parse_args()


def make_fns(weights):
    """Return loss_fn and metric_fn closed over a specific weight tuple."""

    def loss_fn(pred, batch, device):
        loss, _ = multi_task_loss(
            pred,
            batch["dem"].to(device),
            batch["slope"].to(device),
            batch["roughness"].to(device),
            batch["valid"].to(device),
            weights,
        )
        return loss

    def metric_fn(pred, batch, device):
        valid = batch["valid"].to(device)
        em = compute_metrics(pred[:, 0:1], batch["dem"].to(device),       valid)
        sm = compute_metrics(pred[:, 1:2], batch["slope"].to(device),     valid)
        rm = compute_metrics(pred[:, 2:3], batch["roughness"].to(device), valid)
        return {
            "val_elev_rmse":  em["rmse"],
            "val_elev_mae":   em["mae"],
            "val_elev_delta1": em["delta1"],
            "val_slope_mae":  sm["mae"],
            "val_rough_mae":  rm["mae"],
        }

    return loss_fn, metric_fn


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[impl3] device: {device} | args: {vars(args)}")

    train_ds = MCTEDDataset(args.train_dir, augment=True,  compute_auxiliary=True, cache=args.cache)
    val_ds   = MCTEDDataset(args.val_dir,   augment=False, compute_auxiliary=True, cache=args.cache)
    print(f"[impl3] train: {len(train_ds):,}  val: {len(val_ds):,}")

    configs = ["uniform", "elev_emph"] if args.weights == "both" else [args.weights]

    all_histories  = {}
    best_overall   = float("inf")
    best_config    = None
    n_params       = None

    for cfg in configs:
        weights           = WEIGHT_CONFIGS[cfg]
        loss_fn, metric_fn = make_fns(weights)

        model    = UNet(in_channels=1, out_channels=3, num_blocks=4, base_ch=32).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[impl3|{cfg}] parameters: {n_params:,}  weights: {weights}")

        history, best_rmse, best_epoch = run(
            model, train_ds, val_ds, loss_fn, metric_fn,
            primary_metric="val_elev_rmse", args=args, run_tag=cfg,
        )
        all_histories[cfg] = history

        if best_rmse < best_overall:
            best_overall = best_rmse
            best_config  = cfg

    # Promote best config checkpoint → best_model.pt
    src = os.path.join(args.out_dir, f"best_model_{best_config}.pt")
    dst = os.path.join(args.out_dir, "best_model.pt")
    shutil.copy2(src, dst)
    print(f"[impl3] best config: {best_config}  (elev RMSE={best_overall:.4f}) → best_model.pt")

    save_results(
        os.path.join(args.out_dir, "results.json"),
        {
            "implementation": "impl3_multi_output_unet",
            "config":         vars(args),
            "model":          {"num_blocks": 4, "out_channels": 3, "n_params": n_params},
            "best_config":    best_config,
            "best_elev_rmse": best_overall,
            "histories":      all_histories,
        },
    )

    save_curves(
        all_histories,
        args.out_dir,
        filename="training_curves.png",
        title="Implementation 3 — Multi-Output U-Net",
        series=[
            ("train_loss",      0, 0, "Loss", "-"),
            ("val_loss",        0, 0, "Loss", "--"),
            ("val_elev_rmse",   0, 1, "RMSE (m)", "-"),
            ("val_elev_delta1", 0, 2, "Delta-1",  "-"),
            ("val_elev_mae",    1, 0, "MAE (m)",  "-"),
            ("val_slope_mae",   1, 1, "MAE (°)",  "-"),
            ("val_rough_mae",   1, 2, "MAE (m)",  "-"),
        ],
    )


if __name__ == "__main__":
    main()