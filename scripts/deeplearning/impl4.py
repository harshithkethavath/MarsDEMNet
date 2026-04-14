"""
scripts/deeplearning/impl4.py

Implementation 4 — Encoder Depth Ablation (3 / 4 / 5 blocks).
Reuses the impl3 best_model.pt for the 4-block variant if it exists.

Usage (from repo root):
    python scripts/deeplearning/impl4.py [--blocks 3,4,5] [--weights elev_emph] [...]
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
    p.add_argument("--out_dir",     default="outputs/deeplearning/impl4")
    p.add_argument("--impl3_dir",   default="outputs/deeplearning/impl3")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--cache",       action="store_true", help="Load dataset into RAM before training")
    p.add_argument("--weights",     choices=["uniform", "elev_emph"], default="elev_emph")
    p.add_argument("--blocks",      default="3,4,5")
    p.add_argument("--skip_reuse",  action="store_true")
    return p.parse_args()


def make_fns(weights):
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
            "val_elev_rmse":   em["rmse"],
            "val_elev_mae":    em["mae"],
            "val_elev_delta1": em["delta1"],
            "val_slope_mae":   sm["mae"],
            "val_rough_mae":   rm["mae"],
        }

    return loss_fn, metric_fn


def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = WEIGHT_CONFIGS[args.weights]
    blocks  = [int(b.strip()) for b in args.blocks.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[impl4] device: {device} | blocks: {blocks} | weights: {weights}")

    train_ds = MCTEDDataset(args.train_dir, augment=True,  compute_auxiliary=True, cache=args.cache)
    val_ds   = MCTEDDataset(args.val_dir,   augment=False, compute_auxiliary=True, cache=args.cache)
    print(f"[impl4] train: {len(train_ds):,}  val: {len(val_ds):,}")

    loss_fn, metric_fn = make_fns(weights)
    all_histories = {}
    all_meta      = {}

    for nb in blocks:
        tag = f"{nb}block"

        # Reuse impl3 checkpoint for 4-block if available
        if nb == 4 and not args.skip_reuse:
            impl3_ckpt = os.path.join(args.impl3_dir, "best_model.pt")
            if os.path.exists(impl3_ckpt):
                shutil.copy2(impl3_ckpt, os.path.join(args.out_dir, f"best_model_{tag}.pt"))
                print(f"[impl4|{tag}] reused impl3 checkpoint — skipping training")
                all_meta[tag] = {"num_blocks": 4, "source": "reused_from_impl3"}
                # pull history from impl3 results if present
                impl3_res = os.path.join(args.impl3_dir, "results.json")
                if os.path.exists(impl3_res):
                    import json
                    with open(impl3_res) as f:
                        ir = json.load(f)
                    best_cfg = ir.get("best_config", "elev_emph")
                    all_histories[tag] = ir.get("histories", {}).get(best_cfg, {})
                continue

        model    = UNet(in_channels=1, out_channels=3, num_blocks=nb, base_ch=32).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[impl4|{tag}] parameters: {n_params:,}")

        history, best_rmse, best_epoch = run(
            model, train_ds, val_ds, loss_fn, metric_fn,
            primary_metric="val_elev_rmse", args=args, run_tag=tag,
        )
        all_histories[tag] = history
        all_meta[tag] = {
            "num_blocks":     nb,
            "n_params":       n_params,
            "best_epoch":     best_epoch,
            "best_elev_rmse": best_rmse,
        }

    # Summary
    print("\n[impl4] ── Ablation Summary ──")
    print(f"{'Variant':<10} {'Params':>12} {'Best RMSE':>12}")
    for tag, m in all_meta.items():
        n    = m.get("n_params", "N/A")
        rmse = m.get("best_elev_rmse", "N/A")
        print(f"{tag:<10} {str(n) if not isinstance(n,int) else f'{n:,}':>12} "
              f"{rmse if not isinstance(rmse,float) else f'{rmse:.4f}':>12}")

    save_results(
        os.path.join(args.out_dir, "results.json"),
        {
            "implementation": "impl4_encoder_depth_ablation",
            "config":         vars(args),
            "loss_weights":   weights,
            "variants":       all_meta,
            "histories":      all_histories,
        },
    )

    save_curves(
        all_histories,
        args.out_dir,
        filename="ablation_curves.png",
        title="Implementation 4 — Encoder Depth Ablation (3 / 4 / 5 blocks)",
        series=[
            ("train_loss",      0, 0, "Loss",     "-"),
            ("val_loss",        0, 0, "Loss",     "--"),
            ("val_elev_rmse",   0, 1, "RMSE (m)", "-"),
            ("val_elev_delta1", 0, 2, "Delta-1",  "-"),
            ("val_elev_mae",    1, 0, "MAE (m)",  "-"),
            ("val_slope_mae",   1, 1, "MAE (°)",  "-"),
            ("val_rough_mae",   1, 2, "MAE (m)",  "-"),
        ],
    )


if __name__ == "__main__":
    main()