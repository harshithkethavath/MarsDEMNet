"""
scripts/deeplearning/trainer.py

Generic training loop shared by all three deep learning implementations.
Each impl script calls run() with its own model, loss function, and config.
Nothing in here is implementation-specific.
"""

import json
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scripts.deeplearning.unet import compute_metrics


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run(
    model,
    train_ds,
    val_ds,
    loss_fn,          # callable(pred, batch, device) -> scalar loss tensor
    metric_fn,        # callable(pred, batch, device) -> dict of loggable floats
    primary_metric,   # str key in metric_fn output used for early stopping (minimised)
    args,             # namespace with epochs, batch_size, lr, patience, num_workers, out_dir
    run_tag="",       # short label printed in logs, used in checkpoint filenames
):
    """
    Train model for up to args.epochs epochs with early stopping on primary_metric.

    Returns:
        history (dict)       — lists of per-epoch scalars
        best_metric (float)  — best value of primary_metric seen
        best_epoch (int)     — epoch at which best_metric was achieved
    """
    device = next(model.parameters()).device
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history        = {"train_loss": [], "val_loss": [], "lr": []}
    best_metric    = float("inf")
    patience_count = 0
    best_epoch     = -1
    tag            = f"[{run_tag}] " if run_tag else ""

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["optical"].to(device))
            loss = loss_fn(pred, batch, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ---- validate ----
        model.eval()
        val_loss   = 0.0
        val_metrics_accum = {}
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["optical"].to(device))
                val_loss += loss_fn(pred, batch, device).item()
                m = metric_fn(pred, batch, device)
                for k, v in m.items():
                    val_metrics_accum.setdefault(k, []).append(v)
        val_loss /= max(len(val_loader), 1)
        val_metrics = {k: sum(v) / len(v) for k, v in val_metrics_accum.items()}

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- log ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for k, v in val_metrics.items():
            history.setdefault(k, []).append(v)

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(
            f"{tag}epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  {metric_str}  "
            f"lr={current_lr:.2e}  {time.time()-t0:.1f}s"
        )

        # ---- checkpoint ----
        current = val_metrics[primary_metric]
        if current < best_metric:
            best_metric    = current
            best_epoch     = epoch
            patience_count = 0
            ckpt_name      = f"best_model_{run_tag}.pt" if run_tag else "best_model.pt"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": val_metrics},
                os.path.join(args.out_dir, ckpt_name),
            )
            print(f"{tag}  ↳ new best {primary_metric}: {best_metric:.4f}  (saved {ckpt_name})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"{tag}early stopping at epoch {epoch}")
                break

    # final weights
    final_name = f"final_model_{run_tag}.pt" if run_tag else "final_model.pt"
    torch.save({"epoch": epoch, "model_state": model.state_dict()},
               os.path.join(args.out_dir, final_name))

    return history, best_metric, best_epoch


# ---------------------------------------------------------------------------
# results.json helper
# ---------------------------------------------------------------------------

def save_results(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"results saved → {path}")


# ---------------------------------------------------------------------------
# Curve plotting
# ---------------------------------------------------------------------------

def save_curves(histories, out_dir, filename, title, series):
    """
    histories : dict  tag -> history dict
    series    : list of (history_key, axis_row, axis_col, y_label, line_style)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Count unique (row, col) pairs to size the figure
        positions = sorted({(r, c) for _, r, c, _, _ in series})
        n_rows    = max(r for r, c in positions) + 1
        n_cols    = max(c for r, c in positions) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[a] for a in axes]

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        color_map = {tag: colors[i % len(colors)] for i, tag in enumerate(histories)}

        for key, row, col, ylabel, ls in series:
            ax = axes[row][col]
            for tag, h in histories.items():
                if key not in h:
                    continue
                ep = range(1, len(h[key]) + 1)
                label = f"{tag}" if ls == "-" else f"{tag} (val)"
                ax.plot(ep, h[key], color=color_map[tag], linestyle=ls, label=label)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Set subplot titles from first series entry per position
        seen = set()
        for key, row, col, ylabel, ls in series:
            if (row, col) not in seen:
                axes[row][col].set_title(key.replace("_", " ").title())
                seen.add((row, col))

        plt.suptitle(title, fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"curves saved → {path}")
    except Exception as e:
        print(f"warning: could not save curves ({e})")