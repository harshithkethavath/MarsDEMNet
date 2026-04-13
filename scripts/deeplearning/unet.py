"""
scripts/deeplearning/unet.py

Shared U-Net backbone for Implementations 2, 3, and 4.

All three implementations share this single file.  The two constructor
arguments that differ across implementations are:

    out_channels  — 1 for Impl 2 (elevation only),
                    3 for Impls 3 and 4 (elevation + slope + roughness)

    num_blocks    — encoder depth:
                    4 for Impl 2 and 3 (reference depth)
                    3, 4, or 5 for Impl 4 (ablation)

Channel schedule follows the doubling rule from the first block onward:
    num_blocks=3 → [32, 64, 128]          bottleneck: 256
    num_blocks=4 → [32, 64, 128, 256]     bottleneck: 512
    num_blocks=5 → [32, 64, 128, 256, 512] bottleneck: 1024

The decoder mirrors the encoder exactly: one upsampling stage per encoder
block, with a lateral skip connection from the corresponding encoder level.

Input:  (B, 1, 518, 518)   — single-channel CTX optical patch
Output: (B, out_channels, 518, 518)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU layers.

    Using two convolutions per block (not one) is the standard U-Net
    convention.  The first conv integrates spatial context; the second
    refines the feature map within the same receptive field.  Both use
    3×3 kernels with padding=1 so spatial size is preserved within the block
    (downsampling happens outside, via MaxPool in the encoder).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    ConvBlock followed by MaxPool2d(2).

    Returns both the pre-pool feature map (for the skip connection) and
    the pooled feature map (passed to the next encoder level).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv  = ConvBlock(in_ch, out_ch)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)   # pre-pool — saved for skip connection
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    Bilinear upsample → concatenate skip → ConvBlock.

    The skip connection doubles the channel count before the ConvBlock,
    which is why in_ch here refers to the channels coming up from below
    and skip_ch to the channels arriving laterally.  The ConvBlock fuses
    both into out_ch channels.

    Bilinear upsampling is preferred over transposed convolution because
    it avoids the checkerboard artefacts that transposed conv produces
    when the input spatial size is not a clean power of two (518 is not).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Configurable U-Net for dense terrain prediction.

    Args:
        in_channels  (int): Input image channels.  Always 1 for CTX patches.
        out_channels (int): Output map channels.
                            1  → Implementation 2 (elevation only)
                            3  → Implementations 3 and 4 (elev + slope + roughness)
        num_blocks   (int): Number of encoder/decoder stages (3, 4, or 5).
                            Controls model depth and parameter count.
                            Impl 2 and 3 use 4.  Impl 4 sweeps 3, 4, 5.
        base_ch      (int): Channel count at the first encoder block.
                            Doubles at every subsequent block.  Default 32.

    Output head(s):
        When out_channels == 1, a single Conv2d(base_ch, 1, 1) is used.
        When out_channels == 3, three separate Conv2d(base_ch, 1, 1) heads
        are used — one per terrain layer — so each head can specialize
        independently.  The outputs are concatenated along dim=1 before return.
    """

    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        num_blocks:   int = 4,
        base_ch:      int = 32,
    ):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("num_blocks must be >= 2")

        self.out_channels = out_channels

        # Channel schedule: [base_ch, base_ch*2, base_ch*4, ...]
        enc_chs = [base_ch * (2 ** i) for i in range(num_blocks)]
        bot_ch  = enc_chs[-1] * 2   # bottleneck doubles the last encoder stage

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for ch_out in enc_chs:
            self.encoders.append(EncoderBlock(ch_in, ch_out))
            ch_in = ch_out

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(enc_chs[-1], bot_ch)

        # ---- Decoder ----
        # Decoder stages in reverse order.
        # At each stage, input comes from the stage below (up_ch channels)
        # and the lateral skip provides skip_ch channels.
        self.decoders = nn.ModuleList()
        up_ch = bot_ch
        for skip_ch in reversed(enc_chs):
            out_ch = skip_ch          # decoder output matches encoder output at same depth
            self.decoders.append(DecoderBlock(up_ch, skip_ch, out_ch))
            up_ch = out_ch

        # ---- Output head(s) ----
        # 1×1 conv collapses channel depth to prediction channels with no
        # spatial mixing — all spatial structure comes from the decoder.
        if out_channels == 1:
            self.head = nn.Conv2d(base_ch, 1, kernel_size=1)
        else:
            # Separate head per output channel — allows each to learn its own
            # output scale without weight sharing at the final projection.
            self.heads = nn.ModuleList(
                [nn.Conv2d(base_ch, 1, kernel_size=1) for _ in range(out_channels)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Encoder pass: collect skips ----
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # ---- Bottleneck ----
        x = self.bottleneck(x)

        # ---- Decoder pass: consume skips in reverse ----
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # ---- Output ----
        if self.out_channels == 1:
            return self.head(x)                                   # (B, 1, H, W)
        else:
            return torch.cat([h(x) for h in self.heads], dim=1)  # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mae(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error computed only over valid pixels.

    Args:
        pred   : (B, 1, H, W)  — model output for one terrain layer
        target : (B, 1, H, W)  — ground-truth for the same layer
        valid  : (B, H, W) bool — True where ground truth is reliable

    The valid mask is unsqueezed to (B, 1, H, W) so it broadcasts against
    the channel dimension of pred/target without requiring squeeze/unsqueeze
    gymnastics at the call site.

    Returns a scalar tensor (mean over all valid pixels in the batch).
    Returns zero if no valid pixel exists in the batch (degenerate case).
    """
    mask = valid.unsqueeze(1)                    # (B, 1, H, W)
    diff = (pred - target).abs()
    num_valid = mask.sum().clamp(min=1)          # avoid divide-by-zero
    return (diff * mask).sum() / num_valid


def multi_task_loss(
    pred:    torch.Tensor,
    dem:     torch.Tensor,
    slope:   torch.Tensor,
    roughness: torch.Tensor,
    valid:   torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0),
) -> tuple:
    """
    Weighted sum of per-channel masked MAE losses for the multi-output model.

    Args:
        pred      : (B, 3, H, W) — model output [elev, slope, roughness]
        dem       : (B, 1, H, W) — elevation target
        slope     : (B, 1, H, W) — slope target (degrees)
        roughness : (B, 1, H, W) — roughness target (meters)
        valid     : (B, H, W) bool
        weights   : (λ_elev, λ_slope, λ_roughness)

    Returns:
        total_loss : scalar tensor — backpropagate this
        (loss_elev, loss_slope, loss_roughness) : individual losses for logging
    """
    l_elev  = masked_mae(pred[:, 0:1], dem,       valid)
    l_slope = masked_mae(pred[:, 1:2], slope,     valid)
    l_rough = masked_mae(pred[:, 2:3], roughness, valid)

    w_e, w_s, w_r = weights
    total = w_e * l_elev + w_s * l_slope + w_r * l_rough
    return total, (l_elev, l_slope, l_rough)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """
    Compute MAE, RMSE, and delta-1 accuracy over valid pixels.

    All three are the canonical metrics from the proposal and the broader
    monocular depth estimation literature.

    delta-1: fraction of valid pixels where max(pred/gt, gt/pred) < 1.25.
    Computed on relative elevation values — both pred and target are already
    mean-subtracted.  Zero-valued ground truth pixels are excluded from the
    ratio computation to avoid division by zero; they are rare but present
    at flat-fill regions.

    Args:
        pred   : (B, 1, H, W)
        target : (B, 1, H, W)
        valid  : (B, H, W) bool

    Returns dict with keys: mae, rmse, delta1  (all Python floats)
    """
    mask = valid.unsqueeze(1)
    p = pred[mask]
    t = target[mask]

    mae  = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()

    # delta-1: exclude pixels where gt is exactly zero to avoid inf ratios
    nonzero = t.abs() > 1e-6
    if nonzero.sum() > 0:
        ratio   = torch.max(p[nonzero] / t[nonzero], t[nonzero] / p[nonzero])
        delta1  = (ratio < 1.25).float().mean().item()
    else:
        delta1 = 0.0

    return {"mae": mae, "rmse": rmse, "delta1": delta1}