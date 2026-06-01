"""
srgan_ct.py
===========
Super-Resolution GAN for 1-channel 512x512 Medical CT Slices.

Strategy  : Self-Degradation  –  HR = original 512x512 slice
                                  LR = bicubic-downsampled 256x256 copy
Hardware  : NVIDIA RTX 6000 Ada (48 GB VRAM) + workstation CPU
Precision : bfloat16 autocast + TF32 (no GradScaler required)
            RTX 6000 Ada is Ada Lovelace and natively supports bfloat16.

Checkpoint Strategy
-------------------
• During training  : single combined  checkpoint_epochXXXX.pth
                     (G weights + D weights + optimizers + schedulers + epoch)
                     → safe atomic resume, impossible to get G/D out of sync
• End of training  : generator_final.pth   (generator state_dict only)
                     → lean file for inference / deployment / fine-tuning

VRAM notes
----------
• 48 GB VRAM allows batch_size=16 at 512×512 bfloat16 with headroom to spare.
• Generator uses 20 residual blocks (up from 12/16) for stronger representation.
• Gradient checkpointing is still used on the residual stack – it costs almost
  nothing on 48 GB and provides a safety net if you increase batch size further.
• base_channels=96 (up from 64) gives a wider generator for richer features.
• Discriminator channel ceiling raised to 1024 for more discriminative power.

Directory layout (as shown in project screenshot)
-------------------------------------------------
  ./Denoised_Normal/
      C001.npy … C020.npy
  srgan_ct.py                  ← this file, at the same level

Author    : (your name)
"""

# ─────────────────────────────────────────────────────────────
#  Standard Library
# ─────────────────────────────────────────────────────────────
import os
import glob
import time
import math

# ─────────────────────────────────────────────────────────────
#  Third-Party
# ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torchvision.models as tv_models
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
#  TASK 1 – Self-Degrading CT DataLoader
# ══════════════════════════════════════════════════════════════

class SelfDegradingCTDataset(Dataset):
    """
    Scans *data_dirs* for .npy volumes and serves (lr, hr) slice pairs.

    Each .npy file is expected to have shape  (D, H, W)  where
      D = number of axial slices
      H = W = 512   (single-channel CT intensity, already denoised)

    Memory strategy
    ---------------
    Volumes are opened with  np.load(..., mmap_mode='r')  so only the
    requested slice pages are pulled from disk.

    __getitem__ returns
    -------------------
    hr_tensor : FloatTensor [1, 512, 512]  – clamped to [0, 1]
    lr_tensor : FloatTensor [1, 256, 256]  – bicubic ↓2× of hr_tensor
    """

    def __init__(self, data_dirs, max_volumes_per_dir=None):
        super().__init__()
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs

        volume_paths = []
        for data_dir in data_dirs:
            pattern = os.path.join(data_dir, "*.npy")
            found = sorted(glob.glob(pattern))
            if max_volumes_per_dir is not None and len(found) > max_volumes_per_dir:
                found = found[:max_volumes_per_dir]
            print(f"[Dataset]   {data_dir} -> {len(found)} volumes")
            volume_paths.extend(found)

        if not volume_paths:
            raise FileNotFoundError(
                f"No .npy files found in any directory: {data_dirs}. "
                "Check your data paths."
            )

        self._mmaps: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []
        for path in volume_paths:
            vol = np.load(path, mmap_mode='r')
            vol_idx = len(self._mmaps)
            self._mmaps.append(vol)
            for s in range(vol.shape[0]):
                self.index.append((vol_idx, s))

        print(f"[Dataset] Total: {len(volume_paths)} volume(s) -> "
              f"{len(self.index):,} slices total.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vol_idx, slice_idx = self.index[idx]
        raw_slice = self._mmaps[vol_idx][slice_idx]

        hr_np = raw_slice.astype(np.float32)
        hr_tensor = torch.from_numpy(hr_np).unsqueeze(0)
        hr_tensor = hr_tensor.clamp(0.0, 1.0)

        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0),
            size=(256, 256),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0.0, 1.0)

        return lr_tensor, hr_tensor


# ══════════════════════════════════════════════════════════════
#  TASK 2 – Generator  (SRResNet + PixelShuffle ×2)
#           RTX 6000 Ada: 20 residual blocks, base_channels=96
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Classic SRGAN residual block:
      Conv(k=3) -> BN -> PReLU -> Conv(k=3) -> BN  + skip connection
    """

    def __init__(self, channels: int = 96):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling block (×2 scale factor).
    Conv(out_ch = in_ch × 4) -> PixelShuffle(2) -> PReLU
    """

    def __init__(self, in_channels: int = 96):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    SRResNet-based Generator – RTX 6000 Ada configuration.

    Input  : [B, 1, 256, 256]   (LR CT slice)
    Output : [B, 1, 512, 512]   (SR CT slice)

    Architecture summary
    --------------------
    1. Initial Conv (k=9, pad=4) -> PReLU            : feature extraction
    2. 20 × ResidualBlock (channels=96)              : deep feature learning
    3. Post-Res Conv (k=3) -> BN                      : feature aggregation
    4. Skip-add with initial features                : global residual
    5. UpsampleBlock (PixelShuffle ×2)               : 256→512
    6. Final Conv (k=9, pad=4) -> Sigmoid            : reconstruct 1-ch image
    """

    NUM_RES_BLOCKS = 20

    def __init__(self, base_channels: int = 96):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=9,
                      stride=1, padding=4),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels)
              for _ in range(self.NUM_RES_BLOCKS)]
        )

        self.post_res_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
        )

        self.upsample = UpsampleBlock(base_channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=9,
                      stride=1, padding=4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_init = self.initial_conv(x)
        feat_res = grad_checkpoint(
            self.res_blocks, feat_init, use_reentrant=False
        )
        feat_post = self.post_res_conv(feat_res)
        feat_skip = feat_post + feat_init
        feat_up = self.upsample(feat_skip)
        out = self.final_conv(feat_up)
        return out


# ══════════════════════════════════════════════════════════════
#  TASK 3 – Discriminator  (VGG-style, wider for RTX 6000 Ada)
# ══════════════════════════════════════════════════════════════

def _disc_conv_block(
    in_ch: int,
    out_ch: int,
    stride: int,
    use_bn: bool = True,
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=stride, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    """
    VGG-style Discriminator – RTX 6000 Ada configuration.

    Input  : [B, 1, 512, 512]
    Output : [B, 1]
    """

    def __init__(self):
        super().__init__()

        # fmt: off
        self.features = nn.Sequential(
            *_disc_conv_block(  1,  64, stride=1, use_bn=False),   # 512
            *_disc_conv_block( 64,  64, stride=2, use_bn=True),    # 256

            *_disc_conv_block( 64, 128, stride=1, use_bn=True),
            *_disc_conv_block(128, 128, stride=2, use_bn=True),    # 128

            *_disc_conv_block(128, 256, stride=1, use_bn=True),
            *_disc_conv_block(256, 256, stride=2, use_bn=True),    # 64

            *_disc_conv_block(256, 512, stride=1, use_bn=True),
            *_disc_conv_block(512, 512, stride=2, use_bn=True),    # 32

            *_disc_conv_block( 512, 1024, stride=1, use_bn=True),
            *_disc_conv_block(1024, 1024, stride=2, use_bn=True),  # 16
        )
        # fmt: on

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════
#  TASK 4 – Perceptual Loss  (VGG-19 feature matching)
# ══════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    Computes perceptual loss using a frozen VGG-19 (layers 0:36 = relu4_4).
    Single-channel CT images are replicated to 3 channels before VGG forward.
    Loss = MSE( VGG(fake_3ch), VGG(real_3ch) )
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:36]
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_3ch = fake.repeat(1, 3, 1, 1)
        real_3ch = real.repeat(1, 3, 1, 1)
        return F.mse_loss(
            self.feature_extractor(fake_3ch),
            self.feature_extractor(real_3ch),
        )


# ══════════════════════════════════════════════════════════════
#  Utility – visualisation & checkpoint helpers
# ══════════════════════════════════════════════════════════════

def save_visual(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    epoch: int,
    out_dir: str = "./srgan_outputs",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t[0, 0].detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["LR Input (256×256)", "SR Output (512×512)", "HR Target (512×512)"]
    images = [to_np(lr), to_np(sr), to_np(hr)]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visual] Saved -> {save_path}")


# ──────────────────────────────────────────────────────────────
#  Checkpoint helpers  –  Best-of-Both-Worlds strategy
#
#  During training  →  save_checkpoint()
#     Saves ONE combined file:  checkpoint_epochXXXX.pth
#     Contains: G weights, D weights, opt_G, opt_D, sched_G, sched_D, epoch
#     Purpose : safe atomic resume – G/D/optimizers always in sync
#
#  End of training  →  export_generator()
#     Saves ONE lean file:  generator_final.pth  (or generator_epochXXXX.pth)
#     Contains: generator state_dict only
#     Purpose : inference / deployment / fine-tuning (no D baggage)
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sched_G: torch.optim.lr_scheduler.LRScheduler,
    sched_D: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    out_dir: str = "./srgan_outputs",
) -> None:
    """
    Save a SINGLE combined training checkpoint.

    File name : checkpoint_epochXXXX.pth
    Contains  : generator + discriminator weights, both optimizers,
                both schedulers, and the current epoch number.

    Why single file?
    ----------------
    • Atomic – one write, impossible to end up with G from epoch 10 and D
      from epoch 5 if the process is killed mid-save.
    • Auto-resume via find_latest_checkpoint() just works with one glob.
    • Discriminator is only ~70 MB – no size reason to split during training.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch{epoch:04d}.pth")
    torch.save(
        {
            "epoch":                    epoch,
            "generator_state_dict":     generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_G_state_dict":         opt_G.state_dict(),
            "opt_D_state_dict":         opt_D.state_dict(),
            "sched_G_state_dict":       sched_G.state_dict(),
            "sched_D_state_dict":       sched_D.state_dict(),
        },
        ckpt_path,
    )
    print(f"  [Checkpoint] Saved  -> {ckpt_path}")


def export_generator(
    generator: nn.Module,
    epoch: int,
    out_dir: str = "./srgan_outputs",
    tag: str = "final",
) -> None:
    """
    Export generator weights ONLY – for inference / deployment.

    File name : generator_{tag}.pth        (e.g. generator_final.pth)
                generator_epoch{XXXX}.pth  when tag is an epoch number

    Why separate for inference?
    ---------------------------
    • No discriminator, optimizer, or scheduler state – much smaller to ship.
    • Load with:
          G = Generator(base_channels=96)
          G.load_state_dict(torch.load("generator_final.pth"))
          G.eval()
    • Safe to upload to cloud storage / share with collaborators without
      leaking optimizer internals.
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, f"generator_{tag}.pth")
    torch.save(generator.state_dict(), gen_path)
    print(f"  [Export]     Saved  -> {gen_path}  (generator weights only)")


def find_latest_checkpoint(out_dir: str) -> str | None:
    """Return the path of the highest-epoch checkpoint in *out_dir*, or None."""
    pattern = os.path.join(out_dir, "checkpoint_epoch*.pth")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_checkpoint(
    ckpt_path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sched_G: torch.optim.lr_scheduler.LRScheduler,
    sched_D: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    """
    Restore all training state from *ckpt_path*.
    Returns the epoch number stored in the checkpoint (training resumes
    from epoch + 1).
    """
    print(f"[Resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    opt_G.load_state_dict(ckpt["opt_G_state_dict"])
    opt_D.load_state_dict(ckpt["opt_D_state_dict"])
    sched_G.load_state_dict(ckpt["sched_G_state_dict"])
    sched_D.load_state_dict(ckpt["sched_D_state_dict"])

    resumed_epoch = ckpt["epoch"]
    print(f"[Resume] Restored epoch {resumed_epoch}. "
          f"Training will continue from epoch {resumed_epoch + 1}.")
    return resumed_epoch


# ══════════════════════════════════════════════════════════════
#  TASK 5 – RTX 6000 Ada Optimised Training Loop
# ══════════════════════════════════════════════════════════════

def train(
    data_dir: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    num_workers: int = 8,
    lr: float = 1e-4,
    save_every: int = 5,
    out_dir: str = "./srgan_outputs",
    max_volumes: int = None,
    resume_from: str | None = "auto",
) -> None:
    """
    Full SRGAN training loop optimised for the RTX 6000 Ada Generation.

    Checkpoint behaviour
    --------------------
    • Every `save_every` epochs:
        checkpoint_epochXXXX.pth   – full training state (resume-safe)
    • At the very end of training:
        generator_final.pth        – generator weights only (inference-ready)
        generator_epochXXXX.pth    – same weights, tagged with final epoch
                                     (useful if you stop early and want to
                                     know which epoch it came from)

    Parameters
    ----------
    resume_from : str or None
        • "auto"  – automatically find and load the latest checkpoint in
                    *out_dir* (default).  Does nothing if no checkpoint exists.
        • path    – explicit path to a .pth checkpoint file to resume from.
        • None    – always train from scratch, ignoring any existing checkpoints.
    """

    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dataset = SelfDegradingCTDataset(data_dir, max_volumes_per_dir=max_volumes)
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = (use_cuda and num_workers > 0),
        persistent_workers = (num_workers > 0),
        prefetch_factor    = (4 if num_workers > 0 else None),
        drop_last          = True,
    )

    G = Generator(base_channels=96).to(device)
    D = Discriminator().to(device)

    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    G.apply(_init_weights)
    D.apply(_init_weights)

    perceptual_criterion = PerceptualLoss().to(device)
    bce_criterion        = nn.BCEWithLogitsLoss()
    mse_criterion        = nn.MSELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))

    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.5)
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=50, gamma=0.5)

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch = 1

    if resume_from == "auto":
        resume_from = find_latest_checkpoint(out_dir)

    if resume_from is not None:
        start_epoch = load_checkpoint(
            resume_from, G, D, opt_G, opt_D, sched_G, sched_D, device
        ) + 1

    if start_epoch > num_epochs:
        print(f"[Train] Already completed {num_epochs} epochs. Nothing to do.")
        return

    W_ADV    = 0.001
    W_PERCEP = 0.006

    print(f"\n[Train] Starting training from epoch {start_epoch} to {num_epochs}.")
    print(f"  Batches/epoch : {len(loader)}")
    print(f"  Save interval : every {save_every} epoch(s)\n")

    for epoch in range(start_epoch, num_epochs + 1):

        G.train()
        D.train()

        running = {"D": 0.0, "G": 0.0, "mse": 0.0, "adv": 0.0, "percep": 0.0}

        pbar = tqdm(
            enumerate(loader), total=len(loader),
            desc=f"Epoch {epoch:03d}/{num_epochs}", ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for batch_idx, (lr_imgs, hr_imgs) in pbar:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            real_labels = torch.ones (batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device) + 0.1

            # ── (A) Train Discriminator ──────────────────────────
            opt_D.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                real_preds  = D(hr_imgs)
                d_loss_real = bce_criterion(real_preds, real_labels)

                with torch.no_grad():
                    fake_hr = G(lr_imgs)

                fake_preds  = D(fake_hr.detach())
                d_loss_fake = bce_criterion(fake_preds, fake_labels)
                d_loss      = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()

            # ── (B) Train Generator ──────────────────────────────
            opt_G.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                fake_hr      = G(lr_imgs)
                content_loss = mse_criterion(fake_hr, hr_imgs)

                fake_preds_for_G = D(fake_hr)
                adv_loss = bce_criterion(
                    fake_preds_for_G,
                    torch.ones(batch_size, 1, device=device),
                )

                percep_loss = perceptual_criterion(fake_hr, hr_imgs)
                g_loss = (content_loss
                          + W_ADV    * adv_loss
                          + W_PERCEP * percep_loss)

            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            running["D"]      += d_loss.item()
            running["G"]      += g_loss.item()
            running["mse"]    += content_loss.item()
            running["adv"]    += adv_loss.item()
            running["percep"] += percep_loss.item()

            n_done = batch_idx + 1
            pbar.set_postfix_str(
                f"D={running['D']/n_done:.5f} "
                f"G={running['G']/n_done:.5f} "
                f"MSE={running['mse']/n_done:.5f}"
            )

        sched_G.step()
        sched_D.step()

        n = len(loader)
        print(
            f"Epoch [{epoch:04d}/{num_epochs}]  "
            f"D={running['D']/n:.4f}  "
            f"G={running['G']/n:.4f}  "
            f"(MSE={running['mse']/n:.5f}  "
            f"Adv={running['adv']/n:.5f}  "
            f"Percep={running['percep']/n:.5f})  "
            f"lr_G={sched_G.get_last_lr()[0]:.2e}"
        )

        # ── Periodic save (every save_every epochs) ──────────
        if epoch % save_every == 0:
            print(f"\n  [Save] Epoch {epoch}: saving checkpoint and sample image...")
            G.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                    sample_sr = G(lr_imgs[:1])
            save_visual(lr_imgs, sample_sr, hr_imgs, epoch, out_dir)

            # Combined checkpoint for safe resume
            save_checkpoint(G, D, opt_G, opt_D, sched_G, sched_D, epoch, out_dir)

            G.train()
            print(f"  [Save] Done. Next save at epoch {epoch + save_every}.\n")

    # ── End-of-training: export lean generator weights ───────
    print("\n[Train] Training complete. Exporting generator weights for inference...")
    G.eval()
    export_generator(G, epoch=num_epochs, out_dir=out_dir, tag="final")
    export_generator(G, epoch=num_epochs, out_dir=out_dir,
                     tag=f"epoch{num_epochs:04d}")
    print(
        f"\n  Files in {out_dir}:\n"
        f"    checkpoint_epoch*.pth   – full training checkpoints (resume)\n"
        f"    generator_final.pth     – generator only (inference)\n"
        f"    generator_epoch{num_epochs:04d}.pth – same, tagged with epoch\n"
    )
    print("[Train] All done.")


# ══════════════════════════════════════════════════════════════
#  TASK 6 – Execution Block
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DATA_DIRS = [
        "./Denoised_Normal",  # 20 Normal CT volumes (C001–C020)
        "./Denoised_Covid"
    ]

    OUTPUT_DIR    = "./srgan_outputs"
    NUM_EPOCHS    = 100
    BATCH_SIZE    = 16
    NUM_WORKERS   = 8
    LEARNING_RATE = 1e-4
    SAVE_EVERY    = 5
    RESUME_FROM   = "auto"

    for d in DATA_DIRS:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Data directory not found: '{d}'\n"
                "Expected layout:\n"
                "  ./Denoised_Normal/C001.npy … C020.npy\n"
                "  srgan_ct.py\n"
                "Please verify the directory name matches the screenshot."
            )

    import torch as _torch
    _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if _device.type == "cuda":
        _device_str = (
            f"GPU  –  {_torch.cuda.get_device_name(0)} "
            f"({_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)"
        )
    else:
        _device_str = "CPU  (no CUDA GPU detected)"

    print("=" * 65)
    print("  CT Super-Resolution GAN  –  Self-Degradation Strategy")
    print("  Hardware  : NVIDIA RTX 6000 Ada Generation (48 GB VRAM)")
    print("  Precision : bfloat16 autocast + TF32")
    print("=" * 65)
    print(f"  Running on       : {_device_str}")
    for d in DATA_DIRS:
        print(f"  Data directory   : {d}")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Epochs           : {NUM_EPOCHS}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Workers          : {NUM_WORKERS}")
    print(f"  Learning rate    : {LEARNING_RATE}")
    print(f"  Save every       : {SAVE_EVERY} epochs")
    print(f"  Resume from      : {RESUME_FROM}")
    print(f"  Generator        : 20 res-blocks, base_channels=96")
    print(f"  Discriminator    : channels up to 1024, FC head 2048")
    print("=" * 65)
    print()
    print("  Checkpoint strategy:")
    print("    Training  ->  checkpoint_epochXXXX.pth  (G + D + optimizers)")
    print("    Inference ->  generator_final.pth        (G weights only)")
    print("=" * 65)

    train(
        data_dir    = DATA_DIRS,
        num_epochs  = NUM_EPOCHS,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        lr          = LEARNING_RATE,
        save_every  = SAVE_EVERY,
        out_dir     = OUTPUT_DIR,
        max_volumes = 20,
        resume_from = RESUME_FROM,
    )
