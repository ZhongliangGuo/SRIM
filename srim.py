"""SRIM: Scale-Robust Image Immunization against malicious editing.

Protective perturbation for a full-resolution photo that stays effective at every downscaling factor an
instruction-guided editor may apply before encoding the image (paper: arXiv:2610.07464).

The perturbation maximizes the relative latent deviation of the editor family's VAE encoder. Anchor scales
cover the range of downscaling factors implied by the photo size, and each sign-gradient step is spent on
one anchor drawn from a distribution that tracks the scale where the protection is currently weakest
(group-DRO style). One downscaled copy is encoded per step, so the cost equals expectation over
transformation (EOT).
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

KERNELS = ("bicubic", "bilinear", "area")   # antialiased, differentiable resampling kernels drawn during optimization

# Editor family -> (Hugging Face repo, VAE subfolder). The three FLUX.2 editors (klein 4B/9B, dev) share one VAE.
EDITORS = {
    "flux2": ("black-forest-labs/FLUX.2-klein-4B", "vae"),
    "ip2p": ("timbrooks/instruct-pix2pix", "vae"),
}


# ----------------------------------------------------------------------------------------------------------------
# image IO
# ----------------------------------------------------------------------------------------------------------------
def load_image(path, multiple=16):
    """RGB image as a float tensor in [0,1], shape (1,3,H,W), center-cropped to a multiple of `multiple` pixels."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    w2, h2 = w - w % multiple, h - h % multiple
    left, top = (w - w2) // 2, (h - h2) // 2
    im = im.crop((left, top, left + w2, top + h2))
    return torch.from_numpy(np.array(im)).float().div(255).permute(2, 0, 1)[None]


def to_uint8(x):
    return (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def save_image(x, path):
    """Save as 8-bit PNG (lossless; the 8-bit rounding is part of the method)."""
    Image.fromarray(to_uint8(x)).save(path)


# ----------------------------------------------------------------------------------------------------------------
# resize model: the editor downscales by a factor s (pixel count divided by s^2) before encoding
# ----------------------------------------------------------------------------------------------------------------
def scale_range(h, w, max_mp=4.2e6, min_mp=0.25e6):
    """Downscaling factors the owner must cover: editors work between min_mp and max_mp pixels."""
    n = h * w
    return max(1.0, math.sqrt(n / max_mp)), max(1.0, math.sqrt(n / min_mp))


def anchors_for(h, w, ratio=1.4):
    """Geometric grid of anchor scales with ratio at most `ratio` over the range implied by the photo size."""
    lo, hi = scale_range(h, w)
    k = max(1, int(math.ceil(math.log(hi / lo) / math.log(ratio))))
    return tuple(lo * (hi / lo) ** (i / k) for i in range(k + 1))


def target_size(h, w, s, multiple=16):
    """Size after downscaling by s, rounded to a multiple of `multiple` (editors round their working size)."""
    h2 = max(multiple, int(round(h / s / multiple)) * multiple)
    w2 = max(multiple, int(round(w / s / multiple)) * multiple)
    return h2, w2


def resize(x, size, kernel="bicubic"):
    """Differentiable antialiased resampling to `size`; kernel in KERNELS or 'nearest'."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if kernel == "area":
        # integer box pre-filter, then antialiased bilinear (adaptive pooling fails for large ratios on CUDA)
        k = int(min(x.shape[-2] // size[0], x.shape[-1] // size[1]))
        if k >= 2:
            x = F.avg_pool2d(x, k)
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=size[0] < x.shape[-2])
    if kernel == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    return F.interpolate(x, size=size, mode=kernel, align_corners=False, antialias=size[0] < x.shape[-2])


def resize_pil(x, size, kernel="lanczos"):
    """Non-differentiable PIL resampling, what real pipelines do (used only for reporting)."""
    k = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "box": Image.BOX}[kernel]
    im = Image.fromarray(to_uint8(x)).resize((size[1], size[0]), k)
    return torch.from_numpy(np.array(im)).float().div(255).permute(2, 0, 1)[None].to(x.device)


# ----------------------------------------------------------------------------------------------------------------
# surrogate: the editor family's VAE encoder
# ----------------------------------------------------------------------------------------------------------------
class VAEEncoder(torch.nn.Module):
    """Input in [0,1], shape (B,3,H,W) -> latent mean (B,C,h,w) of the editor's VAE (gray-box access)."""

    def __init__(self, editor="flux2", device="cuda", dtype=torch.bfloat16, repo=None):
        """`repo` overrides the Hugging Face repo (any model of the family with the same VAE, e.g. FLUX.2-dev)."""
        super().__init__()
        default_repo, sub = EDITORS[editor]
        repo = repo or default_repo
        if editor == "flux2":
            from diffusers import AutoencoderKLFlux2 as cls
        else:
            from diffusers import AutoencoderKL as cls
        self.vae = cls.from_pretrained(repo, subfolder=sub, torch_dtype=dtype).to(device).eval()
        self.vae.requires_grad_(False)
        if hasattr(self.vae, "enable_gradient_checkpointing"):
            try:
                self.vae.enable_gradient_checkpointing()
            except Exception:
                pass
        self.dtype = dtype

    def forward(self, x):
        return self.vae.encode((x * 2 - 1).to(self.dtype)).latent_dist.mean.float()


def rel_dev(z_adv, z_clean):
    """Relative latent deviation D = ||z_adv - z_clean|| / ||z_clean|| per sample (the objective)."""
    d = (z_adv - z_clean).flatten(1).norm(dim=1)
    return d / z_clean.flatten(1).norm(dim=1).clamp_min(1e-8)


def deviation_at(enc, x, delta, s, kernel, cache):
    """D_s(delta): deviation of the latent the editor computes after downscaling by s with `kernel`."""
    size = target_size(*x.shape[-2:], s)
    key = (size, kernel)
    if key not in cache:
        with torch.no_grad():
            cache[key] = enc(resize(x, size, kernel))
    return rel_dev(enc(resize((x + delta).clamp(0, 1), size, kernel)), cache[key]).mean()


# ----------------------------------------------------------------------------------------------------------------
# SRIM
# ----------------------------------------------------------------------------------------------------------------
def srim(enc, x, eps=4 / 255, steps=200, tau=0.1, ratio=1.4, jitter=0.1, ema=0.3, anchors=None, log=None):
    """Return the perturbation delta (||delta||_inf <= eps) for the photo x in [0,1], shape (1,3,H,W).

    Each step draws one anchor (every anchor once at the start, then with probability ~ exp(-r_k / tau), where
    r_k is an exponential moving average of the deviation observed at anchor k), jitters it by a random factor
    in [e^-jitter, e^jitter], draws a resampling kernel, and takes a sign-gradient step of size eps/4 with cosine
    decay on the deviation measured on that single downscaled copy.
    """
    h, w = x.shape[-2:]
    anchors = tuple(anchors) if anchors is not None else anchors_for(h, w, ratio)
    alpha = eps / 4
    cache = {}
    delta = torch.empty_like(x).uniform_(-eps, eps)
    r = {s: 0.0 for s in anchors}
    seen = {s: False for s in anchors}
    for t in range(steps):
        unseen = [s for s in anchors if not seen[s]]
        if unseen:
            s = unseen[0]
        else:
            p = torch.tensor([-r[a] / tau for a in anchors]).softmax(0)
            s = anchors[torch.multinomial(p, 1).item()]
        ss = max(1.0, s * math.exp(random.uniform(-jitter, jitter))) if s > 1 else s
        delta.requires_grad_(True)
        v = deviation_at(enc, x, delta, ss, random.choice(KERNELS), cache)
        r[s] = v.item() if not seen[s] else (1 - ema) * r[s] + ema * v.item()
        seen[s] = True
        g, = torch.autograd.grad(v, delta)
        a = alpha * 0.5 * (1 + math.cos(math.pi * t / steps)) + 1e-4
        delta = (delta.detach() + a * g.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
        if log and (t % 20 == 0 or t == steps - 1):
            log(f"step {t:4d}  anchor s={s:.2f}  D_s={v.item():.3f}  " + "  ".join(f"r({a:.2f})={r[a]:.3f}" for a in anchors))
    return delta.detach()


def protect(enc, x, **kw):
    """Protected photo: x + delta, rounded to 8 bits as any published image is."""
    delta = srim(enc, x, **kw)
    return ((x + delta).clamp(0, 1) * 255).round() / 255


@torch.no_grad()
def report(enc, x, x_prot, scales=None, kernel="lanczos"):
    """Latent deviation of the protected photo at each downscaling factor, with PIL resampling (as editors do)."""
    h, w = x.shape[-2:]
    scales = scales or anchors_for(h, w)
    out = {}
    for s in scales:
        size = target_size(h, w, s)
        z, zp = enc(resize_pil(x, size, kernel)), enc(resize_pil(x_prot, size, kernel))
        out[s] = rel_dev(zp, z).item()
    return out
