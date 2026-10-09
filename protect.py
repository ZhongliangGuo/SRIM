"""Protect one photo with SRIM.

    python protect.py photo.jpg protected.png            # FLUX.2 family (klein 4B/9B, dev), eps = 4/255, 200 steps
    python protect.py photo.jpg protected.png --eps 8 --editor ip2p

Needs a GPU; the largest encoded copy has about 5 megapixels (the largest anchor with its jitter), independent of
the photo size. Save the output as PNG: JPEG re-compression weakens any protection.
"""
import argparse
import random
import time

import numpy as np
import torch

from srim import VAEEncoder, anchors_for, load_image, protect, report, save_image, scale_range

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("input")
p.add_argument("output", help="protected photo (PNG)")
p.add_argument("--editor", default="flux2", choices=["flux2", "ip2p"], help="editor family whose VAE is the surrogate")
p.add_argument("--vae_repo", default=None, help="Hugging Face repo to load the VAE from (default: see srim.EDITORS; "
               "e.g. black-forest-labs/FLUX.2-dev, whose VAE equals the FLUX.2-klein one)")
p.add_argument("--eps", type=float, default=4.0, help="l_inf budget in 8-bit levels (4 = 4/255)")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--tau", type=float, default=0.1, help="temperature of the anchor distribution")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
p.add_argument("--quiet", action="store_true")
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
x = load_image(args.input).to(args.device)
h, w = x.shape[-2:]
lo, hi = scale_range(h, w)
print(f"{args.input}: {w}x{h} ({h * w / 1e6:.1f} MP), scale range [{lo:.2f}, {hi:.2f}], anchors "
      + ", ".join(f"{a:.2f}" for a in anchors_for(h, w)))
enc = VAEEncoder(args.editor, device=args.device, repo=args.vae_repo)
t0 = time.time()
x_prot = protect(enc, x, eps=args.eps / 255, steps=args.steps, tau=args.tau, log=None if args.quiet else print)
print(f"protected in {time.time() - t0:.0f} s")
save_image(x_prot, args.output)
dev = report(enc, x, x_prot)
print("latent deviation after Lanczos downscaling: " + "  ".join(f"s={s:.2f}: {d:.3f}" for s, d in dev.items()))
print(f"saved {args.output}")
