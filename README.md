# SRIM: Scale-Robust Image Immunization against Malicious Editing

Minimal implementation of **SRIM** from

> *Protective Perturbations Must Survive the Resize: Scale-Robust Image Immunization against Malicious Editing*
> Zhongliang Guo, Yan Lin, Yifei Qian, Daizong Liu, Ognjen Arandjelović, Jianfeng Dong, Chun Pong Lau
> [arXiv:2610.07464](https://arxiv.org/abs/2610.07464)

Instruction-guided editors (FLUX.2, InstructPix2Pix, ...) downscale a shared photo by an unknown factor before
encoding it, and existing protective perturbations lose most of their effect after that resize. SRIM optimizes the
perturbation for the **weakest scale** of the range an editor may use: anchor scales cover the range implied by the
photo size, and every step is spent on the anchor where the protection currently falls behind. Each step encodes a
single downscaled copy, so SRIM costs the same as expectation over transformation (EOT) and never encodes the
full-resolution photo.

## Install

```bash
pip install -r requirements.txt
```

A GPU with about 24 GB of memory is enough for any photo size: the largest encoded copy has about 5 megapixels.
The VAE of the editor family is downloaded from the Hugging Face Hub on first use
(`black-forest-labs/FLUX.2-klein-4B` for the FLUX.2 family, which may require accepting the model license, or
`timbrooks/instruct-pix2pix`).

## Protect a photo

```bash
python protect.py photo.jpg protected.png                 # FLUX.2 family (klein 4B / 9B, dev), eps = 4/255
python protect.py photo.jpg protected.png --eps 8         # stronger, more visible
python protect.py photo.jpg protected.png --editor ip2p   # Stable-Diffusion-based editors (InstructPix2Pix VAE)
python protect.py photo.jpg protected.png --vae_repo black-forest-labs/FLUX.2-dev   # same VAE, if you already have FLUX.2-dev
```

The script prints the anchor scales, the deviation tracked at each anchor during optimization, and, at the end, the
relative latent deviation of the protected photo after Lanczos downscaling by each anchor (the editor's view).
Keep the output as **PNG**; JPEG re-compression weakens any protection. The three FLUX.2 editors share one VAE,
so a photo protected with `--editor flux2` is protected against all of them.

## Use it from Python

```python
import torch
from srim import VAEEncoder, load_image, protect, save_image

enc = VAEEncoder("flux2", device="cuda")
x = load_image("photo.jpg").cuda()                 # (1, 3, H, W) in [0, 1], cropped to multiples of 16
x_prot = protect(enc, x, eps=4 / 255, steps=200)   # 8-bit rounded
save_image(x_prot, "protected.png")
```

`srim.srim(enc, x, ...)` returns the perturbation itself; `srim.report(enc, x, x_prot)` measures the latent deviation
at each anchor with PIL resampling.

## Method in short (`srim.py`)

| component | default |
|---|---|
| objective | relative latent deviation of the editor's VAE encoder after downscaling, maximized by sign-gradient steps (step size eps/4, cosine decay, uniform random init, 8-bit rounding at the end) |
| scale range | `[sqrt(N / 4.2 MP), sqrt(N / 0.25 MP)]` for a photo of N pixels (editors work between 0.25 and 4.2 megapixels) |
| anchors | geometric grid with ratio 1.4 over the range (a 12-megapixel photo gets 6 anchors) |
| per step | one anchor, jittered by a factor in `[e^-0.1, e^0.1]`, resampled with a random antialiased kernel (bicubic, bilinear, area) |
| anchor choice | every anchor once at the start, then with probability `exp(-r_k / tau)`, `tau = 0.1`, where `r_k` is an exponential moving average (weight 0.3) of the deviation observed at anchor k |
| budget, steps | `eps = 4/255`, 200 steps |

## Citation

```bibtex
@article{guo2026srim,
  title   = {Protective Perturbations Must Survive the Resize: Scale-Robust Image Immunization against Malicious Editing},
  author  = {Guo, Zhongliang and Lin, Yan and Qian, Yifei and Liu, Daizong and Arandjelovi{\'c}, Ognjen and Dong, Jianfeng and Lau, Chun Pong},
  journal = {arXiv preprint arXiv:2610.07464},
  year    = {2026}
}
```

## License

MIT.
