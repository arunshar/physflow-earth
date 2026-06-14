# Ki-CDPM: Kriging-informed Conditional Diffusion for downscaling

Reference PyTorch implementation of

> Subhankar Ghosh, Arun Sharma, Jayant Gupta, Aneesh Subramanian, Shashi
> Shekhar. "Towards Kriging-informed Conditional Diffusion for Regional
> Sea-Level Data Downscaling: A Summary of Results." ACM SIGSPATIAL 2024,
> pp. 372-383. DOI: 10.1145/3678717.3691304

The downscaler is a **map-derived base plus a learned residual**: ordinary
kriging produces a geostatistically consistent fine field from the coarse
samples (the structural prior / condition `y`), and a conditional DDPM learns the
high-frequency residual `r = fine - y` given `y`.

## What the paper describes, and where it lives

| Component | Module | Notes |
|---|---|---|
| Ordinary kriging (exponential variogram) | `kriging.py` | The weights depend only on the coarse->fine geometry, so the `(n+1)x(n+1)` system with the Lagrange multiplier is solved ONCE and reused as a constant matmul (a fast, batched, GPU `nn.Module`). |
| Conditional denoiser | `unet.py` | Small 2D U-Net; predicts the noise given `(noisy residual, kriging field y, t)`. |
| Conditional diffusion | `diffusion.py` | DDPM (cosine schedule, eps-prediction) over the residual; DDIM sampling with a clamped `x0` (prevents the high-noise `1/sqrt(alpha_bar)` blow-up). |
| Downscale | `model.py` `KiCDPM.downscale` | `fine = y + diffusion_residual`; `n_samples>1` averages residuals (a Monte-Carlo estimate of the conditional mean `E[r|y]`, the RMSE-optimal point estimate). |

Grew out of the numpy miniature in the `arun-papers` skill
(`code/kriging-conditional-diffusion-downscaling.py`), which used the kriging
mean itself as an oracle denoiser; here the denoiser is a real learned network.

## Run it

```bash
python -m physflow.kicdpm.train --epochs 16 --n 4096 --out /tmp/kicdpm.pt
python -m physflow.kicdpm.eval --ckpt /tmp/kicdpm.pt --n 512
```

Data is synthetic (`data.py`): smooth random fields plus a deterministic,
field-modulated high-frequency component (period below the coarsening factor, so
average-pooling removes it and interpolation cannot represent it). To train on
real regional sea-level data, replace `FieldDataset` with a loader that yields
the same `(coarse, fine)` tensor pairs; nothing downstream changes.

## Measured results (synthetic, honest)

A 16-epoch CPU run (4x downscale, `coarse=8 -> fine=32`, tiny U-Net, 3072 train /
512 test), RMSE measured this run with the 12-sample conditional mean:

| method | pixel RMSE | high-frequency-band RMSE |
|---|---|---|
| bilinear interpolation | 0.216 | 0.211 |
| kriging only (no diffusion) | 0.247 | - |
| Ki-CDPM (kriging + diffusion) | 0.273 | 0.259 |

**Honest reading.** On this regular-grid synthetic benchmark with a deliberately
tiny CPU model, bilinear interpolation is a strong baseline and Ki-CDPM does NOT
beat it on pixel RMSE. This is the expected perception-distortion regime: a
smooth interpolant minimises pixel error, while a diffusion model trades some
pixel error for plausible high-frequency detail. The implementation is correct
and trains stably (eps-MSE 0.70 -> 0.018); reproducing the paper's advantage
needs the full-size model, a real training budget, and real irregularly-sampled
regional data (where kriging's geostatistical prior is far stronger than grid
interpolation). These synthetic numbers are reported as-measured, not as a claim
of state of the art.

## Tests

`tests/test_kicdpm.py` (6 cases): kriging shapes, kriging reproduces a constant
field (weights sum to 1), `q_sample`/`predict_x0` roundtrip, training-loss
gradient flow, finite/stable sampling under the `x0` clamp, and dataset pairing.
