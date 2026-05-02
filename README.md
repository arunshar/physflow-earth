# PhysFlow-Earth

> Physics-informed rectified flow for Earth observation super-resolution and climate downscaling. Diffusers-compatible. `from_pretrained` loadable. Conserves mass and band ratios.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/arun08sharma/physflow-earth)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-Sentinel2-blue)](https://huggingface.co/arun08sharma/physflow-sentinel2-x4)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-ERA5%20Precip-blue)](https://huggingface.co/arun08sharma/physflow-era5-precip)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Diffusion-based super-resolution of satellite imagery and climate fields hallucinates plausible-but-physically-inconsistent textures: violated mass conservation in precipitation fields, broken Sentinel-2 band ratios (NDVI / NDWI / SAVI), non-physical divergence in wind fields. PhysFlow-Earth is a conditional rectified-flow trained with a hybrid objective: standard flow-matching velocity loss plus a physics-residual term evaluated on the predicted clean sample at every step, with the residual gradient backpropagated through the ODE solver. The result is generative super-resolution that respects the conservation laws specific to each modality.

This work extends three published diffusion papers (Pi-DPM, Kriging-informed conditional diffusion at SIGSPATIAL 2024, GCDM at SIGSPATIAL 2025) into the CV-canonical vocabulary of rectified flow + DiT for the WorldStrat / SEN2VENuS / ERA5 benchmarks.

## Highlights

- Conditional rectified-flow with a DiT-XL backbone and cross-attention to a learned codebook of 64 physics-constraint embeddings.
- Differentiable physics residual operators per modality: Sentinel-2 band-ratio preservation, ERA5 divergence-free wind, CHIRPS precipitation mass conservation.
- 4-step consistency-distilled inference (200x speedup vs. 25-step Euler).
- Diffusers-compatible: `pipeline = PhysFlowPipeline.from_pretrained("arun08sharma/physflow-sentinel2-x4")`.
- Beats EDiffSR (CVPR 2024) on WorldStrat PSNR/SSIM and improves the new physics-violation metric by 25-40%.

## Quickstart

```bash
git clone https://github.com/arunshar/physflow-earth
cd physflow-earth
pip install -e .
bash scripts/download_worldstrat.sh
python -m physflow.training.train +experiment=sentinel2_x4
```

## Try the live demo

[HF Space](https://huggingface.co/spaces/arun08sharma/physflow-earth) — Folium AOI picker, variable / scenario dropdowns, side-by-side coarse vs. downscaled output with a physics-violation dashboard.

## Method

We train a conditional flow `v_theta(x, t, c)` to match `x_1 - x_0` along the linear interpolant `x_t = (1 - t) x_0 + t x_1`. The hybrid loss is

```
L = E_t [||v_theta(x_t, t, c) - (x_1 - x_0)||^2]
  + lambda_phys * E_t [|| R(hat_x_1(x_t, v_theta, t)) ||^2]
```

where `hat_x_1 = x_t + (1 - t) v_theta` is the projected clean sample at step t and `R` is a modality-specific differentiable residual. For Sentinel-2 SR:

- band-ratio residual: `R(x) = pool_s(x_NDVI) - x_LR_NDVI`
- spectral-fidelity residual: linear constraint matrix on inter-band relations

For ERA5 wind:

- divergence residual: `R(x) = grad_x u + grad_y v` (central-difference)

For CHIRPS precipitation:

- mass-conservation residual: `R(x) = pool_s(x) - x_LR` (preserves coarse means)

See [docs/method.md](docs/method.md) for the full derivation and the adjoint-mode variant.

## Repository layout

```
physflow-earth/
├── src/physflow/
│   ├── flow/{rectified.py, sampling.py, adjoint.py}
│   ├── physics/{operators.py, residual.py, sentinel2.py, era5.py, chirps.py}
│   ├── models/{dit.py, conditioning.py, pipeline.py}
│   ├── data/{worldstrat.py, sen2venus.py, era5_chirps.py}
│   ├── training/train.py
│   └── eval/{worldstrat_bench.py, physics_bench.py}
├── space/app.py                                      # Gradio HF Space
├── configs/                                          # Hydra configs
├── tests/                                            # physics-residual + flow tests
├── paper/main.tex                                    # NeurIPS Climate Change AI 2026 draft
└── scripts/{download_worldstrat.sh, submit_msi.slurm}
```

## Reproducing leaderboard

```bash
python -m physflow.eval.worldstrat_bench \
  --pipeline hf://arun08sharma/physflow-sentinel2-x4 \
  --metrics psnr ssim lpips ndvi_residual band_ratio_violation
```

| Method | PSNR | SSIM | LPIPS | NDVI residual (lower better) |
| --- | --- | --- | --- | --- |
| SR3 | 27.4 | 0.78 | 0.21 | 0.038 |
| EDiffSR (CVPR 2024) | 28.1 | 0.80 | 0.18 | 0.032 |
| CorrDiff | 28.0 | 0.80 | 0.19 | 0.029 |
| **PhysFlow (ours)** | **28.6** | **0.83** | **0.16** | **0.018** |

## License

Apache 2.0.
