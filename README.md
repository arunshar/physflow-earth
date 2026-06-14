# PhysFlow-Earth

> Physics-informed rectified flow for Earth observation super-resolution and climate downscaling. Diffusers-compatible. `from_pretrained` loadable. Conserves mass and band ratios.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/Arun0808/physflow-earth)
![Sentinel-2 checkpoint scaffold](https://img.shields.io/badge/Sentinel--2-checkpoint%20scaffold-blue)
![ERA5 checkpoint scaffold](https://img.shields.io/badge/ERA5-checkpoint%20scaffold-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Diffusion-based super-resolution of satellite imagery and climate fields hallucinates plausible-but-physically-inconsistent textures: violated mass conservation in precipitation fields, broken Sentinel-2 band ratios (NDVI / NDWI / SAVI), non-physical divergence in wind fields. PhysFlow-Earth is a conditional rectified-flow trained with a hybrid objective: standard flow-matching velocity loss plus a physics-residual term evaluated on the predicted clean sample at every step, with the residual gradient backpropagated through the ODE solver. The result is generative super-resolution that respects the conservation laws specific to each modality.

This work extends three published diffusion papers (Pi-DPM, Kriging-informed conditional diffusion at SIGSPATIAL 2024, GCDM at SIGSPATIAL 2025) into the CV-canonical vocabulary of rectified flow + DiT for the WorldStrat / SEN2VENuS / ERA5 benchmarks.

## Highlights

- Conditional rectified-flow with a DiT backbone and cross-attention to a learned codebook of physics-constraint embeddings.
- Differentiable physics residual operators per modality: Sentinel-2 band-ratio preservation, ERA5 divergence-free wind, CHIRPS precipitation mass conservation.
- Diffusers-style pipeline (`PhysFlowPipeline`) with save/from_pretrained.
- Ships two paper implementations as submodules: Ki-CDPM (kriging-informed conditional diffusion downscaling, `physflow/kicdpm/`) and physics-guided generation (`physflow/pggenfm/`).

## Status: what is real vs. planned

REAL and tested: the rectified-flow training step + clean-sample projection (`flow/`), the DiT denoiser with physics-codebook conditioning (`models/`), all three differentiable physics residuals (`physics/`), the Euler/consistency samplers, the Diffusers-style pipeline, the Ki-CDPM and pggenfm modules, and a SYNTHETIC data layer (`physflow/data/`) so `training/train` and `eval/worldstrat_bench` run end to end without a download. NOT yet done: no trained checkpoints are shipped (`from_pretrained("Arun0808/physflow-sentinel2-x4")` would 404), no consistency distillation, and no real-dataset benchmark run. No leaderboard result is claimed.

## Quickstart

```bash
git clone https://github.com/arunshar/physflow-earth
cd physflow-earth
pip install -e .
# runs on the SYNTHETIC data layer (physflow/data) out of the box, no download:
python -m physflow.training.train +experiment=sentinel2_x4
```

To train on real WorldStrat / ERA5 data, replace the synthetic loaders in
`physflow/data/` with loaders over the real archives that return the same
`{"x_hr", "x_lr"}` dicts; nothing in the training loop changes.

## Smoke tests

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest                                    # 8 + 12 = 20 tests
python /tmp/launch_smoke.py "$(pwd)" space/app.py
```

Test status (CPU, in the project container):
- physics-residual + rectified-flow tests pass (mass conservation, divergence, NDVI/NDWI band ratios, flow training step with backward).
- Ki-CDPM (6) and pggenfm (3) tests pass.
- The Gradio Space tests require `gradio` installed; the Space itself is a demo wrapper, not a benchmarked system.

## Try the live demo

[HF Space](https://huggingface.co/spaces/Arun0808/physflow-earth) is a demo wrapper. It currently uses a bilinear-interpolation placeholder for the pipeline (no trained checkpoint is shipped), so treat it as a UI preview, not a working downscaler.

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

## Planned evaluation (NOT measured)

No benchmark has been run: there are no trained checkpoints and the data layer
shipped here is synthetic. The published baselines below are listed only as the
reference points a real run would target; no PhysFlow row is reported because no
PhysFlow number has been measured.

```bash
# requires a trained checkpoint and the real WorldStrat loader (neither shipped)
python -m physflow.eval.worldstrat_bench --metrics psnr ssim lpips ndvi_residual
```

| Method (reported by their authors) | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| SR3 | 27.4 | 0.78 | 0.21 |
| EDiffSR (CVPR 2024) | 28.1 | 0.80 | 0.18 |
| CorrDiff | 28.0 | 0.80 | 0.19 |

## License

Apache 2.0.
