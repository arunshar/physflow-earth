# Method

## Rectified flow training

Given target samples $x_1$ and standard-Gaussian noise $x_0$, we form
$x_t = (1-t) x_0 + t x_1$ and train a velocity model $v_\theta$ to match the
constant target velocity $x_1 - x_0$:

$$
\mathcal{L}_v = \mathbb{E}_t \left\|v_\theta(x_t, t, c) - (x_1 - x_0)\right\|^2.
$$

## Hybrid physics-informed loss

We project $x_t$ to a "clean" estimate $\hat{x}_1 = x_t + (1-t) v_\theta$ at every
training step and apply a modality-specific differentiable residual $R$:

$$
\mathcal{L} = \mathcal{L}_v + \lambda \, \mathbb{E}_t \|R(\hat{x}_1)\|^2.
$$

The residuals are:

- **Sentinel-2 band-ratio**: pool the predicted NDVI/NDWI to the LR grid and
  match the LR-derived NDVI/NDWI.
- **ERA5 wind divergence**: central-difference $\nabla \cdot (u, v)$.
- **CHIRPS precipitation mass conservation**: average-pool the predicted
  HR field to LR and match $x_{LR}$ exactly.

## Adjoint-mode variant

When tighter conservation is needed, we provide an `AdjointPhysicsLoss`
that integrates the ODE with `torchdiffeq.odeint_adjoint` and applies the
residual at every step. We default to the instantaneous loss because the
adjoint mode is ~3x slower and the gain is marginal once the codebook is
trained.

## Physics codebook

Each DiT block cross-attends to a learned codebook of 64 embeddings. The
model is free to interpret these as "this region must preserve NDVI" or
"this is a region of strong divergence." Empirically, the codebook
converges to a few interpretable directions: vegetation indices,
precipitation hotspots, divergence-free wind regions.
