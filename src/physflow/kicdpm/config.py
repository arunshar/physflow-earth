"""Configuration for Ki-CDPM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KiCDPMConfig:
    # ----- grids ---------------------------------------------------------
    coarse: int = 8                # coarse grid side (HxW = coarse x coarse)
    scale_factor: int = 4          # fine = coarse * scale_factor
    channels: int = 1             # field channels (1 = scalar sea-level field)

    # ----- kriging (exponential variogram) -------------------------------
    sill: float = 1.0
    length_scale: float = 3.0
    nugget: float = 1e-2

    # ----- denoiser ------------------------------------------------------
    base_ch: int = 48
    ch_mult: tuple[int, ...] = (1, 2)
    time_dim: int = 128

    # ----- diffusion -----------------------------------------------------
    timesteps: int = 200
    beta_schedule: str = "cosine"
    ddim_steps: int = 30
    x0_clamp: float = 4.0          # clamp predicted clean residual each DDIM step
                                   # (data is normalised, so residuals are O(1));
                                   # prevents the high-noise 1/sqrt(alpha_bar) blow-up

    # ----- training ------------------------------------------------------
    lr: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 30
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0

    @property
    def fine(self) -> int:
        return self.coarse * self.scale_factor
