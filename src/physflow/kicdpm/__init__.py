"""Ki-CDPM: Kriging-informed Conditional Diffusion for grid downscaling.

Reference PyTorch implementation of

    Subhankar Ghosh, Arun Sharma, Jayant Gupta, Aneesh Subramanian, Shashi
    Shekhar. "Towards Kriging-informed Conditional Diffusion for Regional
    Sea-Level Data Downscaling: A Summary of Results." ACM SIGSPATIAL 2024,
    pp. 372-383. DOI: 10.1145/3678717.3691304

The downscaler is a map-derived base plus a learned residual: ordinary kriging
produces a geostatistically consistent fine-resolution field from the coarse
samples (the structural prior / condition y), and a conditional DDPM learns the
high-frequency residual r = fine - y given y. This recovers detail that kriging
alone smooths away while staying anchored to the kriging mean.

  * kriging.py   -- ordinary kriging as a precomputed linear operator (the
                    weights depend only on the coarse->fine geometry, so the
                    kriging system is solved once and reused as a matmul).
  * unet.py      -- small 2D conditional U-Net denoiser (noisy residual + y).
  * diffusion.py -- Gaussian DDPM over the residual, conditioned on y.
  * data.py      -- synthetic smooth random fields (sea-level-like) -> coarse/fine.
  * model.py     -- KiCDPM: downscale(coarse) = y + diffusion_sample(cond=y).
  * train.py / eval.py -- training and RMSE-vs-bilinear / vs-kriging evaluation.

Grew out of the numpy miniature in the arun-papers skill
(code/kriging-conditional-diffusion-downscaling.py), which used the kriging mean
itself as an oracle denoiser; here the denoiser is a real learned network.
"""

from __future__ import annotations

from .config import KiCDPMConfig
from .diffusion import GridDiffusion
from .kriging import OrdinaryKriging
from .model import KiCDPM
from .unet import ConditionalUNet2D

__all__ = ["KiCDPMConfig", "GridDiffusion", "OrdinaryKriging", "KiCDPM", "ConditionalUNet2D"]
