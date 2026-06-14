"""Physics-guided generation for PhysFlow-Earth (Physics-guided Generative FM).

Reference implementation of the guidance mechanism in

    Arun Sharma, Majid Farhadloo, Mingzhou Yang, Bharat Jayaprakash, William
    Northrop, Shashi Shekhar. "Towards Physics-guided Generative Foundation
    Models." 1st ACM SIGSPATIAL Workshop on Generative and Agentic AI for
    Multi-Modality Space-Time Intelligence (GeoGenAgent '25).
    DOI: 10.1145/3764915.3770717

A generative foundation model proposes a sample; physics guidance then projects
that sample onto the physics-consistent manifold by descending the differentiable
physics residual while staying anchored to the proposal. The physics violation of
the guided sample is provably no larger than the proposal's, which is what the
paper reports (violation drops after guidance). This reuses PhysFlow-Earth's
existing differentiable physics residuals (mass conservation, band ratio,
divergence) as the guidance energy.

Grew out of the numpy miniature in the arun-papers skill
(code/physics-guided-generative-fm.py).
"""

from __future__ import annotations

from .guidance import PhysicsGuidance, physics_guided_refine

__all__ = ["PhysicsGuidance", "physics_guided_refine"]
