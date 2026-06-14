"""Physics guidance: project a generated sample onto the physics manifold.

Given a generated field x0 and a differentiable physics-residual energy
E(x; condition), guidance minimises

    L(x) = E(x; condition) + anchor * ||x - x0||^2

by gradient descent. The anchor keeps the guided sample near the generative
proposal (so guidance corrects physics violations rather than discarding the
sample), while E drives the average-pool / band-ratio / divergence residuals
toward zero. Returns the guided sample and a before/after violation trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GuidanceResult:
    sample: torch.Tensor
    violation_before: float
    violation_after: float

    @property
    def reduction(self) -> float:
        if self.violation_before == 0:
            return 0.0
        return 1.0 - self.violation_after / self.violation_before


def physics_guided_refine(
    x0: torch.Tensor,
    energy: nn.Module,
    condition: dict,
    steps: int = 100,
    lr: float = 0.2,
    anchor: float = 0.05,
) -> GuidanceResult:
    """Descend energy(x, condition) + anchor*||x - x0||^2 from the proposal x0."""
    with torch.no_grad():
        before = float(energy(x0, condition))
    x = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = energy(x, condition) + anchor * (x - x0).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = float(energy(x, condition))
    return GuidanceResult(sample=x.detach(), violation_before=before, violation_after=after)


class PhysicsGuidance(nn.Module):
    """Wrap a generator so its samples are physics-guided after generation.

    ``generator(condition)`` should return a proposed field x0; ``energy`` is a
    PhysFlow physics-residual module (e.g. physflow.physics.PhysicsResidual).
    """

    def __init__(self, generator: nn.Module, energy: nn.Module, steps: int = 100,
                 lr: float = 0.2, anchor: float = 0.05) -> None:
        super().__init__()
        self.generator = generator
        self.energy = energy
        self.steps, self.lr, self.anchor = steps, lr, anchor

    @torch.no_grad()
    def _propose(self, condition: dict) -> torch.Tensor:
        return self.generator(condition)

    def generate(self, condition: dict) -> GuidanceResult:
        x0 = self._propose(condition)
        return physics_guided_refine(x0, self.energy, condition, self.steps, self.lr, self.anchor)
