# Physics-guided Generative FM (pggenfm)

Reference implementation of the guidance mechanism in

> Arun Sharma, Majid Farhadloo, Mingzhou Yang, Bharat Jayaprakash, William
> Northrop, Shashi Shekhar. "Towards Physics-guided Generative Foundation
> Models." 1st ACM SIGSPATIAL Workshop on Generative and Agentic AI for
> Multi-Modality Space-Time Intelligence (GeoGenAgent '25).
> DOI: 10.1145/3764915.3770717

A generative model proposes a sample; physics guidance then projects that sample
onto the physics-consistent manifold by descending a differentiable physics
residual while staying anchored to the proposal:

```
x* = argmin_x  E(x; condition)  +  anchor * || x - x0 ||^2
```

`E` reuses PhysFlow-Earth's existing differentiable physics residuals (mass
conservation `avgpool(x) - x_lr`, band ratios, divergence). The anchor keeps the
guided sample near the generative proposal, so guidance corrects physics
violations rather than discarding the sample.

| Component | Module |
|---|---|
| Guidance optimiser + before/after trace | `guidance.py` `physics_guided_refine`, `PhysicsGuidance` |
| Self-contained demo | `demo.py` |

Grew out of the numpy miniature in the `arun-papers` skill
(`code/physics-guided-generative-fm.py`).

## Run it

```bash
python -m physflow.pggenfm.demo
```

## Measured result (honest)

A toy generator proposes a high-resolution field that violates the coarse
observation (drifted block means + spurious divergence). After 150 guidance
steps, measured this run:

| physics violation | value |
|---|---|
| before guidance | 0.268 |
| after guidance | 0.029 |
| reduction | 89.3% |

The violation drops sharply because the mass-conservation energy is convex; the
guided sample respects the coarse observation while staying close to the
proposal. (Numbers are measured by `demo.py`, not a benchmark claim.)

## Tests

`tests/test_pggenfm.py` (3 cases): guidance reduces the violation, a stronger
anchor keeps the guided sample closer to the proposal, and the demo runs.
