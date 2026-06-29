# physflow-earth — reproduced evidence

_Generated 2026-06-29T10:30:32Z by running the test suite on the real/tested code in this repo._

These are **reproduced** results: the code runs and every assertion below holds. Benchmark/leaderboard numbers in the paper (PSNR, mIoU, speedups) remain **targets, not reproduced**, and are labeled as such throughout.

## Test suite (`pytest -v`)

```
tests/test_physics_residuals.py::test_average_pool_inverse_of_nearest_upsample PASSED [  5%]
tests/test_physics_residuals.py::test_horizontal_divergence_zero_on_constant_field PASSED [ 10%]
tests/test_physics_residuals.py::test_horizontal_divergence_recovers_linear_gradient PASSED [ 15%]
tests/test_physics_residuals.py::test_mass_conservation_residual_zero_on_perfect_upsample PASSED [ 20%]
tests/test_physics_residuals.py::test_band_ratio_residual_zero_on_perfect_upsample PASSED [ 25%]
tests/test_physics_residuals.py::test_divergence_residual_module_runs PASSED [ 30%]
tests/test_rectified_flow.py::test_velocity_loss_decreases PASSED        [ 35%]
tests/test_rectified_flow.py::test_physics_loss_is_nonnegative PASSED    [ 40%]
tests/test_smoke.py::test_top_level_imports PASSED                       [ 45%]
tests/test_smoke.py::test_physics_imports PASSED                         [ 50%]
tests/test_smoke.py::test_flow_imports PASSED                            [ 55%]
tests/test_smoke.py::test_models_imports PASSED                          [ 60%]
tests/test_smoke.py::test_dit_forward_shape PASSED                       [ 65%]
tests/test_smoke.py::test_pipeline_inference_shape PASSED                [ 70%]
tests/test_smoke.py::test_rectified_flow_step_e2e PASSED                 [ 75%]
tests/test_smoke.py::test_space_app_importable PASSED                    [ 80%]
tests/test_smoke.py::test_space_ui_builds PASSED                         [ 85%]
tests/test_smoke.py::test_space_constants_defined PASSED                 [ 90%]
tests/test_smoke.py::test_space_requirements_parseable PASSED            [ 95%]
tests/test_smoke.py::test_space_readme_has_hf_frontmatter PASSED         [100%]

============================== 20 passed in 2.00s ==============================
```

## Reproduced demo (headline number)

The horizontal-divergence operator returns max|div| = 0.0 (exact) on a divergence-free constant field, confirming the conservation residual is correct.
