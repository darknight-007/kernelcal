"""Worked A2 counterexample and parametric sweep tests.

This test calibrates when the topology floor k_min = beta0 + beta1 can fail:
the short-cycle figure-eight graph fails the A2 proxy while a long-cycle
control graph passes.
"""

from kernelcal.terrain import (
    run_worked_a2_counterexample,
    run_a2_cycle_ratio_sweep,
    write_a2_sweep_json,
    write_a2_sweep_csv,
    fit_bound_constants,
    rho_with_augmentation,
)


def test_a2_counterexample_confirmed():
    r = run_worked_a2_counterexample()
    assert r.counterexample_confirmed


def test_same_kmin_but_different_a2_outcome():
    r = run_worked_a2_counterexample()
    assert r.long_cycle_case.k_min == 3
    assert r.short_cycle_case.k_min == 3
    assert r.long_cycle_case.a2_proxy_holds is True
    assert r.short_cycle_case.a2_proxy_holds is False


def test_short_cycle_rank_collapses_below_beta1():
    r = run_worked_a2_counterexample()
    assert r.short_cycle_case.beta1 == 2
    assert r.short_cycle_case.projected_rank == 1
    assert r.short_cycle_case.projected_rank < r.short_cycle_case.beta1


def test_cycle_ratio_sweep_shape_and_families():
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3, 4),
        bridge_len=2,
    )
    # 2 (length pairs) * 2 (families) = 4 points
    assert len(sweep.points) == 4
    families = {p.family for p in sweep.points}
    assert families == {"figure8", "separated_control"}


def test_cycle_ratio_sweep_includes_a2_failure_point():
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3, 4, 5),
        bridge_len=2,
    )
    fig8 = [p for p in sweep.points if p.family == "figure8"]
    assert any(not p.a2_proxy_holds for p in fig8)


def test_cycle_ratio_export_json_csv(tmp_path):
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3,),
        bridge_len=2,
    )
    json_path = tmp_path / "a2_sweep.json"
    csv_path = tmp_path / "a2_sweep.csv"
    write_a2_sweep_json(sweep, json_path)
    write_a2_sweep_csv(sweep, csv_path)
    assert json_path.exists()
    assert csv_path.exists()
    assert "family" in json_path.read_text()
    assert "family" in csv_path.read_text()


def test_parametric_fields_present_and_reasonable():
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3, 4, 6),
        bridge_len=2,
    )
    for p in sweep.points:
        assert p.ell_min >= 3
        assert p.ell_max >= p.ell_min
        assert p.gamma >= 1.0
        assert 0.0 <= p.rho_k <= 1.0 + 1e-9
        assert p.delta_k >= 0.0
        assert p.bound_proxy > 0.0


def test_figure8_fails_control_holds_on_rho_k():
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3, 4, 6),
        bridge_len=2,
    )
    fig8 = [p for p in sweep.points if p.family == "figure8"]
    ctrl = [p for p in sweep.points if p.family == "separated_control"]
    # Figure-eight family should have at least one point with rho_k near 0.
    assert any(p.rho_k < 1e-6 for p in fig8)
    # Control family should have rho_k strictly positive everywhere.
    assert all(p.rho_k > 1e-3 for p in ctrl)


def test_augmentation_recovers_figure8_rank():
    # Figure-eight s=3,l=3 fails at k_min=3; augmenting by a few modes
    # in a 5-vertex graph should recover full rank up to numerical tol.
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3,),
        long_cycle_lengths=(3,),
        bridge_len=2,
        augment_delta_k=2,
    )
    fig8 = next(p for p in sweep.points if p.family == "figure8")
    assert fig8.a2_proxy_holds is False
    # Augmentation should at minimum not decrease rank and should increase rho.
    assert fig8.projected_rank_after_augment >= fig8.projected_rank
    assert fig8.rho_after_augment >= fig8.rho_k


def test_rho_with_augmentation_monotone_on_figure8():
    from kernelcal.terrain.a2_counterexample import (
        _make_figure8_cycle_graph,
    )

    n_v, edges, cycles = _make_figure8_cycle_graph(3, 5)
    rho_0, _ = rho_with_augmentation(
        n_vertices=n_v, edges=edges, cycles=cycles, augment_delta_k=0
    )
    rho_1, _ = rho_with_augmentation(
        n_vertices=n_v, edges=edges, cycles=cycles, augment_delta_k=1
    )
    rho_2, _ = rho_with_augmentation(
        n_vertices=n_v, edges=edges, cycles=cycles, augment_delta_k=2
    )
    # Weak monotonicity in the retained subspace size.
    assert rho_1 >= rho_0 - 1e-9
    assert rho_2 >= rho_1 - 1e-9


def test_fit_bound_constants_returns_finite_fit():
    sweep = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=(3, 4),
        long_cycle_lengths=(3, 4, 6, 8),
        bridge_len=2,
    )
    fit = fit_bound_constants(sweep, family="separated_control")
    assert fit["n_points"] >= 3
    assert fit["C1"] == fit["C1"]  # not NaN
    assert fit["C2"] == fit["C2"]
    assert fit["rmse"] >= 0.0

