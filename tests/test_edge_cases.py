"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): d = 0, großes d, Grenze, mehrfache Wurzel, extremes R, b = 0, dh = 0, θ ganzzahlig, große Matrizen."""

from fractions import Fraction as F

import numpy as np
import pytest

import rvz_dynamics as K
import rvz_exact as X


def test_d_zero_is_the_undelayed_design_and_has_no_critical_r():
    k1, k2 = K.gain(1.0)
    assert K.rho_poly(k1, k2, 0) == pytest.approx(0.422, abs=5e-4)
    assert K.r_krit(0) is None
    xs, us = K.simulate(k1, k2, 0, (1.0, 0.0), 3)
    assert us[0] == pytest.approx(-k1) and xs[1].tolist() == [1.0, -k1]


def test_large_d_uses_the_polynomial_roots_and_the_state_matrix_only_as_a_band():
    """d = 40: die d Nullwurzeln der Zustandsmatrix rauschen; ρ kommt aus np.roots von P_d, die Zustandsmatrix stimmt nur im Band 10⁻³."""
    k1, k2 = K.gain(1000.0)
    Km = np.array([[k1, k2]])
    a = K.rho_poly(k1, k2, 40)
    b = K.rho_eig(K.aug_delay(K.B0, Km, 40))
    assert a > 1.0 and abs(a - b) < 1e-3


def test_gain_exactly_on_the_boundary_is_a_band_not_an_equality():
    r = K.scalar_rho(K.scalar_gain(), 2)       # der Wert kann 1 − ε oder 1 + ε sein: nie auf ρ == 1 prüfen, nur als Band
    assert abs(r - 1.0) < 1e-9
    assert K.scalar_rho(K.scalar_gain(), 1) < 1.0 - 0.2 and K.scalar_rho(K.scalar_gain(), 3) > 1.0 + 0.05   # beide Seiten der Grenze sind weit genug entfernt


def test_multiple_root_is_resolved_only_to_about_the_cube_root_of_epsilon():
    assert K.rho_poly(1 / 27, 1 / 3, 1) == pytest.approx(2 / 3, abs=1e-4)
    assert X.schur_stable(K.poly_wagen(F(1, 27), F(1, 3), 1))                  # exakt stabil


@pytest.mark.parametrize("R", [1e-6, 1e-3, 1e4, 1e8])
def test_extreme_r_gives_a_finite_stable_undelayed_design(R):
    k1, k2 = K.gain(R)
    assert np.isfinite(k1) and np.isfinite(k2) and k1 > 0 and k2 > 0
    assert K.rho_poly(k1, k2, 0) < 1.0


def test_invalid_inputs_are_rejected():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            K.lqr(bad)
    with pytest.raises(ValueError):
        K.poly_wagen(0.4, 1.2, -1)
    with pytest.raises(ValueError):
        K.simulate(0.4, 1.2, -1, (1, 0), 3)
    with pytest.raises(ValueError):
        K.simulate(0.4, 1.2, 1, (1, 0), 3, dh=-1)
    with pytest.raises(ValueError):
        K.theta_split(-0.1)


def test_dead_plant_b_zero_has_the_open_loop_spectrum():
    """b = 0 ohne Vorhersage: die Strecke reagiert nicht auf u, die Schleife hat die Wurzeln von A (zweifach 1) und die Nullwurzeln der Verzugskette: ρ = 1, nie stabil.
    (Mit Vorhersage hat der Regler eigene Zustände, deren Eigenwerte hier nicht Gegenstand sind.)"""
    k1, k2 = K.gain(1.0)
    for d in range(4):
        assert K.rho_loop(k1, k2, d, 0, b=0.0) == pytest.approx(1.0, abs=1e-6)


def test_assumed_delay_zero_for_positive_d_is_the_loop_without_prediction():
    k1, k2 = K.gain(10.0)
    for d in range(1, 5):
        assert K.rho_loop(k1, k2, d, 0) == pytest.approx(K.rho_poly(k1, k2, d), abs=1e-9)


def test_theta_zero_and_integers_have_no_rounding_noise():
    Km = K.lqr(1.0, "ZOH")[1]
    assert K.theta_split(0.0) == (0, 0.0) and K.theta_split(2.0) == (2, 0.0) and K.theta_split(0.3 + 0.6 + 0.1)[0] in (0, 1)
    assert K.rho_eig(K.theta_matrix(Km, 0.0)) == pytest.approx(K.rho_eig(K.aug_delay(K.B_ZOH, Km, 0)), abs=1e-12)
    assert K.rho_eig(K.theta_matrix(Km, 3.0)) == pytest.approx(K.rho_eig(K.aug_delay(K.B_ZOH, Km, 3)), abs=1e-9)


def test_largest_matrix_of_the_app_builds_and_is_finite():
    k1, k2 = K.gain(1.0)
    M = K.loop_matrix(k1, k2, 8, 8)
    assert M.shape == (26, 26) and np.all(np.isfinite(M))
    assert K.rho_eig(M) == pytest.approx(K.rho_poly(k1, k2, 0), abs=1e-3)      # Vorhersage-Schleife: Wurzeln des Entwurfs, Nullwurzeln rauschen


def test_critical_values_at_the_reference_design():
    assert K.theta_krit(1.0) == pytest.approx(0.6209, abs=2e-4)
    assert K.r_krit(1) == pytest.approx(6.854, abs=1e-3)
