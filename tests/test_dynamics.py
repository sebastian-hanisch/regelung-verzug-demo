"""Einheiten von Hand (fünf Handbeispiele): Bruch gleich Bruch, wo es geht; Gleitkomma nur als Band.

Jeder Erwartungswert steht hier als Rechnung oder als von Hand hergeleiteter Bruch, nicht als Abschrift der Ausgabe des Codes."""

from fractions import Fraction as F

import numpy as np
import pytest

import rvz_dynamics as K
import rvz_exact as X
from rvz_rng import SplitMix64


def test_hand_polynomials_p0_and_p1(mini):
    k1, k2 = mini
    assert K.poly_wagen(k1, k2, 0) == [1, F(-4, 5), F(1, 5)]                     # z² − (4/5) z + 1/5, Wurzeln 2/5 ± i/5
    assert K.poly_wagen(k1, k2, 1) == [1, -2, F(11, 5), F(-4, 5)]                # z³ − 2 z² + (11/5) z − 4/5
    assert K.rho_poly(0.4, 1.2, 0) == pytest.approx(np.sqrt(0.2), abs=1e-12)    # ρ² = 1/5
    assert K.rho_poly(0.4, 1.2, 1) == pytest.approx(1.1725, abs=5e-4)


def test_hand_schur_cohn_decisions(mini):
    k1, k2 = mini
    assert X.schur_stable(K.poly_wagen(k1, k2, 0)) is True
    assert X.schur_stable(K.poly_wagen(k1, k2, 1)) is False
    assert X.poly_roots_sum(K.poly_wagen(k1, k2, 1)) == 2                        # Vieta: Koeffizient von z^(d+1)


def test_hand_trajectories_without_prediction(mini):
    k1, k2 = mini
    xs, _ = X.trajectory(k1, k2, 1, (1, 0), 4, predict=False)
    assert [tuple(v) for v in xs[1:5]] == [(1, 0), (1, F(-2, 5)), (F(3, 5), F(-4, 5)), (F(-1, 5), F(-18, 25))]


def test_hand_trajectories_with_prediction(mini):
    k1, k2 = mini
    xs, _ = X.trajectory(k1, k2, 1, (1, 0), 4, predict=True)
    assert [tuple(v) for v in xs[2:5]] == [(1, F(-2, 5)), (F(3, 5), F(-8, 25)), (F(7, 25), F(-22, 125))]
    # danach exakt x_{m+1} = (A − BK) x_m
    for m in range(2, 4):
        x1, x2 = xs[m]
        assert xs[m + 1] == [x1 + x2, x2 - (k1 * x1 + k2 * x2)]


def test_hand_best_static_gain_is_a_triple_root():
    p = K.poly_wagen(F(1, 27), F(1, 3), 1)
    assert p == [1, -2, F(4, 3), F(-8, 27)]                                      # (z − 2/3)³ = z³ − 2 z² + (4/3) z − 8/27
    assert p == [1, -3 * F(2, 3), 3 * F(2, 3) ** 2, -F(2, 3) ** 3]
    assert X.schur_stable(p)
    assert K.rho_poly(1 / 27, 1 / 3, 1) == pytest.approx(2 / 3, abs=1e-4)       # dreifache Wurzel: np.roots verliert etwa ε^(1/3)


def test_hand_deadbeat_without_delay_only():
    assert K.poly_wagen(F(1), F(2), 0) == [1, 0, 0]                              # P_0 = z²: Deadbeat
    assert K.rho_poly(1.0, 2.0, 0) < 1e-6
    assert K.rho_poly(1.0, 2.0, 1) > 0.5                                          # mit Verzug nicht mehr (Schranke 2/3)


def test_hand_equilibrium_with_and_without_prediction(mini):
    k1, k2 = mini
    assert X.equilibrium_x1(k1, k2, 1, True) == F(11, 2)                         # x̂ = (x1, −1): −(2/5 x1 − 6/5) = −1
    assert X.equilibrium_x1(k1, k2, 1, False) == 1 / k1 == F(5, 2)
    assert X.equilibrium_x1(F(1, 27), F(1, 3), 1, False) == 27
    assert X.equilibrium_x1(k1, k2, 0, True) == F(5, 2)                          # d = 0: dieselbe Lage wie ohne Vorhersage


def test_hand_gamma_at_half_period():
    g0, g1 = X.gamma_exact(F(1, 2))
    assert g0 == [F(1, 8), F(1, 2)] and g1 == [F(3, 8), F(1, 2)]
    assert [g0[0] + g1[0], g0[1] + g1[1]] == [F(1, 2), 1]                        # Γ0 + Γ1 = B


def test_hand_theta_half_period_polynomial_and_radii(mini):
    k1, k2 = mini
    M = X.theta_matrix_exact(k1, k2, 0, F(1, 2))
    assert X.charpoly(M) == [1, F(-27, 20), F(13, 10), F(-11, 20)]
    Kv = np.array([[0.4, 1.2]])
    assert K.rho_eig(K.theta_matrix(Kv, 0.5)) == pytest.approx(0.919, abs=5e-4)
    assert K.rho_eig(K.theta_matrix(Kv, 0.0)) == pytest.approx(0.600, abs=5e-4)
    assert K.rho_eig(K.theta_matrix(Kv, 1.0)) == pytest.approx(1.231, abs=5e-4)


def test_hand_scalar_riccati_and_bracketing():
    phi = (1 + 5 ** 0.5) / 2
    p = phi                                                                       # p = 1 + p − p²/(1 + p)  ⇒  p² = 1 + p
    assert p ** 2 == pytest.approx(1 + p, rel=1e-14)
    assert K.scalar_gain() == pytest.approx(p / (1 + p), rel=1e-14) == pytest.approx((5 ** 0.5 - 1) / 2, rel=1e-14)
    # K = 3/5 stabil genau für d ≤ 2, K = 5/8 bei d = 2 instabil (Schur–Cohn exakt)
    assert [X.scalar_stable(F(3, 5), d) for d in range(5)] == [True, True, True, False, False]
    assert X.scalar_stable(F(5, 8), 2) is False
    assert K.scalar_rho(K.scalar_gain(), 0) == pytest.approx(0.381966, abs=1e-6)  # 1 − K
    assert K.scalar_rho(K.scalar_gain(), 1) == pytest.approx(0.786151, abs=1e-6)  # √K


def test_scalar_gain_sits_on_the_lager_boundary_at_d2_as_a_band():
    r = K.scalar_rho(K.scalar_gain(), 2)
    assert abs(r - 1.0) < 1e-9                       # Band, nie ρ == 1
    assert K.scalar_gain() == pytest.approx(K.lager_grenze(3), rel=1e-14)       # 2 sin(π/10) = (√5 − 1)/2


def test_schur_cohn_agrees_with_numpy_roots_on_400_random_polynomials():
    rng = SplitMix64(1809)
    mism = 0
    for _ in range(400):
        n = 2 + rng.below(5)
        co = [F(rng.below(19) - 9, 8) for _ in range(n)]
        c = [F(1)] + co
        ro = np.roots([float(v) for v in c])
        near = abs(float(np.max(np.abs(ro))) - 1) < 1e-7
        if not near and X.schur_stable(c) != bool(np.max(np.abs(ro)) < 1 - 1e-9):
            mism += 1
    assert mism == 0
