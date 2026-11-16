"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne dass man das Ergebnis kennt.

Je Eigenschaft 20 bis 200 Zufallsinstanzen mit festem Zufalls-Seed (ganzzahliger SplitMix64), plus Randfälle. Jede Eigenschaft gilt im Modell dieser Demo (Wagen, Q = I, ganzzahliger Verzug, linear)."""

import numpy as np
import pytest

import rvz_dynamics as K
from rvz_rng import SplitMix64


def _uni(rng, lo, hi):
    return lo + (hi - lo) * rng.random()


def test_sum_of_roots_is_two_for_d_at_least_one_and_rho_respects_the_vieta_bound():
    """Koeffizient von z^(d+1) in P_d: Summe der d + 2 Wurzeln = 2 (200 Fälle); ρ ≥ 2/(d + 2) für jede statische Verstärkung (2 000 Zufallsgewinne, d = 1 … 8, 0 Verletzungen)."""
    rng = SplitMix64(1822)
    for _ in range(200):
        d = 1 + rng.below(8)
        k1, k2 = _uni(rng, 0, 2), _uni(rng, 0, 4)
        assert float(np.sum(K.roots_wagen(k1, k2, d)).real) == pytest.approx(2.0, abs=1e-9)
    viol = 0
    for _ in range(2000):
        d = 1 + rng.below(8)
        k1, k2 = _uni(rng, 0, 2), _uni(rng, 0, 4)
        viol += int(K.rho_poly(k1, k2, d) < 2.0 / (d + 2) - 1e-9)
    assert viol == 0


def test_response_is_linear_in_the_initial_deviation_and_in_the_disturbance():
    rng = SplitMix64(5)
    for _ in range(30):
        d = rng.below(5)
        dh = d if rng.below(2) else 0
        k1, k2 = K.gain(10.0 ** _uni(rng, 0, 3))
        a = (_uni(rng, -2, 2), _uni(rng, -2, 2))
        b = (_uni(rng, -2, 2), _uni(rng, -2, 2))
        c = _uni(rng, 0.5, 3)
        xa, _ = K.simulate(k1, k2, d, a, 40, dh=dh)
        xb, _ = K.simulate(k1, k2, d, b, 40, dh=dh)
        xab, _ = K.simulate(k1, k2, d, (a[0] + c * b[0], a[1] + c * b[1]), 40, dh=dh)
        assert np.max(np.abs(xab - (xa + c * xb))) < 1e-9 * max(1.0, float(np.max(np.abs(xab))))
        # Störkraft: Bahn mit w = c w₁ gleicht c-mal der mit w₁ (Start in Ruhe)
        xw1, _ = K.simulate(k1, k2, d, (0, 0), 40, dh=dh, w=1.0)
        xwc, _ = K.simulate(k1, k2, d, (0, 0), 40, dh=dh, w=c)
        assert np.max(np.abs(xwc - c * xw1)) < 1e-9 * max(1.0, float(np.max(np.abs(xwc))))


def test_equilibrium_is_linear_in_w():
    k1, k2 = K.gain(1.0)
    for d in range(5):
        G = K.pred_gain(k1, k2, d)
        assert K.equilibrium_x1(G, d, 2.5) == pytest.approx(2.5 * K.equilibrium_x1(G, d, 1.0), rel=1e-10)


def test_without_delay_the_roots_are_the_eigenvalues_of_a_minus_bk():
    rng = SplitMix64(11)
    for _ in range(50):
        k1, k2 = K.gain(10.0 ** _uniform_log(rng))
        Acl = K.A - K.B0 @ np.array([[k1, k2]])
        assert np.sort_complex(np.roots(K.poly_wagen(k1, k2, 0))) == pytest.approx(np.sort_complex(np.linalg.eigvals(Acl)), abs=1e-9)


def _uniform_log(rng):
    return _uni(rng, -3, 4)


def test_prediction_gives_the_undelayed_trajectory_after_the_blind_periods():
    """Mit dh = d: x_{d+m} der Vorhersage-Schleife gleicht der Bahn ohne Verzug, die bei A^d x₀ startet (20 Fälle, d = 1 … 5)."""
    rng = SplitMix64(23)
    for _ in range(20):
        d = 1 + rng.below(5)
        R = 10.0 ** _uni(rng, -2, 3)
        k1, k2 = K.gain(R)
        x0 = (_uni(rng, -1, 1), _uni(rng, -1, 1))
        xp, _ = K.simulate(k1, k2, d, x0, d + 30, dh=d)
        start = np.linalg.matrix_power(K.A, d) @ np.array(x0)
        xi, _ = K.simulate(k1, k2, 0, start, 30)
        assert np.max(np.abs(xp[d:d + 31] - xi)) < 1e-9


def test_scaling_q_and_r_by_the_same_factor_does_not_change_the_gain():
    rng = SplitMix64(31)
    for _ in range(30):
        R = 10.0 ** _uni(rng, -2, 2)
        c = 10.0 ** _uni(rng, -2, 2)
        k = K.lqr_q(np.eye(2), R)
        kc = K.lqr_q(c * np.eye(2), c * R)
        assert np.max(np.abs(k - kc)) < 1e-8


def test_cost_with_prediction_for_a_position_offset_is_p11_plus_d():
    """x₀ = (1, 0), Q = I: der Wagen bleibt in den blinden Perioden stehen, J = P₁₁ + d (gemessen 3,947 / 4,947 / 5,947 / 6,947 bei R = 1)."""
    for R in (0.1, 1.0, 10.0, 100.0):
        P = K.lqr(R)[0]
        for d in range(0, 7):
            assert K.cost_pred_closed(R, d, (1.0, 0.0)) == pytest.approx(P[0, 0] + d, rel=1e-12)
            k1, k2 = K.gain(R)
            assert K.cost_loop(K.pred_gain(k1, k2, d), d, (1.0, 0.0), R) == pytest.approx(P[0, 0] + d, rel=1e-9)


def test_integer_computation_time_equals_the_integer_model_and_gamma_sums_to_b():
    rng = SplitMix64(41)
    for _ in range(20):
        R = 10.0 ** _uni(rng, -2, 2)
        m = rng.below(4)
        Km = K.lqr(R, "ZOH")[1]
        assert K.rho_eig(K.theta_matrix(Km, float(m))) == pytest.approx(K.rho_eig(K.aug_delay(K.B_ZOH, Km, m)), abs=1e-9)
        tau = rng.random()
        g0, g1 = K.gamma(tau)
        assert np.max(np.abs(g0 + g1 - K.B_ZOH)) < 1e-15


def test_stability_set_in_r_is_one_interval_per_d_on_random_points():
    """ρ(R) kreuzt 1 bei jedem d genau einmal: 2 000 zufällige R = 10^U(−3, 6) mit anderer Verschiebung als das Gitter der Messung, d = 1 … 8, 0 Abweichungen von „stabil genau dann, wenn R > R_krit“."""
    import math

    rng = SplitMix64(1820)
    rk = {d: K.r_krit(d) for d in range(1, 9)}
    mism = checked = 0
    for _ in range(2000):
        lg = _uni(rng, -3, 6)
        k1, k2 = K.gain(10.0 ** lg)
        for d in range(1, 9):
            if abs(lg - math.log10(rk[d])) < 1e-9:
                continue
            checked += 1
            mism += int((K.rho_poly(k1, k2, d) < 1.0) != (lg > math.log10(rk[d])))
    assert checked > 15000 and mism == 0


def test_wrong_assumed_delay_is_unstable_for_r_at_most_one_on_random_cells():
    """2 000 zufällige (R, d, dh) mit R = 10^U(−3, 0), d = 1 … 4, dh ≠ d, |dh − d| ≤ 2: instabil in allen (gemessen: 0 von 30 Rasterzellen stabil)."""
    rng = SplitMix64(1823)
    n = unstable = 0
    while n < 2000:
        R = 10.0 ** _uni(rng, -3, 0)
        d = 1 + rng.below(4)
        dh = d + rng.below(5) - 2
        if dh == d or dh < 0:
            continue
        k1, k2 = K.gain(R)
        unstable += int(K.rho_loop(k1, k2, d, dh) >= 1.0)
        n += 1
    assert unstable == n
