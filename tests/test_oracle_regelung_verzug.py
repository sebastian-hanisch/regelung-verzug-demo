"""Orakel-Tests: jede Kernroutine gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2, §5).

Orakel und Prüfling teilen sich nichts außer den Konstanten A und B:
 - Spektralradius aus den Wurzeln von P_d  <->  Eigenwerte der Zustandsmatrix (Dimension 2 (d + 1)); Stellverzug <-> Messverzug,
 - charakteristisches Polynom der Vorhersage-Schleife exakt (Faddeev–LeVerrier mit Brüchen)  <->  z^(3d) (z² + (k₂ − 2) z + 1 − k₂ + k₁),
 - Gewinn der Vorhersage  <->  DARE des um die Stellgrößen-Vergangenheit erweiterten Systems (SciPy),
 - Γ₀, Γ₁ in geschlossener Form  <->  Van-Loan-Matrixexponential und Zweiteilung der Periode; Rechenzeit θ  <->  Zeitschritt-Simulation mit `expm`,
 - Kosten per Lyapunov  <->  Simulation; Gleichgewicht  <->  Gleichungssystem in Brüchen und Simulation; R_krit  <->  eigene Bisektion über die Zustandsmatrix,
 - Gewinn des N-Schritt-Problems  <->  gestapeltes Kleinste-Quadrate-Problem (nicht die Riccati-Rekursion des Prüflings),
 - Monte-Carlo  <->  Lyapunov (Band 3 %).
Gleitkomma nur mit Toleranz (Eigenwerte, DARE, expm); Gemessenes in der Randspalte der Docstrings."""

from fractions import Fraction as F

import numpy as np
import pytest
import scipy.linalg as la

import rvz_dynamics as K
import rvz_exact as X
from rvz_rng import SplitMix64

A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
BZ = np.array([[0.5], [1.0]])


def _uni(rng, lo, hi):
    return lo + (hi - lo) * rng.random()


def test_rho_from_polynomial_equals_rho_from_state_matrix_800_cases():
    """Gemessen im Scratch-Lauf: größte Abweichung 5,1·10⁻¹⁵ (800 Fälle); hier 200 Fälle, Toleranz 10⁻⁹."""
    rng = SplitMix64(1809)
    worst = 0.0
    for i in range(200):
        d = rng.below(7)
        if i % 2 == 0:
            R = 10.0 ** _uni(rng, -3, 4)
            k1, k2 = K.gain(R)
        else:
            k1, k2 = _uni(rng, 0, 1.5), _uni(rng, 0, 3.0)
        Km = np.array([[k1, k2]])
        worst = max(worst, abs(K.rho_poly(k1, k2, d) - K.rho_eig(K.aug_delay(B, Km, d))))
    assert worst < 1e-9


def test_actuator_delay_has_the_same_root_magnitudes_as_measurement_delay():
    """Stellverzug (Zustand x_k, u_{k−1} … u_{k−d}) gegen Messverzug: Beträge der Wurzeln gleich (gemessen 2,9·10⁻¹⁵ in 28 Fällen)."""
    worst = 0.0
    for R in (10.0, 1.0, 0.1, 0.01):
        Km = K.lqr(R)[1]
        for d in range(7):
            a = np.sort(np.abs(np.roots(K.poly_wagen(float(Km[0, 0]), float(Km[0, 1]), d))))
            b = np.sort(np.abs(np.linalg.eigvals(K.aug_actuator(B, Km, d))))
            m = max(len(a), len(b))
            a = np.concatenate([np.zeros(m - len(a)), a])
            b = np.concatenate([np.zeros(m - len(b)), b])
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < 1e-6                              # Nullwurzeln der Zustandsmatrix rauschen (Band 10⁻⁶)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_prediction_loop_characteristic_polynomial_is_exact(d):
    """Faddeev–LeVerrier mit Brüchen: det(zI − M) = z^(3d) (z² + (k₂ − 2) z + 1 − k₂ + k₁), Matrix 3d + 2; drei Gewinne."""
    for k1, k2 in ((F(2, 5), F(6, 5)), (F(1, 4), F(1, 2)), (F(7, 10), F(9, 5))):
        M = K.loop_matrix(k1, k2, d, d, b=F(1), exact=True)
        assert M.shape == (3 * d + 2, 3 * d + 2)
        assert X.charpoly(M) == [1, k2 - 2, 1 - k2 + k1] + [0] * (3 * d)


def test_prediction_trajectory_equals_the_undelayed_loop_exactly():
    """x_{d+m} = (A − BK)^m A^d x₀ in Brüchen (gemessen 15 von 15; hier d = 1 … 5, zwei Startwerte, m = 0 … 9)."""
    k1, k2 = F(2, 5), F(6, 5)
    Af = np.array([[F(1), F(1)], [F(0), F(1)]], dtype=object)
    Acl = np.array([[F(1), F(1)], [-k1, 1 - k2]], dtype=object)
    for d in range(1, 6):
        for x0 in ((1, 0), (0, 1)):
            xs, _ = X.trajectory(k1, k2, d, x0, d + 10, predict=True)
            start = np.array([F(x0[0]), F(x0[1])], dtype=object)
            for _ in range(d):
                start = Af @ start
            for m in range(10):
                assert list(xs[d + m]) == list(start)
                start = Acl @ start


@pytest.mark.parametrize("R", [10.0, 1.0, 0.1, 0.01])
def test_prediction_gain_equals_dare_gain_of_the_extended_system(R):
    """Zustand (x_k, u_{k−d}, …, u_{k−1}), Schieberegister, Kosten x'Qx + R u²: DARE-Gewinn gleich [K A^d, K A^{d−1} B, …, K B] (gemessen 6,6·10⁻¹³)."""
    Km = K.lqr(R)[1]
    for d in range(1, 7):
        n = 2 + d
        Aa = np.zeros((n, n))
        Aa[:2, :2] = A
        Aa[:2, 2] = B.ravel()
        for i in range(d - 1):
            Aa[2 + i, 2 + i + 1] = 1.0
        Ba = np.zeros((n, 1))
        Ba[-1, 0] = 1.0
        Qa = np.zeros((n, n))
        Qa[:2, :2] = np.eye(2)
        Pa = la.solve_discrete_are(Aa, Ba, Qa, np.array([[R]]))
        Ka = np.linalg.solve(np.array([[R]]) + Ba.T @ Pa @ Ba, Ba.T @ Pa @ Aa)
        want = np.zeros((1, n))
        want[0, :2] = (Km @ np.linalg.matrix_power(A, d)).ravel()
        for j in range(d):
            want[0, 2 + j] = (Km @ np.linalg.matrix_power(A, d - 1 - j) @ B)[0, 0]
        assert np.max(np.abs(Ka - want)) < 1e-8
        # und der Prüfling liefert dieselben Einträge
        G = K.pred_gain(float(Km[0, 0]), float(Km[0, 1]), d)
        assert np.max(np.abs(np.array(G) - want.ravel())) < 1e-12


def test_loop_matrix_equals_the_probing_construction():
    """Die direkt gebaute Zustandsmatrix gleicht der durch Probieren (Einheitsvektoren durch einen Schritt) gebauten; Modellfehler b und falscher Verzug eingeschlossen."""
    rng = SplitMix64(77)

    def probe(G, d, b):
        dh = len(G) - 2
        D = 2 * (d + 1) + dh
        M = np.zeros((D, D))
        for c in range(D):
            z = np.zeros(D)
            z[c] = 1.0
            xs = [z[2 * i:2 * i + 2] for i in range(d + 1)]
            uh = z[2 * (d + 1):]
            u = -(G[0] * xs[d][0] + G[1] * xs[d][1])
            for j in range(dh):
                u -= G[2 + j] * uh[dh - 1 - j]
            x_next = [xs[0][0] + xs[0][1], xs[0][1] + b * u]
            new = list(x_next) + [v for i in range(d) for v in xs[i]]
            if dh > 0:
                new += [u] + list(uh[:dh - 1])
            M[:, c] = new
        return M

    for _ in range(60):
        d, dh = rng.below(6), rng.below(6)
        k1, k2, b = _uni(rng, 0, 1), _uni(rng, 0, 2), _uni(rng, 0.2, 2)
        G = K.pred_gain(k1, k2, dh)
        assert np.max(np.abs(K.loop_from_gain(G, d, b) - probe(G, d, b))) < 1e-12


def test_gamma_matches_van_loan_and_half_period_splitting():
    """Γ₀, Γ₁ gegen Matrixexponential (Van Loan) und gegen die Zweiteilung der Periode, 101 Werte τ (gemessen 1,7·10⁻¹⁶ bzw. 2,2·10⁻¹⁶)."""
    Ac = np.array([[0.0, 1.0], [0.0, 0.0]])

    def vanloan(t):
        M = np.zeros((3, 3))
        M[:2, :2] = Ac
        M[:2, 2] = [0.0, 1.0]
        E = la.expm(M * t)
        return E[:2, :2], E[:2, 2:]

    worst = 0.0
    for tau in np.linspace(0.0, 1.0, 101):
        g0, g1 = K.gamma(float(tau))
        _, G0 = vanloan(1 - tau)
        _, Gfull = vanloan(1.0)
        worst = max(worst, float(np.max(np.abs(g0 - G0))), float(np.max(np.abs(g1 - (Gfull - G0)))))
        Phi1, Ga = vanloan(tau)
        Phi2, Gb = vanloan(1 - tau)
        worst = max(worst, float(np.max(np.abs(Phi2 @ Phi1 - A))), float(np.max(np.abs(Phi2 @ Ga - g1))), float(np.max(np.abs(Gb - g0))))
        assert np.max(np.abs(g0 + g1 - BZ)) < 1e-15            # Γ0 + Γ1 = B für jedes τ
    assert worst < 1e-12


def test_integer_theta_equals_the_integer_model_and_theta_to_m_plus_one():
    for R in (10.0, 1.0, 0.1, 0.01):
        Km = K.lqr(R, "ZOH")[1]
        for m in range(4):
            assert K.rho_eig(K.theta_matrix(Km, float(m))) == pytest.approx(K.rho_eig(K.aug_delay(BZ, Km, m)), abs=1e-9)
            assert K.rho_eig(K.theta_matrix(Km, m + 1 - 1e-9)) == pytest.approx(K.rho_eig(K.aug_delay(BZ, Km, m + 1)), abs=1e-6)


def test_time_step_simulation_agrees_with_the_spectral_radius_for_fractional_theta():
    """Zeitschritt-Simulation über Teilintervalle mit expm (ohne die Γ-Formeln) gegen ρ; gemessen 4,9·10⁻⁴ relativ in 17 Fällen, Band 5·10⁻³."""
    worst, n = 0.0, 0
    for R in (10.0, 1.0, 0.1):
        Km = K.lqr(R, "ZOH")[1]
        for theta in (0.0, 0.2, 0.5, 0.8, 1.0, 1.3):
            r = K.rho_eig(K.theta_matrix(Km, theta))
            if r > 1.5 or r < 0.3:
                continue
            N = 300 if r >= 1 else int(min(600, max(120, 150 / max(-np.log10(r), 1e-3))))
            worst = max(worst, abs(K.simulate_theta_growth(Km, theta, N) - r) / r)
            n += 1
    assert n >= 15 and worst < 5e-3


def test_cost_by_lyapunov_equals_cost_by_simulation():
    """Σ x'x + R u² per Lyapunov gegen die aufsummierte Simulation (gemessen 2,8·10⁻¹⁵ ohne, 9,4·10⁻¹⁶ mit Vorhersage); relativ 10⁻⁹."""
    cases = [(R, d, pred) for R in (1.0, 10.0, 100.0) for d in (1, 2, 3) for pred in (False, True) if pred or K.rho_design(R, d) < 0.99]
    assert len(cases) >= 12
    for R, d, pred in cases:
        k1, k2 = K.gain(R)
        dh = d if pred else 0
        for x0 in ((1.0, 0.0), (0.0, 1.0)):
            xs, us = K.simulate(k1, k2, d, x0, 4000, dh=dh)
            j_sim = float(np.sum(xs[:-1] ** 2) + R * np.sum(us ** 2))
            j_lyap = K.cost_loop(K.pred_gain(k1, k2, dh), d, x0, R)
            assert j_lyap == pytest.approx(j_sim, rel=1e-9)
            if pred:
                assert K.cost_pred_closed(R, d, x0) == pytest.approx(j_sim, rel=1e-9)


def test_equilibrium_matches_exact_linear_system_and_simulation():
    """x₁* mit Vorhersage (w/k₁)(1 + k₂ d + k₁ d(d−1)/2): Gleichungssystem in Brüchen (21 von 21 gemessen), für den DARE-Gewinn gegen Simulation (1,4·10⁻¹⁴)."""
    for k1, k2 in ((F(2, 5), F(6, 5)), (F(1, 4), F(1, 2)), (F(7, 10), F(9, 5))):
        for d in range(7):
            assert X.equilibrium_x1(k1, k2, d, True) == (1 / k1) * (1 + k2 * d + k1 * F(d * (d - 1), 2))
        for d in range(5):
            assert X.equilibrium_x1(k1, k2, d, False) == 1 / k1
    k1, k2 = K.gain(1.0)
    for d in range(0, 5):
        xs, _ = K.simulate(k1, k2, d, (0.0, 0.0), 6000, dh=d, w=1.0)
        assert xs[-1, 0] == pytest.approx(K.equilibrium_formula(k1, k2, d, 1.0), rel=1e-9)
        assert xs[-1, 1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_r_krit_equals_an_independent_bisection_over_the_state_matrix(d):
    lo, hi = 1e-3, 1e7
    for _ in range(100):
        mid = (lo * hi) ** 0.5
        Km = K.lqr(mid)[1]
        if K.rho_eig(K.aug_delay(B, Km, d)) < 1.0:
            hi = mid
        else:
            lo = mid
    assert K.r_krit(d) == pytest.approx(hi, rel=1e-6)


def _lifted_gain(R, N, Pf):
    """Gewinn der ersten Stufe des unbeschränkten N-Schritt-Problems als gestapeltes Kleinste-Quadrate-Problem (ohne Riccati-Rekursion)."""
    Phi = np.vstack([np.linalg.matrix_power(A, k) for k in range(1, N + 1)])
    Gam = np.zeros((2 * N, N))
    for k in range(1, N + 1):
        for j in range(k):
            Gam[2 * (k - 1):2 * k, j:j + 1] = np.linalg.matrix_power(A, k - 1 - j) @ B
    Qbar = np.zeros((2 * N, 2 * N))
    for k in range(1, N):
        Qbar[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = np.eye(2)
    Qbar[2 * (N - 1):, 2 * (N - 1):] = Pf
    H = Gam.T @ Qbar @ Gam + R * np.eye(N)
    return np.linalg.solve(H, Gam.T @ Qbar @ Phi)[0:1, :]


@pytest.mark.parametrize("R", [1.0, 0.1])
def test_first_stage_gain_of_the_n_step_problem_equals_k_for_dare_terminal_cost(R):
    """Terminalkosten P_f = DARE: K_N = K für N = 1 … 12 (gemessen 4,4·10⁻¹⁶ bei R = 1); mit P_f = Q nicht (0,744 / 0,244 / 0,0006 bei N = 1 / 2 / 5)."""
    P, Km = K.lqr(R)
    for N in range(1, 13):
        assert np.max(np.abs(_lifted_gain(R, N, P) - Km)) < 1e-9                        # u_0 = −K_N x_0 mit K_N aus dem Normalgleichungssystem
    dev = [float(np.max(np.abs(_lifted_gain(R, N, np.eye(2)) - Km))) for N in (1, 2, 5, 12)]
    assert dev[0] > 0.5 and dev[1] > 0.1 and dev[2] < 2e-3 and dev[3] < 1e-5
    for N in (1, 2, 5, 12):
        assert np.max(np.abs(K.riccati_gain_n(R, N, "q") - _lifted_gain(R, N, np.eye(2)))) < 1e-9
        assert np.max(np.abs(K.riccati_gain_n(R, N, "dare") - Km)) < 1e-9


def test_monte_carlo_equals_the_lyapunov_cost_within_a_3_percent_band():
    """d = 2, Vorhersage, R = 1, Zufalls-Seed 1814, 40 000 Perioden (Scratch-Lauf mit 400 000: 0,28 %); Band 3 % (K.monte_carlo_cost nutzt den ganzzahligen SplitMix64)."""
    k1, k2 = K.gain(1.0)
    G = K.pred_gain(k1, k2, 2)
    mc = K.monte_carlo_cost(G, 2, 1.0, 40000, 1814)
    assert mc == pytest.approx(K.stationary_cost(G, 2, 1.0), rel=0.03)


def test_scalar_dare_equals_the_golden_ratio():
    p = float(la.solve_discrete_are(np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]))[0, 0])
    assert p == pytest.approx((1 + 5 ** 0.5) / 2, abs=1e-12)
    assert K.scalar_gain() == pytest.approx(p / (1 + p), abs=1e-12)


def test_scalar_polynomial_has_the_roots_of_the_lager_recurrence():
    """Lager ohne Leitung: i_{t+1} = i_t − α i_{t+1−L}; Begleitmatrix der Rekursion gegen die Wurzeln von z^(d+1) − z^d + K (L = d + 1, α = K)."""
    Kg = K.scalar_gain()
    for d in range(6):
        L = d + 1
        C = np.zeros((L, L))
        C[0, 0] = 1.0
        C[0, L - 1] -= Kg
        for i in range(1, L):
            C[i, i - 1] = 1.0
        assert np.sort(np.abs(np.linalg.eigvals(C))) == pytest.approx(np.sort(np.abs(np.roots(K.scalar_poly(Kg, d)))), abs=1e-9)
