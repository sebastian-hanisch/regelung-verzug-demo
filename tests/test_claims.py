"""Claims-Tests: jede Zahl der Befunde B1 bis B16 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: Wagen, Q = I, B = [0, 1]ᵀ (B3-Gegenprobe und B7: B = [½, 1]ᵀ), ohne Stellgrenzen, bekannter ganzzahliger Verzug. Zahlen, die auf anderen Plattformen anders ausfallen können
(Eigenwerte, DARE, Nelder–Mead, Monte-Carlo), stehen als Band; Zählungen, die an keiner Schwelle knapp liegen, mit Toleranz ± 2; exakt nur, was in Brüchen gerechnet ist.
Die vorgerechneten Zahlen (`data/rvz_frozen.json`) werden auf Konsistenz mit ihrer Erzeugung (kleinere Neu-Rechnung) und mit der README geprüft.

Zahl → Test (Auszug der README-Tabelle): B1 test_b1_*, B2 test_b2_*, B3 test_b3_*, B4 test_b4_*, B5 test_b5_*, B6 test_b6_*, B7 test_b7_*, B8 test_b8_*, B9 test_b9_*, B10 test_b10_*, B11 test_b11_*,
B12 test_b12_*, B13 test_b13_*, B14 test_b14_*, B15 test_b15_*, B16 test_b16_*. Was nicht im Test steht, ist in der README als „Messung“, „frühere Messung“ oder „früherer Rechenlauf“ gekennzeichnet."""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from rvz_rng import SplitMix64
import make_frozen
import rvz_constants as C
import rvz_dynamics as K
import rvz_evaluation as ev
import rvz_exact as X
import rvz_presets as P

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads((ROOT / "data" / "rvz_frozen.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
NB = " "


def _has(*numbers, text=None):
    flat = (README if text is None else text).replace(NB, " ").replace("\\*", "*")
    for n in numbers:
        assert n in flat, n


def _rho(R, d):
    k1, k2 = K.gain(R)
    return K.rho_poly(k1, k2, d)


# --- B1 --------------------------------------------------------------------------------------------------------------------------------

def test_b1_spectral_radius_against_d_and_the_stable_undelayed_grid():
    want = {1000.0: (0.881, 0.857, 0.858, 0.951), 100.0: (0.795, 0.764, 0.987, 1.072), 10.0: (0.650, 0.961, 1.160, 1.211), 1.0: (0.422, 1.193, 1.329, 1.339),
            0.1: (0.362, 1.321, 1.416, 1.403)}
    for R, vals in want.items():
        for d, v in enumerate(vals):
            assert _rho(R, d) == pytest.approx(v, abs=6e-4), (R, d)
    for R, v in {0.01: 1.346, 0.001: 1.349}.items():
        assert _rho(R, 1) == pytest.approx(v, abs=6e-4)
    assert all(_rho(10.0 ** lg, 0) < 1.0 for lg in np.linspace(-3, 6, 4001))
    _has("0,422 / 1,193 / 1,329 / 1,339", "0,362 / 1,321 / 1,416 / 1,403", "0,650 / 0,961 / 1,160 / 1,211", "0,795 / 0,764 / 0,987 / 1,072", "0,881 / 0,857 / 0,858 / 0,951",
         "1,193 / 1,321 / 1,346 / 1,349", "0,422 / 0,362 / 0,380 / 0,382", "alle 4 001 Gitterpunkte R = 10⁻³ bis 10⁶ stabil")
    assert [round(_rho(R, 0), 3) for R in (1.0, 0.1, 0.01, 0.001)] == [0.422, 0.362, 0.380, 0.382]


# --- B2 --------------------------------------------------------------------------------------------------------------------------------

def test_b2_largest_stable_d_per_r():
    expected = {0.001: 0, 0.01: 0, 0.1: 0, 1.0: 0, 3.0: 0, 10.0: 1, 30.0: 1, 100.0: 2, 300.0: 2, 1000.0: 3, 3000.0: 5, 10000.0: 7, 100000.0: 12}
    by_R = FROZEN["largest_stable_d"]["by_R"]
    assert {float(R): v["largest_d_without_gap"] for R, v in by_R.items()} == expected
    assert all(v["stable_total"] == v["largest_d_without_gap"] + 1 for v in by_R.values()) and len(by_R) == 13 and FROZEN["largest_stable_d"]["dmax"] == 119
    for R in (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0):
        k1, k2 = K.gain(R)
        stable = [K.rho_poly(k1, k2, d) < 1 - 1e-9 for d in range(41)]
        n = 0
        while n < 40 and stable[n + 1]:
            n += 1
        assert n == expected[R] and sum(stable) == n + 1, R                  # keine Lücke bis d = 40
    _has("R = 1 und kleiner sowie R = 3: d = 0", "R = 10 und 30: 1", "R = 100 und 300: 2", "R = 1 000: 3", "R = 3 000: 5", "R = 10 000: 7", "R = 100 000: 12", "In allen 13 gemessenen R", "bis d = 119")


# --- B3 --------------------------------------------------------------------------------------------------------------------------------

def test_b3_critical_r_one_crossing_and_the_second_model():
    krit = {1: 6.854, 2: 82.79, 3: 369.9, 4: 1092.0, 5: 2555.0, 6: 5147.0, 7: 9336.0, 8: 15673.0}
    for d, v in krit.items():
        assert K.r_krit(d) == pytest.approx(v, rel=1e-3), d
        assert FROZEN["r_krit"][str(d)] == pytest.approx(K.r_krit(d), rel=1e-9)
    grid = make_frozen.r_grid_crossings()                                    # 4 001 Punkte, neu gerechnet
    assert grid == FROZEN["r_grid"] and all(v["changes"] == 1 for v in grid["by_d"].values()) and len(grid["by_d"]) == 8 and grid["points"] == 4001
    rnd = make_frozen.r_random_crossings(2000)                               # 2 000 Zufallspunkte neu gerechnet, andere Verschiebung als das Gitter
    assert rnd["mismatches"] == 0 and rnd["checked"] > 15000
    fz = FROZEN["r_random"]
    assert fz["mismatches"] == 0 and fz["n"] == 20000 and fz["checked"] == 160000 and fz["seed"] == 1820
    assert K.r_krit(1, "ZOH") == pytest.approx(6.888, abs=1e-3) and K.r_krit(2, "ZOH") == pytest.approx(82.71, abs=2e-2)
    zoh_d1 = [K.rho_eig(K.aug_delay(K.B_ZOH, K.lqr(R, "ZOH")[1], 1)) for R in (1.0, 0.1, 0.01, 0.001)]
    assert zoh_d1 == pytest.approx([1.198, 1.339, 1.370, 1.374], abs=1e-3) and all(v > 1.0 for v in zoh_d1)
    _has("1,198 / 1,339 / 1,370 / 1,374 (instabil)")
    _has("6,854 / 82,79 / 369,9 / 1 092 / 2 555 / 5 147 / 9 336 / 15 673", "in 8 von 8 d", "160 000 Zufallsprüfungen (20 000 zufällige R, d = 1 bis 8, Zufalls-Seed 1820): 0 Abweichungen",
         "6,888 (d = 1) und 82,71 (d = 2)", "Genau ein Wechsel instabil → stabil im Gitter von 4 001 Punkten")


# --- B4 --------------------------------------------------------------------------------------------------------------------------------

def test_b4_random_q_counts_are_reproduced_and_stand_in_the_readme():
    for kind, want, seed in (("diag", 1702, 1810), ("voll", 1725, 1811)):
        live = make_frozen.m5_counts(kind)
        fz = FROZEN["m5_" + kind]
        assert fz["stable"] == want and fz["total"] == 4000 and fz["seed"] == seed and fz["skipped"] == 0
        assert abs(live["stable"] - want) <= 2 and live["total"] == 4000          # Band ± 2 (Zählung aus Neu-Rechnung)
        for dec, (s, t) in fz["by_decade"].items():
            assert abs(live["by_decade"][dec][0] - s) <= 2 and live["by_decade"][dec][1] == t
    assert FROZEN["m5_diag"]["by_decade"] == {"-3": [0, 574], "-2": [0, 571], "-1": [63, 601], "0": [231, 583], "1": [342, 546], "2": [520, 579], "3": [546, 546]}
    dg, vl = FROZEN["m5_diag"]["by_decade"], FROZEN["m5_voll"]["by_decade"]
    assert dg["-3"][0] + dg["-2"][0] == 0 and dg["-3"][1] + dg["-2"][1] == 1145       # unter R = 10⁻¹ bei diagonalem Q: 0 von 1 145
    assert vl["-3"][0] + vl["-2"][0] == 1 and vl["-3"][1] + vl["-2"][1] == 1199       # bei vollem Q: 1 von 1 199
    shares = [100 * 1702 / 4000, 100 * 1725 / 4000, 100 * 1722 / 4000, 100 * 1761 / 3999]               # 42,55 / 43,125 / 43,05 / 44,04 %
    assert shares == pytest.approx([42.6, 43.1, 43.1, 44.0], abs=0.06) and 43.0 < min(shares) + 0.5 and max(shares) < 44.1 and min(shares) > 42.5
    _has("**1 702 von 4 000**", "**1 725 von 4 000**", "10⁻³: 0/574, 10⁻²: 0/571, 10⁻¹: 63/601, 10⁰: 231/583, 10¹: 342/546, 10²: 520/579, 10³: 546/546", "Zufalls-Seed 1810 bzw. 1811",
         "1 722 von 4 000", "1 761 von 3 999", "rund 43 bis 44 %", "42,6 und 43,1 % im Bau, 43,1 und 44,0 % in der früheren Messung", "bei diagonalem Q unter R = 10⁻¹ nie",
         "in keinem der 1 145 Entwürfe (0/574 und 0/571); bei vollem Q in 1 von 1 199", "einer 2 × 2-Matrix G aus vier unabhängigen Normalzahlen N(0, 1)")
    retraction = "die früher genannten „43 bis 45 %“ stimmten nicht, 45 % erreicht keine Zählung"
    assert retraction in README and "45 %" not in README.replace(retraction, "")                       # 45 % steht nur als zurückgenommene Angabe da
    assert "1 702 von 4 000" in ev.m5_text() and "1 725 von 4 000" in ev.m5_text()


# --- B5 --------------------------------------------------------------------------------------------------------------------------------

def test_b5_actuator_delay_equals_measurement_delay():
    worst, n = 0.0, 0
    for R in (10.0, 1.0, 0.1, 0.01):
        Km = K.lqr(R)[1]
        for d in range(7):
            a = np.sort(np.abs(np.roots(K.poly_wagen(float(Km[0, 0]), float(Km[0, 1]), d))))
            b = np.sort(np.abs(np.linalg.eigvals(K.aug_actuator(K.B0, Km, d))))
            m = max(len(a), len(b))
            worst = max(worst, float(np.max(np.abs(np.concatenate([np.zeros(m - len(a)), a]) - np.concatenate([np.zeros(m - len(b)), b])))))
            n += 1
    assert n == 28 and worst < 1e-6
    _has("28 Fälle: R = 10, 1, 0,1, 0,01; d = 0 bis 6", "2,9·10⁻¹⁵", "Band 10⁻⁶")


# --- B6 --------------------------------------------------------------------------------------------------------------------------------

def test_b6_scalar_case():
    Kg = K.scalar_gain()
    assert Kg == pytest.approx(0.618034, abs=1e-6)
    assert [round(K.scalar_rho(Kg, d), 3) for d in range(6)] == [0.382, 0.786, 1.0, 1.070, 1.097, 1.108]
    assert abs(K.scalar_rho(Kg, 2) - 1.0) < 1e-9 and Kg == pytest.approx(2 * math.sin(math.pi / 10), rel=1e-14)
    assert [X.scalar_stable(F(3, 5), d) for d in range(5)] == [True, True, True, False, False] and X.scalar_stable(F(5, 8), 2) is False
    _has("0,382 / 0,786 / 1,000 / 1,070 / 1,097 / 1,108", "z^(d+1) − z^d + K", "L = d + 1 und α = K", "2 sin(π/10) = (√5 − 1)/2", "ρ liegt um weniger als 10⁻⁹ neben 1",
         "K = 3/5 stabil für d = 0, 1, 2 und instabil für d = 3, 4", "K = 5/8 bei d = 2 instabil", "keine Erklärung")


# --- B7 --------------------------------------------------------------------------------------------------------------------------------

def test_b7_critical_computation_time_and_the_grid_checks():
    ts = {100.0: 2.114, 10.0: 1.130, 3.0: 0.800, 1.0: 0.621, 0.1: 0.464, 0.01: 0.437, 0.001: 0.434}
    for R, v in ts.items():
        assert K.theta_krit(R) == pytest.approx(v, abs=6e-4), R
    assert [round(K.rho_theta(1.0, t), 3) for t in (0.0, 0.25, 0.5, 0.75, 1.0)] == pytest.approx([0.434, 0.693, 0.918, 1.076, 1.198], abs=6e-4)
    assert K.rho_theta(0.1, 0.5) == pytest.approx(1.031, abs=6e-4) and K.rho_theta(1.0, 0.5) < 1.0 < K.rho_theta(0.1, 0.5)
    for R, ang in ((10.0, 41.5), (1.0, 59.2), (0.1, 69.3)):
        assert ev.theta_info(R, 0.5)["angle"] == pytest.approx(ang, abs=0.06)
    live = make_frozen.theta_crossings(n_R=40, per_R=10)                       # Gitter 3 001 Punkte (7 R) und 400 Zufallspunkte neu gerechnet
    assert live["grid_changes_by_R"] == FROZEN["theta"]["grid_changes_by_R"] and set(live["grid_changes_by_R"].values()) == {1} and len(live["grid_changes_by_R"]) == 7
    assert live["random"]["mismatches"] == 0 and live["random"]["checked"] > 350
    fz = FROZEN["theta"]["random"]
    assert fz["mismatches"] == 0 and fz["checked"] == 5000 and fz["n_R"] == 500 and fz["seed"] == 1821
    assert fz["per_R"] == 10 and fz["n_R"] * fz["per_R"] == fz["checked"]
    _has("2,114 / 1,130 / 0,800 / 0,621 / 0,464 / 0,437 / 0,434 Perioden für R = 100 / 10 / 3 / 1 / 0,1 / 0,01 / 0,001", "in 7 von 7 R genau ein Wechsel",
         "5 000 Zufallspunkte (500 zufällige R = 10^U(−3, 2) mit je 10 gleichverteilt gezogenen θ zwischen 0 und 3, Zufalls-Seed 1821): 0 Abweichungen",
         "0,434 / 0,693 / 0,918 / 1,076 / 1,198", "hält R = 1 (ρ = 0,918) und R = 0,1 nicht (1,031)", "41,5° (R = 10), 59,2° (R = 1), 69,3° (R = 0,1)")


# --- B8 --------------------------------------------------------------------------------------------------------------------------------

def test_b8_prediction_is_exact_in_fractions():
    k1, k2 = F(2, 5), F(6, 5)
    ok = 0
    Af = np.array([[F(1), F(1)], [F(0), F(1)]], dtype=object)
    Acl = np.array([[F(1), F(1)], [-k1, 1 - k2]], dtype=object)
    for d in range(1, 6):
        for x0 in ((1, 0), (0, 1)):
            xs, _ = X.trajectory(k1, k2, d, x0, d + 10, predict=True)
            z = np.array([F(x0[0]), F(x0[1])], dtype=object)
            for _ in range(d):
                z = Af @ z
            good = True
            for m in range(10):
                good = good and list(xs[d + m]) == list(z)
                z = Acl @ z
            ok += int(good)
    assert ok == 10
    ok = 0
    for d in range(1, 5):
        for kk1, kk2 in ((F(2, 5), F(6, 5)), (F(1, 4), F(1, 2)), (F(7, 10), F(9, 5))):
            M = K.loop_matrix(kk1, kk2, d, d, b=F(1), exact=True)
            ok += int(X.charpoly(M) == [1, kk2 - 2, 1 - kk2 + kk1] + [0] * (3 * d))
    assert ok == 12
    _has("10 von 10", "früherer Rechenlauf: 15 von 15 mit drei Startwerten", "12 von 12 (d = 1 bis 4, drei Gewinne", "15 von 15 bis 17 × 17", "z^(3d) (z² + (k₂ − 2) z + 1 − k₂ + k₁)")


# --- B9 --------------------------------------------------------------------------------------------------------------------------------

def test_b9_prediction_gain_equals_the_extended_dare_gain():
    import scipy.linalg as la

    A, B = K.A, K.B0
    worst = 0.0
    for R in (10.0, 1.0, 0.1, 0.01):
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
            G = np.array(K.pred_gain(float(Km[0, 0]), float(Km[0, 1]), d))
            worst = max(worst, float(np.max(np.abs(Ka.ravel() - G))))
    assert worst < 1e-8
    # 2 100 zufällige Störungen des Gewinns (700 je d = 1 bis 3, R = 1, Komponenten mal 1 + 0,1·N(0, 1), Zufalls-Seed 1826): keine senkt die stationären Kosten
    rng = SplitMix64(1826)
    k1, k2 = K.gain(1.0)
    total = stable = lower = 0
    for d in (1, 2, 3):
        G0 = K.pred_gain(k1, k2, d)
        J0 = K.stationary_cost(G0, d, 1.0)
        for _ in range(700):
            G = [g * (1.0 + 0.1 * rng.normal()) for g in G0]
            total += 1
            if K.rho_eig(K.loop_from_gain(G, d, 1.0)) < 1.0 - 1e-9:
                stable += 1
                lower += int(K.stationary_cost(G, d, 1.0) < J0 * (1 - 1e-12))
    assert total == 2100 and lower == 0 and abs(stable - 2002) <= 3
    _has("R = 10, 1, 0,1, 0,01 und d = 1 bis 6, größte Abweichung unter 10⁻⁸ (früherer Rechenlauf 6,6·10⁻¹³)", "2 100 zufälligen Störungen des Gewinns die stationären Kosten (700 je d = 1 bis 3;",
         "1 + 0,1·N(0, 1) multipliziert, ganzzahliger SplitMix64, Zufalls-Seed 1826; 2 002 der 2 100 gestörten Schleifen sind stabil")


# --- B10 -------------------------------------------------------------------------------------------------------------------------------

def test_b10_mpc_bridge():
    for R in (1.0, 0.1):
        K1 = K.lqr(R)[1]
        assert max(float(np.max(np.abs(K.riccati_gain_n(R, N) - K1))) for N in range(1, 13)) < 1e-9
    dev = [float(np.max(np.abs(K.riccati_gain_n(1.0, N, "q") - K.lqr(1.0)[1]))) for N in (1, 2, 5, 12)]
    assert dev[0] == pytest.approx(0.744, abs=5e-4) and dev[1] == pytest.approx(0.244, abs=5e-4) and dev[2] == pytest.approx(0.0006, abs=5e-5) and dev[3] < 1e-5
    fz = FROZEN["m9"]
    assert fz["1.0"]["max_dev_dare_terminal"] < 1e-9 and fz["0.1"]["max_dev_dare_terminal"] < 1e-9 and fz["1.0"]["dev_q_terminal"]["12"] < 1e-5
    _has("N = 1 bis 12 (Abweichung unter 10⁻⁹; Messung 4,4·10⁻¹⁶ bei R = 1, 6,7·10⁻¹⁶ bei R = 0,1)", "0,744 / 0,244 / 0,0006 ab (N = 1 / 2 / 5, R = 1)", "für N = 12 um weniger als 10⁻⁵",
         "seine erste Stufe auf den verspäteten Zustand an, ist es in diesem Modell die Schleife u_k = −K x_(k−d) (Modellierungsannahme dieser Demo, nicht gemessen)")


# --- B11 -------------------------------------------------------------------------------------------------------------------------------

def test_b11_model_error_in_the_input_gain():
    want = {(1.0, 0): (0.01, 1.936), (1.0, 1): (0.335, 1.531), (1.0, 2): (0.536, 1.383), (1.0, 3): (0.659, 1.319), (1.0, 4): (0.728, 1.286), (1.0, 5): (0.769, 1.250),
            (0.1, 0): (0.01, 1.592), (0.1, 1): (0.595, 1.405), (0.1, 5): (0.822, 1.179)}
    for (R, d), (lo, hi) in want.items():
        band = ev.tolerance_bands(R)[d]
        assert band["bmin"] == pytest.approx(lo, abs=1.5e-3) and band["bmax"] == pytest.approx(hi, abs=1.5e-3), (R, d)
        assert band["runs"] == 1                                              # genau ein stabiles Intervall
        assert band["edge"] == (d == 0)                                       # nur bei d = 0 bis zum Gitterrand 0,01 stabil
    widths = [ev.tolerance_bands(1.0)[d]["bmax"] - ev.tolerance_bands(1.0)[d]["bmin"] for d in range(1, 6)]
    assert all(widths[i + 1] < widths[i] for i in range(4))
    live = make_frozen.tolerance_random(n_R=8)
    assert live["falling"] == 8 and FROZEN["tolerance_random"] == {"n_R": 60, "falling": 60, "seed": 1824}
    _has("d = 0: bis 1,936 (untere Grenze nicht erreicht: bis zum Gitterrand 0,01 stabil)", "d = 1: 0,335 bis 1,531", "d = 2: 0,536 bis 1,383", "d = 3: 0,659 bis 1,319", "d = 4: 0,728 bis 1,286",
         "d = 5: 0,769 bis 1,250", "d = 0: bis 1,592; d = 1: 0,595 bis 1,405; d = 5: 0,822 bis 1,179", "60 zufälligen R = 10^U(−2, 1) (Zufalls-Seed 1824)", "von 0,01 bis 1,94 bei d = 0", "von 0,77 bis 1,25 bei d = 5")


# --- B12 -------------------------------------------------------------------------------------------------------------------------------

def test_b12_wrong_assumed_delay():
    stable = {R: ev.dh_map(R)["stable_off_diagonal"] for R in (1000.0, 100.0, 10.0, 1.0, 0.1)}
    assert stable == {1000.0: 15, 100.0: 8, 10.0: 1, 1.0: 0, 0.1: 0} and sum(stable.values()) == 24
    assert all(ev.dh_map(R)["cells_off_diagonal"] == 15 for R in stable)
    k1, k2 = K.gain(1.0)
    assert [round(K.rho_loop(k1, k2, 1, dh), 3) for dh in (0, 2, 3)] == [1.193, 1.671, 1.855]
    live = make_frozen.falsch_dh_random(300)
    assert live["unstable"] == 300 and FROZEN["falsch_dh"] == {"n": 2000, "unstable": 2000, "seed": 1823}
    draws = list(make_frozen.falsch_dh_draws(2000))                           # der Bereich von dh: dh = d − 2 bis d + 2, dh ≠ d, dh ≥ 0 (Irrtum um höchstens 2 Perioden)
    assert len(draws) == 2000 and {dh - d for _, d, dh in draws} == {-2, -1, 1, 2} and min(dh for _, _, dh in draws) >= 0
    assert {d for _, d, _ in draws} == {1, 2, 3, 4} and max(R for R, _, _ in draws) <= 1.0 and min(R for R, _, _ in draws) >= 1e-3
    _has("stabil in **24 von 75** Zellen", "R = 1 000: 15, R = 100: 8, R = 10: 1, **R = 1: 0, R = 0,1: 0**", "dh = 0 (keine Vorhersage) 1,193, dh = 2: 1,671, dh = 3: 1,855",
         "dh gleichverteilt von d − 2 bis d + 2 (dh ≠ d, dh ≥ 0, also der angenommene Verzug um höchstens 2 Perioden daneben): **in 2 000 von 2 000 instabil** (Zufalls-Seed 1823)",
         "in 2 000 von 2 000 zufälligen Fällen instabil", "(um höchstens 2 Perioden daneben)")
    _has("(um höchstens 2 Perioden daneben) und R ≤ 1 ist die Vorhersage in 2 000 von 2 000 Zufallszellen instabil (Experiment 5)", text=APP)


# --- B13 -------------------------------------------------------------------------------------------------------------------------------

def test_b13_best_static_gain_and_the_vieta_bound():
    rho = {int(d): v for d, v in FROZEN["static_best"]["rho"].items()}
    for d, v in {1: 0.66667, 2: 0.78868, 3: 0.84495, 4: 0.87749, 5: 0.89871, 6: 0.91366}.items():
        assert rho[d]["rho"] == pytest.approx(v, abs=1e-3)                    # Nelder–Mead: nur als Band 10⁻³
    for d in range(1, 7):
        assert rho[d]["rho"] >= 2.0 / (d + 2) - 1e-9                          # Absolutaussage: nicht kleiner als 2/(d + 2)
        assert K.rho_poly(rho[d]["k1"], rho[d]["k2"], d) == pytest.approx(rho[d]["rho"], abs=1e-6)     # mehrfache Wurzeln: np.roots streut je Plattform um bis 10⁻⁸ (Linux-Check 2026-11-16: 6·10⁻⁹ bei d = 3)
    for d in (1, 2, 3):                                                       # Neu-Rechnung (Gitter + Nelder–Mead) als Band 10⁻³
        assert make_frozen.best_rho(d)["rho"] == pytest.approx(rho[d]["rho"], abs=1e-3)
    assert X.schur_stable(K.poly_wagen(F(1, 27), F(1, 3), 1)) and K.poly_wagen(F(1, 27), F(1, 3), 1) == [1, -2, F(4, 3), F(-8, 27)]
    assert rho[1]["rho"] == pytest.approx(2 / 3, abs=1e-4) and rho[1]["k1"] == pytest.approx(1 / 27, abs=1e-3) and rho[1]["k2"] == pytest.approx(1 / 3, abs=1e-3)
    assert FROZEN["vieta"]["violations"] == 0 and FROZEN["vieta"]["n"] == 20000 and FROZEN["vieta"]["seed"] == 1822 and FROZEN["vieta"]["max_sum_dev"] < 1e-9
    stat_b = {r["d"]: (r["bmax"], r["rho"]) for r in ev.static_b_rows()}
    assert [round(stat_b[d][0], 2) for d in range(1, 5)] == [2.95, 3.07, 3.10, 3.11]
    _has("0,66667 bei K = (1/27, 1/3), **dreifache Wurzel 2/3, die Schranke genau erreicht (exakt in Brüchen)**", "d = 2: 0,78868; 3: 0,84495; 4: 0,87749; 5: 0,89871; 6: 0,91366",
         "20 000 Zufallsgewinne ohne Verletzung, Zufalls-Seed 1822", "bis b = 2,95 / 3,07 / 3,10 / 3,11 (ρ = 0,667 / 0,789 / 0,845 / 0,877)", "(A − BK)² = 0",
         "nicht erreichbar** (ohne Verzug mit K = (1, 2)", "auf dem verspäteten Zustand allein (ohne Vorhersage)", "für K = (1, 2) ist das charakteristische Polynom der Schleife z^(3d+2), exakt in Brüchen für d = 1 bis 5 geprüft")
    assert np.max(np.abs(np.linalg.matrix_power(K.A - K.B0 @ np.array([[1.0, 2.0]]), 2))) == 0.0     # Deadbeat ohne Verzug: exakt, ganze Zahlen
    for d in range(1, 6):                                                    # mit Vorhersage und K = (1, 2): Polynom der Schleife exakt z^(3d+2), also Deadbeat (nur numerisch ρ > 0)
        M = K.loop_matrix(F(1), F(2), d, d, b=F(1), exact=True)
        assert X.charpoly(M) == [1] + [0] * (3 * d + 2)
        assert K.rho_loop(1.0, 2.0, d, d) < 0.1                               # Band: schlecht konditionierte Mehrfachwurzel (gemessen 2,6·10⁻⁸ / 2,4·10⁻³ / 1,6·10⁻² / 3,0·10⁻² / 6,6·10⁻²)
    assert all(K.rho_poly(1.0, 2.0, d) >= 2.0 / (d + 2) - 1e-9 for d in range(1, 6))        # ohne Vorhersage mit Verzug: Schranke 2/(d + 2)


# --- B14 -------------------------------------------------------------------------------------------------------------------------------

def test_b14_steady_state_error():
    k1, k2 = K.gain(1.0)
    pred = [K.equilibrium_x1(K.pred_gain(k1, k2, d), d, 1.0) for d in range(6)]
    assert pred == pytest.approx([2.369, 5.316, 9.264, 14.21, 20.16, 27.10], abs=6e-3)
    assert [pred[d] / pred[0] for d in range(6)] == pytest.approx([1, 2.24, 3.91, 6.00, 8.51, 11.44], abs=6e-3)
    for d in range(6):
        assert pred[d] == pytest.approx(K.equilibrium_formula(k1, k2, d, 1.0), rel=1e-12)
    static = {d: 1.0 / v["k1"] for d, v in ((int(d), v) for d, v in FROZEN["static_best"]["rho"].items())}
    assert [static[d] for d in range(1, 6)] == pytest.approx([27.0, 77.6, 153.0, 255.0, 381.0], rel=4e-3)      # Nelder–Mead: Band
    soft = [1.0 / K.gain(1.2 * K.r_krit(d))[0] for d in range(1, 6)]
    assert soft == pytest.approx([4.53, 12.6, 24.6, 40.8, 60.9], abs=0.06)
    assert pred[1] > soft[0] and all(pred[d] < soft[d - 1] for d in range(2, 6))        # d = 1 größer als der weiche Entwurf, ab d = 2 kleiner
    assert all(pred[d] > pred[0] for d in range(1, 6))
    h1 = ev.hand_example(1)                                                  # Handbeispiel: 11/2 mit Vorhersage, 5/2 nur Fixpunkt einer instabilen Schleife
    assert h1["x1_pred"] == F(11, 2) and h1["x1_nopred"] == F(5, 2) and h1["stable"] is False and ev.hand_example(0)["stable"] is True
    kk1, kk2 = F(2, 5), F(6, 5)                                              # Herleitung: x̂₁ = x₁ + u* d(d−1)/2, x̂₂ = d u*, u* = −w  =>  k₁ x₁ = w (1 + k₂ d + k₁ d(d−1)/2), in Brüchen
    for d in range(0, 7):
        w = F(1)
        x1 = w * (1 + kk2 * d + kk1 * F(d * (d - 1), 2)) / kk1
        u = -w
        xh1, xh2 = x1 + u * F(d * (d - 1), 2), d * u
        assert u == -(kk1 * xh1 + kk2 * xh2) and X.equilibrium_x1(kk1, kk2, d, True) == x1
    _has("Herleitung des Dauerfehlers mit Vorhersage", "x̂₁ = x₁ + u* d(d − 1)/2 und x̂₂ = d u*", "k₁ x₁ = w (1 + k₂ d + k₁ d(d − 1)/2)",
         "die 5/2 ohne Vorhersage ist nur der Fixpunkt einer instabilen Schleife und wird nie erreicht")
    _has("\\hat x_1 = x_1 + u^*\\,d(d-1)/2", "k_1 x_1 = w\\,(1 + k_2 d + k_1 d(d-1)/2)", text=APP)
    _has("2,369 / 5,316 / 9,264 / 14,21 / 20,16 / 27,10 (Faktor 1 / 2,24 / 3,91 / 6,00 / 8,51 / 11,44)", "27,0 / 77,6 / 153 / 255 / 381", "4,53 / 12,6 / 24,6 / 40,8 / 60,9",
         "bei d = 1 ist er größer als beim weichen Entwurf (5,32 gegen 4,53), ab d = 2 kleiner (9,26 gegen 12,6)", "(w/k₁)(1 + k₂ d + k₁ d(d − 1)/2)")
    _has("Faktor 2,24 bei d = 1, R = 1", text=APP)


# --- B15 -------------------------------------------------------------------------------------------------------------------------------

def test_b15_costs_and_decay_times():
    k1, k2 = K.gain(1.0)
    P11 = float(K.lqr(1.0)[0][0, 0])
    assert K.cost_ideal(1.0, (1.0, 0.0)) == pytest.approx(2.947, abs=6e-4) and K.cost_ideal(1.0, (0.0, 1.0)) == pytest.approx(4.613, abs=6e-4)
    J10 = [K.cost_pred_closed(1.0, d, (1.0, 0.0)) for d in range(1, 5)]
    assert J10 == pytest.approx([3.947, 4.947, 5.947, 6.947], abs=6e-4) and J10 == pytest.approx([P11 + d for d in range(1, 5)], rel=1e-12)
    assert [100 * (j / P11 - 1) for j in J10] == pytest.approx([33.9, 67.9, 101.8, 135.7], abs=0.06)
    J01 = [K.cost_pred_closed(1.0, d, (0.0, 1.0)) for d in range(1, 5)]
    assert J01 == pytest.approx([13.30, 28.88, 53.35, 88.72], abs=6e-3)
    soft10, soft01 = [], []
    for d in range(1, 5):
        kl = K.gain(1.2 * K.r_krit(d))
        soft10.append(K.cost_loop([kl[0], kl[1]], d, (1.0, 0.0), 1.0))
        soft01.append(K.cost_loop([kl[0], kl[1]], d, (0.0, 1.0), 1.0))
    assert soft10 == pytest.approx([11.58, 13.13, 16.70, 20.61], abs=6e-3) and soft01 == pytest.approx([102.6, 276.1, 663.3, 1336.0], abs=0.6)
    assert [100 * (j / P11 - 1) for j in (soft10[0], soft10[3])] == pytest.approx([293, 599], abs=1.0)
    c10 = {int(d): v for d, v in FROZEN["static_best"]["cost_x10"].items()}
    c01 = {int(d): v for d, v in FROZEN["static_best"]["cost_x01"].items()}
    assert [c10[d]["cost"] for d in range(1, 5)] == pytest.approx([4.646, 6.855, 9.193, 11.57], abs=6e-3)
    assert [c01[d]["cost"] for d in range(1, 5)] == pytest.approx([21.61, 74.86, 189.0, 388.4], abs=0.06)
    assert [100 * (c10[d]["cost"] / P11 - 1) for d in range(1, 5)] == pytest.approx([58, 133, 212, 293], abs=1.0)
    for d in range(1, 5):                                                     # die vorgerechneten Gewinne ergeben die vorgerechneten Kosten (Lyapunov, exakt nachgerechnet)
        assert K.cost_loop([c10[d]["k1"], c10[d]["k2"]], d, (1.0, 0.0), 1.0) == pytest.approx(c10[d]["cost"], rel=1e-9)
        assert K.cost_loop([c01[d]["k1"], c01[d]["k2"]], d, (0.0, 1.0), 1.0) == pytest.approx(c01[d]["cost"], rel=1e-9)
        assert J10[d - 1] < c10[d]["cost"] < soft10[d - 1] and J01[d - 1] < c01[d]["cost"] < soft01[d - 1]     # Vorhersage: die kleinste Zahl in jeder Zeile
    # Abklingzeit
    krit = ev.krit_rows()
    assert [ev.decay_of(k1, k2, d, d, K.rho_poly(k1, k2, 0))[0] for d in range(5)] == [5, 6, 7, 8, 9]
    assert [r["decay_soft"] for r in krit[:4]] == [213, 314, 434, 548] and krit[7]["decay_soft"] == 996
    assert [r["decay_d0"] for r in krit] == [11, 21, 30, 40, 49, 58, 68, 77]    # fairer Vergleich: derselbe große R (1,2 · R_krit) ohne Verzug
    for r in krit[:4]:
        assert ev.takt(r["R_soft"], 0).decay == r["decay_d0"] and ev.takt(r["R_soft"], r["d"]).decay == r["decay_soft"]
    stat = [ev.decay_of(FROZEN["static_best"]["rho"][str(d)]["k1"], FROZEN["static_best"]["rho"][str(d)]["k2"], d, 0, FROZEN["static_best"]["rho"][str(d)]["rho"])[0] for d in range(1, 5)]
    assert stat == [21, 37, 52, 67] and [b - a for a, b in zip(stat, stat[1:])] == [16, 15, 15]
    _has("ohne Verzug 2,947 (= P₁₁)", "3,947 / 4,947 / 5,947 / 6,947 für d = 1 bis 4", "33,9 / 67,9 / 101,8 / 135,7 %; genau P₁₁ + d", "11,58 / 13,13 / 16,70 / 20,61 (+293 bis +599 %)",
         "4,646 / 6,855 / 9,193 / 11,57 (+58 / +133 / +212 / +293 %)", "ohne Verzug 4,613; Vorhersage 13,30 / 28,88 / 53,35 / 88,72; beste statische 21,61 / 74,86 / 189,0 / 388,4; weicher Entwurf 102,6 / 276,1 / 663,3 / 1 336",
         "Vorhersage 5 / 6 / 7 / 8 / 9 Perioden; weicher Entwurf (R = 1,2 · R_krit) 213 / 314 / 434 / 548 (d = 1 bis 4; bis d = 8: 996); beste statische Verstärkung 21 / 37 / 52 / 67", "um genau 1 je d, bei der besten statischen Verstärkung um 15 bis 16",
         "klingt ohne Verzug nach 11 / 21 / 30 / 40 Perioden ab (d = 1 bis 4; 49 / 58 / 68 / 77 für d = 5 bis 8)", "derselbe große R ohne Verzug klingt nach 11 bis 40 Perioden ab",
         "die kleinste Zahl unter den Gesetzen mit Verzug")
    _has("Zeile „Kosten und Abklingzeit nach dem Stoß“", "Zeile „MPC-Brücke“")


# --- B16 -------------------------------------------------------------------------------------------------------------------------------

def test_b16_scatter_and_monte_carlo():
    k1, k2 = K.gain(1.0)
    ideal = K.stationary_cost(K.pred_gain(k1, k2, 0), 0, 1.0)
    pred = [K.stationary_cost(K.pred_gain(k1, k2, d), d, 1.0) for d in range(1, 5)]
    assert ideal == pytest.approx(4.613, abs=6e-4) and pred == pytest.approx([13.30, 28.88, 53.35, 88.72], abs=6e-3)
    c01 = {int(d): v["cost"] for d, v in FROZEN["static_best"]["cost_x01"].items()}
    stat = {int(d): v["cost"] for d, v in FROZEN["static_best"]["stationary"].items()}
    assert [stat[d] for d in range(1, 5)] == pytest.approx([21.61, 74.86, 189.0, 388.4], abs=0.06) and all(stat[d] == pytest.approx(c01[d], rel=1e-4) for d in range(1, 5))
    for d in range(1, 5):                                                     # Identität durch Linearität: Streuung gleich Kosten für x₀ = (0, 1)
        assert pred[d - 1] == pytest.approx(K.cost_pred_closed(1.0, d, (0.0, 1.0)), rel=1e-9)
    want = {10000: 3.96, 20000: 1.97, 50000: 0.43}
    for n, dev in want.items():
        m = ev.mc_run(1.0, 2, n, 1814)
        assert 100 * abs(m["dev"]) == pytest.approx(dev, abs=0.06), n
    assert ev.verdict_mc(ev.mc_run(1.0, 2, 10000, 1814))[0] == "warning" and ev.verdict_mc(ev.mc_run(1.0, 2, 20000, 1814))[0] == "success"
    assert ev.mc_run(1.0, 2, 20000, 1814)["mc"] == pytest.approx(29.45, abs=6e-3) and ev.mc_run(1.0, 2, 50000, 1814)["mc"] == pytest.approx(29.00, abs=6e-3)
    _has("ohne Verzug 4,613; mit Vorhersage 13,30 / 28,88 / 53,35 / 88,72 (d = 1 bis 4)", "21,61 / 74,86 / 189,0 / 388,4", "29,45 bei 20 000, 29,00 bei 50 000 Perioden gegen 28,88", "1,97 % bzw. 0,43 %", "bei 10 000 Perioden 3,96 %",
         "im Orakel-Test 40 000 Perioden, Band 3 %")


# --- Presets: je Preset die erwartete Verdikt-Zeile ------------------------------------------------------------------------------------------

def _settings(name):
    full = {key: C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default) for key, state_key in P.PRESET_KEYS.items()}
    return ev.Settings(logR=full["logR"], d=full["d"], theta=full["theta"], law=full["law"], x0=full["x0"], b=full["b"], dh_off=full["dh_off"], w=full["w"])


def test_each_preset_reports_exactly_the_effect_it_has():
    S = {name: _settings(name) for name in C.PRESETS}
    # Eine Periode Verzug (R = 1): instabil
    s = S["Eine Periode Verzug (R = 1)"]
    kind, text = ev.verdict_takt(ev.takt(s.R, s.d))
    assert kind == "warning" and "ρ = 1,193 statt 0,422" in text
    # Skalar auf der Grenze: Skalar d = 2 ist Grenze, der Wagen R = 1 ist bei d = 2 instabil
    s = S["Skalar auf der Grenze (d = 2)"]
    assert ev.verdict_scalar(ev.scalar_rows(5)[s.d])[0] == "info" and ev.verdict_takt(ev.takt(s.R, s.d))[0] == "warning"
    # Weich genug (R = 10): hält mit d = 1 (ρ = 0,961)
    s = S["Weich genug (R = 10)"]
    kind, text = ev.verdict_takt(ev.takt(s.R, s.d))
    assert kind == "success" and "ρ = 0,961" in text
    # Eine halbe Periode Rechenzeit: hält R = 1 (ρ = 0,918), meldet keine Instabilität
    s = S["Eine halbe Periode Rechenzeit"]
    kind, text = ev.verdict_theta(ev.theta_info(s.R, s.theta))
    assert kind == "success" and "ρ = 0,918" in text and s.theta == 0.5
    assert ev.verdict_theta(ev.theta_info(0.1, 0.5))[0] == "warning"           # R = 0,1 hielte nicht
    # Mit Vorhersage (d = 3): Mehrkosten +101,8 %
    s = S["Mit Vorhersage (d = 3)"]
    J0, rows = ev.law_rows(s.R, s.d, s.x0)
    kind, text = ev.verdict_vorhersage(s.d, s.x0, rows, s.R, ev.takt(s.R, 0).decay)
    assert "(ohne Verzug 5 bei R = 1)" in text and "+101,8" in text and "Abklingzeit ist 8 Perioden" in text and "ohne Vorhersage ist instabil" in text
    # Falsche Masse (b = 0,5) bei d = 2: mit Vorhersage instabil, ohne Verzug stabil
    s = S["Falsche Masse (b = 0,5)"]
    kind, text = ev.verdict_modell(ev.modell_info(s.R, s.d, s.dh, s.b))
    assert kind == "warning" and "ist instabil" in text
    k1, k2 = K.gain(s.R)
    assert K.rho_loop(k1, k2, 0, 0, 0.5) < 1.0
    # Störkraft (w = 2): Dauerfehler 10,63 gegen 4,74
    s = S["Störkraft (w = 2)"]
    kind, text = ev.verdict_stoer(ev.stoer_info(s.R, s.d, s.w))
    assert "Gleichgewichtslage 10,63 mit Vorhersage gegen 4,74 ohne Verzug" in text


def test_app_text_numbers_match_the_readme_numbers():
    _has("von 0,01 bis 1,94 bei d = 0", "0,77 bis 1,25 bei d = 5", text=APP)
    krit = ev.krit_rows()
    assert krit[0]["rkrit"] == pytest.approx(6.854, abs=1e-3)


def test_app_loads_and_renders_every_preset_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    for name in C.PRESETS:
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert not at.exception, name
        assert len(at.success) + len(at.info) + len(at.warning) >= 8, name     # Kernfragen und Verdikte der Experimente erscheinen
