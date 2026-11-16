"""Kern der Demo (nur numpy und scipy): der Wagen mit Verzug, die Wurzeln der Schleife, die Vorhersage, Kosten und Dauerfehler.

Modell (alles eigene Herleitung in diesem Modell, bei diesen Parametern): Wagen x_{k+1} = A x_k + B (b u_k + w), A = [[1, 1], [0, 1]], Q = I, Ts = 1 Periode.
Zwei Fassungen von B: die Standardfassung B = [0, 1]ᵀ (ganzzahlige Verzüge) und die exakte Abtast-Halte-Fassung B = [½, 1]ᵀ (Rechenzeit als Bruchteil der Periode).
Das Gesetz u_k = −K x_{k−d} nutzt den Gewinn K der DARE des Entwurfs ohne Verzug; bis k = d − 1 liegt keine Messung vor (blind: u = 0).
Mit Vorhersage: x̂_k = A^dh x_{k−d} + Σ_{j<dh} A^(dh−1−j) B u_{k−dh+j}, u_k = −K x̂_k.

Frühere Messskripte (nicht im Repository) werden nicht importiert; die Messlogik ist hier neu geschrieben. Die Zustandsmatrizen entstehen direkt (nicht durch
Probieren), die Gegenprobe mit dem Probieren steht in tests/test_oracle_regelung_verzug.py.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import scipy.linalg as la

A = np.array([[1.0, 1.0], [0.0, 1.0]])
B0 = np.array([[0.0], [1.0]])        # Standardfassung
B_ZOH = np.array([[0.5], [1.0]])     # exakte Abtast-Halte-Diskretisierung von x'' = u
Q = np.eye(2)
PHI = (1.0 + math.sqrt(5.0)) / 2.0

R_GRID_LO, R_GRID_HI, R_GRID_PER_DECADE = -3.0, 7.0, 10   # grobes Gitter der Bisektion nach R_krit (log10 R)


# --- Entwurf ---------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def lqr(R: float, B_key: str = "B0") -> tuple[np.ndarray, np.ndarray]:
    """DARE-Lösung P und Gewinn K (u = −K x) des Wagens, Q = I, Kostengewicht R. B_key: "B0" (Standardfassung) oder "ZOH"."""
    if not (math.isfinite(R) and R > 0):
        raise ValueError(f"Kostengewicht R muss endlich und positiv sein, nicht {R}")
    B = B0 if B_key == "B0" else B_ZOH
    Rm = np.array([[float(R)]])
    P = la.solve_discrete_are(A, B, Q, Rm)
    K = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)
    P.setflags(write=False)
    K.setflags(write=False)
    return P, K


def gain(R: float, B_key: str = "B0") -> tuple[float, float]:
    K = lqr(float(R), B_key)[1]
    return float(K[0, 0]), float(K[0, 1])


def riccati_gain_n(R: float, N: int, terminal: str = "dare") -> np.ndarray:
    """Gewinn der ersten Stufe des unbeschränkten N-Schritt-Problems (Riccati-Rekursion rückwärts); terminal "dare" (P_f = DARE) oder "q" (P_f = Q)."""
    Rm = np.array([[float(R)]])
    Pk = lqr(float(R))[0].copy() if terminal == "dare" else Q.copy()
    Kn = None
    for _ in range(N):
        Kn = np.linalg.solve(Rm + B0.T @ Pk @ B0, B0.T @ Pk @ A)
        Pk = Q + A.T @ Pk @ (A - B0 @ Kn)
    return Kn


# --- Wurzeln ohne Vorhersage -----------------------------------------------------------------------------------------------------------

def poly_wagen(k1, k2, d: int) -> list:
    """Koeffizienten (höchste Potenz zuerst) von P_d(z) = z^d (z−1)² + k2 (z−1) + k1 für B = [0, 1]ᵀ; gilt für Zahlen und Brüche."""
    if d < 0:
        raise ValueError(f"Verzug d muss eine ganze Zahl ≥ 0 sein, nicht {d}")
    p = [0] * (d + 3)
    p[0], p[1], p[2] = 1, -2, 1
    p[d + 1] = p[d + 1] + k2
    p[d + 2] = p[d + 2] + k1 - k2
    return p


def roots_wagen(k1: float, k2: float, d: int) -> np.ndarray:
    return np.roots(np.array(poly_wagen(float(k1), float(k2), d), dtype=float))


def rho_poly(k1: float, k2: float, d: int) -> float:
    """Spektralradius aus den Wurzeln von P_d (nicht aus der Zustandsmatrix: deren d Nullwurzeln rauschen)."""
    return float(np.max(np.abs(roots_wagen(k1, k2, d))))


def aug_delay(B: np.ndarray, K: np.ndarray, d: int) -> np.ndarray:
    """Zustandsmatrix von x_{k+1} = A x_k − B K x_{k−d}; Zustand (x_k, x_{k−1}, …, x_{k−d}), Dimension 2 (d + 1)."""
    n = 2
    M = np.zeros((n * (d + 1), n * (d + 1)))
    M[:n, :n] = A
    M[:n, n * d:n * (d + 1)] += -B @ K
    for i in range(1, d + 1):
        M[n * i:n * (i + 1), n * (i - 1):n * i] = np.eye(n)
    return M


def aug_actuator(B: np.ndarray, K: np.ndarray, d: int) -> np.ndarray:
    """Stellverzug: x_{k+1} = A x_k + B u_{k−d}, u_k = −K x_k; Zustand (x_k, u_{k−1}, …, u_{k−d}); zweiter Rechenweg zu M6."""
    if d == 0:
        return A - B @ K
    M = np.zeros((2 + d, 2 + d))
    M[:2, :2] = A
    M[:2, 2 + d - 1] = B.ravel()
    M[2, :2] = -K.ravel()
    for i in range(1, d):
        M[2 + i, 2 + i - 1] = 1.0
    return M


def rho_eig(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def rho_design(R: float, d: int) -> float:
    k1, k2 = gain(R)
    return rho_poly(k1, k2, d)


def lqr_q(Qm: np.ndarray, R: float) -> np.ndarray:
    """Gewinn K des Wagens (B = [0, 1]ᵀ) für ein beliebiges symmetrisches Q (für die Zufallsentwürfe der Zählung)."""
    Rm = np.array([[float(R)]])
    P = la.solve_discrete_are(A, B0, Qm, Rm)
    return np.linalg.solve(Rm + B0.T @ P @ B0, B0.T @ P @ A)


def _stable_design(R: float, d: int, B_key: str) -> bool:
    if B_key == "B0":
        return rho_design(R, d) < 1.0
    return rho_eig(aug_delay(B_ZOH, lqr(float(R), "ZOH")[1], d)) < 1.0


@lru_cache(maxsize=64)
def r_krit(d: int, B_key: str = "B0") -> float | None:
    """Kleinstes R des DARE-Entwurfs (Q = I), ab dem die Schleife mit d Perioden Verzug stabil ist (Bisektion auf log R, grobes Gitter 10 je Dekade).

    B_key "B0": B = [0, 1]ᵀ (Standardfassung), "ZOH": B = [½, 1]ᵀ. Voraussetzung (gemessen, nicht bewiesen): genau ein Wechsel instabil → stabil auf dem Gitter
    von 4 001 Punkten (vorgerechnet, data/rvz_frozen.json). d = 0: None (immer stabil)."""
    if d <= 0:
        return None
    n = int(round((R_GRID_HI - R_GRID_LO) * R_GRID_PER_DECADE)) + 1
    logs = np.linspace(R_GRID_LO, R_GRID_HI, n)
    first = None
    for i, lg in enumerate(logs):
        if _stable_design(10.0 ** lg, d, B_key):
            first = i
            break
    if first is None:
        return None
    if first == 0:
        return float(10.0 ** logs[0])
    lo, hi = float(logs[first - 1]), float(logs[first])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _stable_design(10.0 ** mid, d, B_key):
            hi = mid
        else:
            lo = mid
    return float(10.0 ** hi)


# --- Vorhersage und allgemeine lineare Gesetze ----------------------------------------------------------------------------------------

def pred_gain(k1, k2, dh: int) -> list:
    """Gewinn der Vorhersage als Gesetz auf (x_{k−d}, u_{k−dh}, …, u_{k−1}): [K A^dh, K A^(dh−1) B, …, K B] = [k1, k1 dh + k2, k1 (dh−1−j) + k2 …] (B = [0, 1]ᵀ)."""
    return [k1, k1 * dh + k2] + [k1 * (dh - 1 - j) + k2 for j in range(dh)]


def _loop_rows(G: list, d: int, b) -> list[list]:
    """Zustandsmatrix als Liste von Zeilen (Zahlen oder Brüche): z = (x_k, …, x_{k−d}, u_{k−1}, …, u_{k−dh}), u_k = −(G[:2]·x_{k−d} + Σ_j G[2+j] u_{k−dh+j}),
    x_{k+1} = A x_k + B (b u_k + w), dh = len(G) − 2."""
    dh = len(G) - 2
    D = 2 * (d + 1) + dh
    zero = 0 * G[0]
    row = [zero] * D                       # u_k = −row · z
    row[2 * d] = G[0]
    row[2 * d + 1] = G[1]
    for j in range(dh):
        row[2 * (d + 1) + dh - 1 - j] = G[2 + j]
    M = [[zero] * D for _ in range(D)]
    M[0][0] = 1 + zero
    M[0][1] = 1 + zero
    M[1][1] = 1 + zero
    for c in range(D):
        M[1][c] = M[1][c] - b * row[c]
    for i in range(1, d + 1):
        for r in range(2):
            M[2 * i + r][2 * (i - 1) + r] = 1 + zero
    if dh > 0:
        for c in range(D):
            M[2 * (d + 1)][c] = -row[c]
        for i in range(1, dh):
            M[2 * (d + 1) + i][2 * (d + 1) + i - 1] = 1 + zero
    return M


def loop_from_gain(G: list, d: int, b=1.0, exact: bool = False) -> np.ndarray:
    """Zustandsmatrix der Schleife mit dem linearen Gesetz G; b ist die wahre Beschleunigungsverstärkung (der Regler nimmt 1 an)."""
    M = _loop_rows(list(G), d, b)
    return np.array(M, dtype=object if exact else float)


def loop_matrix(k1, k2, d: int, dh: int, b=1.0, exact: bool = False) -> np.ndarray:
    """Schleife mit Vorhersage über dh Perioden bei wahrem Verzug d (dh = 0: keine Vorhersage)."""
    return loop_from_gain(pred_gain(k1, k2, dh), d, b, exact)


def loop_row(G: list, d: int) -> np.ndarray:
    """Zeile r mit u_k = −r · z (für Kosten und Streuung)."""
    dh = len(G) - 2
    D = 2 * (d + 1) + dh
    row = np.zeros(D)
    row[2 * d], row[2 * d + 1] = G[0], G[1]
    for j in range(dh):
        row[2 * (d + 1) + dh - 1 - j] = G[2 + j]
    return row


def rho_loop(k1: float, k2: float, d: int, dh: int, b: float = 1.0) -> float:
    return rho_eig(loop_matrix(k1, k2, d, dh, b))


# --- Simulation ------------------------------------------------------------------------------------------------------------------------

def simulate(k1: float, k2: float, d: int, x0, T: int, dh: int = 0, w: float = 0.0, b: float = 1.0):
    """Bahn x_0 … x_T und Stellgrößen u_0 … u_{T−1}. dh = 0: u_k = −K x_{k−d}; dh > 0: mit Vorhersage über dh Perioden; blind (u = 0) für k < d.
    Strecke: x1⁺ = x1 + x2, x2⁺ = x2 + b u + w."""
    if d < 0 or dh < 0:
        raise ValueError(f"Verzug d = {d} und angenommener Verzug dh = {dh} müssen ≥ 0 sein")
    x1, x2 = float(x0[0]), float(x0[1])
    xs1, xs2, us = [x1], [x2], []
    for k in range(T):
        if k < d:
            u = 0.0
        else:
            y1, y2 = xs1[k - d], xs2[k - d]
            h1, h2 = y1 + dh * y2, y2
            for j in range(dh):
                idx = k - dh + j
                uj = us[idx] if idx >= 0 else 0.0
                h1 += (dh - 1 - j) * uj
                h2 += uj
            u = -(k1 * h1 + k2 * h2)
        us.append(u)
        x1, x2 = x1 + x2, x2 + b * u + w
        xs1.append(x1)
        xs2.append(x2)
    return np.column_stack([xs1, xs2]), np.array(us)


def decay_time(xs: np.ndarray, frac: float = 0.01) -> tuple[int, float]:
    """Letzte Periode, in der ‖x_k‖₂ über `frac` der Spitze liegt; zweiter Wert: die Spitze."""
    nm = np.linalg.norm(xs, axis=1)
    peak = float(nm.max())
    idx = np.nonzero(nm > frac * peak)[0]
    return int(idx[-1]), peak


# --- Kosten, Gleichgewicht, Streuung ---------------------------------------------------------------------------------------------------

def cost_loop(G: list, d: int, x0, R: float) -> float:
    """Σ_k x'Qx + R u² nach der Anfangsabweichung x0, Q = I; blinde Perioden (u = 0) für k < d, danach exakt per Lyapunov der Schleife."""
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(d):
        xs.append(A @ xs[-1])
    J = sum(float(x @ x) for x in xs[:d])
    M = loop_from_gain(G, d, 1.0)
    D = M.shape[0]
    row = loop_row(G, d)
    C = np.zeros((D, D))
    C[:2, :2] = Q
    C += R * np.outer(row, row)
    X = la.solve_discrete_lyapunov(M.T, C)
    z = np.zeros(D)
    for j in range(d + 1):
        z[2 * j:2 * j + 2] = xs[d - j]
    return J + float(z @ X @ z)


def cost_pred_closed(R: float, d: int, x0) -> float:
    """Kosten mit Vorhersage in geschlossener Form: blinde Perioden plus x_d' P x_d (P aus der DARE)."""
    P = lqr(float(R))[0]
    x = np.asarray(x0, dtype=float)
    J = 0.0
    for _ in range(d):
        J += float(x @ x)
        x = A @ x
    return J + float(x @ P @ x)


def cost_ideal(R: float, x0) -> float:
    x = np.asarray(x0, dtype=float)
    return float(x @ lqr(float(R))[0] @ x)


def equilibrium_x1(G: list, d: int, w: float = 1.0) -> float:
    """Gleichgewichtslage x1* bei konstanter Störkraft w (Lösung von (I − M) z = g w)."""
    M = loop_from_gain(G, d, 1.0)
    D = M.shape[0]
    g = np.zeros(D)
    g[1] = w
    return float(np.linalg.solve(np.eye(D) - M, g)[0])


def equilibrium_formula(k1: float, k2: float, d: int, w: float = 1.0, predict: bool = True) -> float:
    """x1* = (w/k1)(1 + k2 d + k1 d (d−1)/2) mit Vorhersage, w/k1 ohne (eigene Herleitung: siehe Expander „Mathematische Formulierung“ der App, Abschnitt Dauerfehler). Gleichgewichtslage heißt Fixpunkt der Gleichungen; erreicht wird er nur bei stabiler Schleife."""
    if not predict:
        return w / k1
    return (w / k1) * (1 + k2 * d + k1 * d * (d - 1) / 2)


def stationary_cost(G: list, d: int, R: float, sigma: float = 1.0) -> float:
    """Kosten je Periode E[x'Qx + R u²] unter Störkraft N(0, σ²) je Periode (stationär, Lyapunov)."""
    M = loop_from_gain(G, d, 1.0)
    D = M.shape[0]
    g = np.zeros(D)
    g[1] = sigma
    S = la.solve_discrete_lyapunov(M, np.outer(g, g))
    row = loop_row(G, d)
    return float(np.trace(Q @ S[:2, :2]) + R * (row @ S @ row))


def monte_carlo_cost(G: list, d: int, R: float, periods: int, seed: int, burn: int = 1000) -> float:
    """Mittel von x'Qx + R u² über `periods` Perioden mit Störkraft N(0, 1) aus dem ganzzahligen SplitMix64 (Zufalls-Seed `seed`), ohne die ersten `burn`."""
    from rvz_rng import SplitMix64

    rng = SplitMix64(seed)
    M = loop_from_gain(G, d, 1.0)
    row = loop_row(G, d)
    z = np.zeros(M.shape[0])
    acc = 0.0
    for t in range(periods):
        if t >= burn:
            u = -(row @ z)
            acc += z[0] * z[0] + z[1] * z[1] + R * u * u
        z = M @ z
        z[1] += rng.normal()
    return acc / max(periods - burn, 1)


# --- Modellfehler ---------------------------------------------------------------------------------------------------------------------

def b_interval(G: list, d: int, step: float = 0.002, lo: float = 0.01, hi: float = 6.0):
    """Stabiler Bereich der wahren Verstärkung b (der Regler nimmt b = 1 an): Gitter lo … hi, Ränder per Bisektion verfeinert.

    Rückgabe (b_min, b_max, Zahl der stabilen Teilintervalle, b_min am Gitterrand?). b_min = lo heißt: bis zum Gitterrand stabil (untere Grenze nicht erreicht)."""
    M0 = loop_from_gain(G, d, 0.0)
    M1 = loop_from_gain(G, d, 1.0)
    bs = np.arange(lo, hi, step)
    stack = M0[None, :, :] + bs[:, None, None] * (M1 - M0)[None, :, :]
    rho = np.max(np.abs(np.linalg.eigvals(stack)), axis=1)
    stab = rho < 1.0
    if not stab.any():
        return None, None, 0, False
    runs = int(stab[0]) + int(np.sum(stab[1:] & ~stab[:-1]))
    i0 = int(np.argmax(stab))
    i1 = len(stab) - 1 - int(np.argmax(stab[::-1]))

    def stable_at(b):
        return rho_eig(M0 + b * (M1 - M0)) < 1.0

    def refine(a, c, ascending):
        left, right = a, c
        for _ in range(50):
            mid = 0.5 * (left + right)
            if stable_at(mid) == ascending:
                right = mid
            else:
                left = mid
        return 0.5 * (left + right)

    at_edge = i0 == 0
    bmin = float(bs[0]) if at_edge else refine(float(bs[i0 - 1]), float(bs[i0]), True)
    bmax = float(bs[-1]) if i1 == len(bs) - 1 else refine(float(bs[i1]), float(bs[i1 + 1]), False)
    return bmin, bmax, runs, at_edge


# --- Rechenzeit als Bruchteil der Periode (B = [½, 1]ᵀ) ---------------------------------------------------------------------------------

def gamma(tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Γ0 = ∫_0^(1−τ) e^{A_c r} B_c dr und Γ1 = ∫_(1−τ)^1 e^{A_c r} B_c dr für x'' = u mit Halteglied (geschlossene Form)."""
    g0 = np.array([[(1 - tau) ** 2 / 2], [1 - tau]])
    g1 = np.array([[(1 - (1 - tau) ** 2) / 2], [tau]])
    return g0, g1


def theta_split(theta: float) -> tuple[int, float]:
    if theta < 0 or not math.isfinite(theta):
        raise ValueError(f"Rechenzeit θ muss endlich und ≥ 0 sein, nicht {theta}")
    m = int(math.floor(theta + 1e-12))
    tau = theta - m
    if tau < 1e-12:
        tau = 0.0
    return m, tau


def theta_matrix(K: np.ndarray, theta: float) -> np.ndarray:
    """Zustandsmatrix mit Rechenzeit θ = m + τ Perioden; Zustand (x_k, u_{k−1}, …, u_{k−m−1}), u_k = −K x_k, K aus der DARE von B = [½, 1]ᵀ.
    In der Periode wirkt zuerst u_{k−m−1}, nach dem Bruchteil τ dann u_{k−m}: x_{k+1} = A x_k + Γ1 u_{k−m−1} + Γ0 u_{k−m}."""
    m, tau = theta_split(theta)
    g0, g1 = gamma(tau)
    D = 2 + m + 1
    M = np.zeros((D, D))
    M[:2, :2] = A

    def add(j, g):
        if j == 0:
            M[:2, :2] += -g @ K
        else:
            M[:2, 2 + j - 1] += g.ravel()

    add(m, g0)
    add(m + 1, g1)
    M[2, :2] = -K.ravel()
    for i in range(1, m + 1):
        M[2 + i, 2 + i - 1] = 1.0
    return M


def rho_theta(R: float, theta: float) -> float:
    K = lqr(float(R), "ZOH")[1]
    return rho_eig(theta_matrix(K, theta))


@lru_cache(maxsize=256)
def theta_krit(R: float) -> float | None:
    """Erste Rechenzeit θ* (in Perioden, 0 bis 3), ab der der Entwurf mit B = [½, 1]ᵀ instabil wird (Gitter 0,01, dann Bisektion); None: bis 3 stabil."""
    K = lqr(float(R), "ZOH")[1]
    ths = np.round(np.arange(0.0, 3.0 + 1e-9, 0.01), 10)
    bad = None
    for i, t in enumerate(ths):
        if rho_eig(theta_matrix(K, float(t))) >= 1.0:
            bad = i
            break
    if bad is None:
        return None
    if bad == 0:
        return 0.0
    lo, hi = float(ths[bad - 1]), float(ths[bad])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rho_eig(theta_matrix(K, mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def simulate_theta_growth(K: np.ndarray, theta: float, N: int = 400) -> float:
    """Unabhängiger Rechenweg zu rho_theta: Zeitschritt-Simulation über Teilintervalle mit dem Matrixexponential (ohne die Γ-Formeln); Wachstum je Periode aus der Steigung des Logarithmus."""
    Ac = np.array([[0.0, 1.0], [0.0, 0.0]])
    m, tau = theta_split(theta)

    def vanloan(t):
        M = np.zeros((3, 3))
        M[:2, :2] = Ac
        M[:2, 2] = [0.0, 1.0]
        E = la.expm(M * t)
        return E[:2, :2], E[:2, 2]

    Phi_a, Ga = vanloan(tau) if tau > 0 else (np.eye(2), np.zeros(2))
    Phi_b, Gb = vanloan(1 - tau)
    x = np.array([1.0, 0.0])
    hist: dict[int, float] = {}
    norms = []
    for j in range(N):
        hist[j] = float(-(K @ x)[0])
        x = Phi_a @ x + Ga * hist.get(j - m - 1, 0.0)
        x = Phi_b @ x + Gb * hist.get(j - m, 0.0)
        norms.append(np.linalg.norm(x))
    w = min(100, N // 3)
    tail = np.maximum(np.array(norms[-w:]), 1e-300)
    slope = np.polyfit(np.arange(w), np.log(tail), 1)[0]
    return float(np.exp(slope))


# --- Skalar x⁺ = x + u, q = r = 1 ------------------------------------------------------------------------------------------------------

def scalar_gain() -> float:
    """K = p/(1 + p) mit p = φ (Goldener Schnitt): p² = 1 + p aus der Riccati-Gleichung p = 1 + p − p²/(1 + p), also K = 1/φ."""
    return 1.0 / PHI


def scalar_poly(K: float, d: int) -> list:
    """z^(d+1) − z^d + K (höchste Potenz zuerst); d = 0: z − (1 − K)."""
    if d == 0:
        return [1.0, K - 1.0]
    return [1.0, -1.0] + [0.0] * (d - 1) + [K]


def scalar_rho(K: float, d: int) -> float:
    return float(np.max(np.abs(np.roots(scalar_poly(K, d)))))


def lager_grenze(L: int) -> float:
    """Grenze α_krit = 2 sin(π/(4L − 2)) des Lagers ohne Leitung mit Lieferzeit L (lager-lieferzeit-demo); hier nur als Gegenprobe des Skalarfalls mit L = d + 1."""
    return 2.0 * math.sin(math.pi / (4 * L - 2))
