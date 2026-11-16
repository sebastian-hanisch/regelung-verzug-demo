"""Exakte Rechnung mit Brüchen (`fractions`) zu den Handbeispielen: Schur–Cohn-Prüfung, charakteristisches Polynom (Faddeev–LeVerrier), Bahnen und Gleichgewicht.

Alles ohne Gleitkomma: eine Aussage wie „K = (2/5, 6/5) ist bei d = 1 instabil“ oder „x1* = 11/2“ ist hier ein Bruch gleich Bruch, kein Band.
Nur Standardbibliothek und numpy (Objekt-Matrizen). Die Schur–Cohn-Prüfung ist in tests/ gegen numpy.roots an 400 Zufallspolynomen abgesichert.
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np

import rvz_dynamics as D


def schur_stable(coeffs) -> bool:
    """Schur–Cohn exakt: liegen alle Wurzeln von c[0] z^n + … + c[n] echt im Einheitskreis? (Brüche; Wurzeln auf dem Kreis gelten als nicht stabil)"""
    c = [F(v) for v in coeffs]
    if c[0] < 0:
        c = [-v for v in c]
    while len(c) > 1:
        n = len(c) - 1
        a_n, a_0 = c[0], c[-1]
        if abs(a_0) >= abs(a_n):
            return False
        c = [a_n * c[i] - a_0 * c[n - i] for i in range(n)]
    return c[0] > 0


def charpoly(M) -> list[F]:
    """Koeffizienten des charakteristischen Polynoms det(zI − M), höchste Potenz zuerst, exakt (Faddeev–LeVerrier mit Brüchen)."""
    n = len(M)
    Mo = np.array([[F(v) for v in row] for row in M], dtype=object)
    eye = np.array([[F(int(i == j)) for j in range(n)] for i in range(n)], dtype=object)
    c = [F(1)]
    Mk = np.array([[F(0)] * n for _ in range(n)], dtype=object)
    for k in range(1, n + 1):
        Mk = Mo @ Mk + c[-1] * eye
        c.append(-sum((Mo @ Mk)[i, i] for i in range(n)) / k)
    return c


def poly_roots_sum(coeffs) -> F:
    """Summe der Wurzeln nach Vieta: −c[1]/c[0]."""
    return -F(coeffs[1]) / F(coeffs[0])


def trajectory(k1: F, k2: F, d: int, x0, T: int, predict: bool):
    """Bahn x_0 … x_T des Wagens (B = [0, 1]ᵀ) in Brüchen; blinde Perioden u = 0 für k < d; predict: Vorhersage über dh = d Perioden."""
    dh = d if predict else 0
    xs = [[F(x0[0]), F(x0[1])]]
    us: list[F] = []
    for k in range(T):
        if k < d:
            u = F(0)
        else:
            y1, y2 = xs[k - d]
            h1, h2 = y1 + dh * y2, y2
            for j in range(dh):
                idx = k - dh + j
                uj = us[idx] if idx >= 0 else F(0)
                h1 += (dh - 1 - j) * uj
                h2 += uj
            u = -(k1 * h1 + k2 * h2)
        us.append(u)
        x1, x2 = xs[k]
        xs.append([x1 + x2, x2 + u])
    return xs, us


def _solve(Mat, rhs):
    n = len(rhs)
    Aug = [list(Mat[i]) + [rhs[i]] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if Aug[r][c] != 0)
        Aug[c], Aug[piv] = Aug[piv], Aug[c]
        pv = Aug[c][c]
        Aug[c] = [v / pv for v in Aug[c]]
        for r in range(n):
            if r != c and Aug[r][c] != 0:
                f = Aug[r][c]
                Aug[r] = [a - f * b for a, b in zip(Aug[r], Aug[c])]
    return [Aug[i][n] for i in range(n)]


def equilibrium_x1(k1: F, k2: F, d: int, predict: bool, w: F = F(1)) -> F:
    """Gleichgewichtslage x1* bei konstanter Störkraft w als exakte Lösung des linearen Gleichungssystems (I − M) z = g w."""
    dh = d if predict else 0
    M = D.loop_matrix(k1, k2, d, dh, b=F(1), exact=True)
    n = M.shape[0]
    mat = [[F(int(i == j)) - M[i][j] for j in range(n)] for i in range(n)]
    rhs = [F(0)] * n
    rhs[1] = F(w)
    return _solve(mat, rhs)[0]


def gamma_exact(tau: F) -> tuple[list[F], list[F]]:
    tau = F(tau)
    g0 = [(1 - tau) ** 2 / 2, 1 - tau]
    g1 = [(1 - (1 - tau) ** 2) / 2, tau]
    return g0, g1


def theta_matrix_exact(k1: F, k2: F, m: int, tau: F) -> list[list[F]]:
    """Zustandsmatrix der Rechenzeit θ = m + τ (B = [½, 1]ᵀ, Gewinn K = (k1, k2)) in Brüchen, Zustand (x_k, u_{k−1}, …, u_{k−m−1}); nur für ganze m ≥ 0."""
    g0, g1 = gamma_exact(tau)
    Dm = 2 + m + 1
    M = [[F(0)] * Dm for _ in range(Dm)]
    K = [k1, k2]
    A = [[F(1), F(1)], [F(0), F(1)]]
    for i in range(2):
        for j in range(2):
            M[i][j] = A[i][j]

    def add(j, g):
        for i in range(2):
            if j == 0:
                for c in range(2):
                    M[i][c] -= g[i] * K[c]
            else:
                M[i][2 + j - 1] += g[i]

    add(m, g0)
    add(m + 1, g1)
    M[2][0], M[2][1] = -k1, -k2
    for i in range(1, m + 1):
        M[2 + i][2 + i - 1] = F(1)
    return M


def scalar_stable(K: F, d: int) -> bool:
    """Skalar x⁺ = x + u mit u_k = −K x_{k−d}: exakt stabil? (Polynom z^(d+1) − z^d + K)"""
    if d == 0:
        return schur_stable([F(1), K - 1])
    return schur_stable([F(1), F(-1)] + [F(0)] * (d - 1) + [F(K)])
