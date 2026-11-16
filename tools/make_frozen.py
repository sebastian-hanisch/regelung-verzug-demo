"""Erzeugt data/rvz_frozen.json: die vorgerechneten Messreihen der Demo (Nelder–Mead-Optima, Zufallszählungen, Gitter-Gegenproben).

Aufruf (Wurzelordner des Repos): `python tools/make_frozen.py` (etwa eine bis zwei Minuten) oder `python tools/make_frozen.py --check` (vergleicht mit der Datei).
Zufall nur aus dem ganzzahligen SplitMix64 (`rvz_rng.py`), nie Python-`hash()` oder `random`. Die Funktionen sind so geschrieben, dass tests/ sie in kleinerer
Größe neu rechnen kann (Parameter `n`); die Nelder–Mead-Optima sind keine Beweise für das globale Minimum und plattformabhängig im letzten Bit (Band in den Tests).

Die Messlogik ist neu geschrieben (nicht aus den Planungsskripten `rg_v18_*` kopiert); Zahlen der Vormessung gelten als Erwartung und werden hier neu gemessen."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rvz_dynamics as D  # noqa: E402
from rvz_rng import SplitMix64  # noqa: E402

OUT = ROOT / "data" / "rvz_frozen.json"
SEEDS = {"m5_diag": 1810, "m5_voll": 1811, "kreuzung_R": 1820, "kreuzung_theta": 1821, "vieta": 1822, "falsch_dh": 1823, "toleranz_b": 1824, "exakt_zufall": 1825}


def _uniform(rng: SplitMix64, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


# --- M5: Zufallsentwürfe mit d = 1 ---------------------------------------------------------------------------------------------------

def m5_counts(kind: str, n: int = 4000) -> dict:
    """Q diagonal (Einträge 10^U(−2, 2)) oder voll (G Gᵀ + 10⁻³ I, G aus Normalzahlen), R = 10^U(−3, 4); stabil mit d = 1 (Spektralradius aus P_1)."""
    rng = SplitMix64(SEEDS["m5_" + kind])
    stable = total = skipped = 0
    by_dec: dict[int, list[int]] = {}
    for _ in range(n):
        if kind == "diag":
            Qm = np.diag([10.0 ** _uniform(rng, -2, 2), 10.0 ** _uniform(rng, -2, 2)])
        else:
            g = np.array([[rng.normal(), rng.normal()], [rng.normal(), rng.normal()]])
            Qm = g @ g.T + 1e-3 * np.eye(2)
        lg = _uniform(rng, -3, 4)
        try:
            K = D.lqr_q(Qm, 10.0 ** lg)
        except Exception:
            skipped += 1
            continue
        ok = D.rho_poly(float(K[0, 0]), float(K[0, 1]), 1) < 1.0
        dec = int(math.floor(lg))
        by_dec.setdefault(dec, [0, 0])
        by_dec[dec][1] += 1
        by_dec[dec][0] += int(ok)
        total += 1
        stable += int(ok)
    return {"stable": stable, "total": total, "skipped": skipped, "seed": SEEDS["m5_" + kind], "by_decade": {str(k): v for k, v in sorted(by_dec.items())}}


# --- Gegenproben auf einmaliges Kreuzen ----------------------------------------------------------------------------------------------

def r_grid_crossings(points: int = 4001) -> dict:
    """Gitter log10 R = −3 … 6 (4 001 Punkte): Zahl der Wechsel stabil/instabil je d = 1 … 8 und die Zahl der Entwürfe je d, die stabil sind."""
    logs = np.linspace(-3, 6, points)
    ks = [D.gain(10.0 ** lg) for lg in logs]
    out = {}
    for d in range(1, 9):
        st = np.array([D.rho_poly(k1, k2, d) < 1 for k1, k2 in ks])
        out[str(d)] = {"changes": int(np.sum(st[1:] != st[:-1])), "stable_points": int(st.sum())}
    return {"points": points, "by_d": out}


def r_random_crossings(n: int = 20000) -> dict:
    """n zufällige R = 10^U(−3, 6): stabil genau dann, wenn R > R_krit(d)? Gezählt für d = 1 … 8 (Fälle näher als 10⁻⁹ in log R an R_krit ausgenommen)."""
    rng = SplitMix64(SEEDS["kreuzung_R"])
    rk = {d: D.r_krit(d) for d in range(1, 9)}
    checked = mism = 0
    for _ in range(n):
        lg = _uniform(rng, -3, 6)
        k1, k2 = D.gain(10.0 ** lg)
        for d in range(1, 9):
            if abs(lg - math.log10(rk[d])) < 1e-9:
                continue
            st = D.rho_poly(k1, k2, d) < 1.0
            checked += 1
            mism += int(st != (lg > math.log10(rk[d])))
    return {"n": n, "checked": checked, "mismatches": mism, "seed": SEEDS["kreuzung_R"]}


def theta_crossings(n_R: int = 500, per_R: int = 10) -> dict:
    """Gitter θ = 0 … 3 (3 001 Punkte) für sieben R: Zahl der Wechsel; dazu n_R zufällige R = 10^U(−3, 2) mit je per_R zufälligen θ: stabil genau dann, wenn θ < θ*(R)?"""
    ths = np.linspace(0.0, 3.0, 3001)
    grid = {}
    for R in (100.0, 10.0, 3.0, 1.0, 0.1, 0.01, 0.001):
        K = D.lqr(R, "ZOH")[1]
        st = np.array([D.rho_eig(D.theta_matrix(K, float(t))) < 1 for t in ths])
        grid[str(R)] = int(np.sum(st[1:] != st[:-1]))
    rng = SplitMix64(SEEDS["kreuzung_theta"])
    checked = mism = 0
    for _ in range(n_R):
        R = 10.0 ** _uniform(rng, -3, 2)
        ts = D.theta_krit(R)
        K = D.lqr(R, "ZOH")[1]
        for _ in range(per_R):
            th = _uniform(rng, 0, 3)
            if ts is None or abs(th - ts) < 1e-6:
                continue
            st = D.rho_eig(D.theta_matrix(K, th)) < 1.0
            checked += 1
            mism += int(st != (th < ts))
    return {"grid_changes_by_R": grid, "random": {"n_R": n_R, "per_R": per_R, "checked": checked, "mismatches": mism, "seed": SEEDS["kreuzung_theta"]}}


def largest_stable_d(Rs=(1e5, 1e4, 3e3, 1e3, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.1, 0.01, 0.001), dmax: int = 119) -> dict:
    """Je R das größte d ohne Lücke ab 0 und die Zahl aller stabilen d bis dmax (ohne Lücke heißt: beide Zahlen gleich + 1)."""
    out = {}
    for R in Rs:
        k1, k2 = D.gain(R)
        st = [D.rho_poly(k1, k2, d) < 1 - 1e-9 for d in range(dmax + 1)]
        run = 0
        while run + 1 <= dmax and st[run + 1]:
            run += 1
        out[repr(float(R))] = {"largest_d_without_gap": run, "stable_total": int(sum(st))}
    return {"dmax": dmax, "by_R": out}


# --- C6: beste statische Verstärkung ohne Vorhersage -------------------------------------------------------------------------------------

def best_rho(d: int) -> dict:
    """Kleinster gefundener Spektralradius einer statischen Verstärkung u = −K x_{k−d} (Gitter 101 × 151, dann Nelder–Mead); kein Beweis für das globale Minimum."""
    best, bk = 9.0, (0.0, 0.0)
    for k1 in np.linspace(0.0, 1.0, 101):
        for k2 in np.linspace(0.0, 3.0, 151):
            v = D.rho_poly(float(k1), float(k2), d)
            if v < best:
                best, bk = v, (float(k1), float(k2))
    r = minimize(lambda k: D.rho_poly(k[0], k[1], d), bk, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    return {"k1": float(r.x[0]), "k2": float(r.x[1]), "rho": float(r.fun), "vieta": 2.0 / (d + 2)}


def best_cost(d: int, x0, R: float = 1.0) -> dict:
    """Kleinste gefundene Kosten Σ x'Qx + R u² einer statischen Verstärkung ohne Vorhersage (Nelder–Mead aus dem ρ-optimalen Gewinn und aus dem weichen LQR-Entwurf)."""
    starts = []
    bo = best_rho(d)
    starts.append((bo["k1"], bo["k2"]))
    rk = D.r_krit(d)
    k1, k2 = D.gain(1.2 * rk)
    starts.append((k1, k2))

    def f(k):
        if D.rho_poly(k[0], k[1], d) >= 0.9995:
            return 1e9
        return D.cost_loop([float(k[0]), float(k[1])], d, x0, R)

    best = None
    for s in starts:
        r = minimize(f, s, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return {"k1": float(best.x[0]), "k2": float(best.x[1]), "cost": float(best.fun)}


def best_stationary(d: int, R: float = 1.0) -> dict:
    """Kleinste gefundene stationäre Kosten je Periode (Störkraft N(0, 1)) einer statischen Verstärkung ohne Vorhersage (Nelder–Mead aus dem weichen LQR-Entwurf)."""
    k1, k2 = D.gain(1.2 * D.r_krit(d))

    def f(k):
        if D.rho_poly(k[0], k[1], d) >= 0.9995:
            return 1e9
        return D.stationary_cost([float(k[0]), float(k[1])], d, R)

    r = minimize(f, (k1, k2), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
    return {"k1": float(r.x[0]), "k2": float(r.x[1]), "cost": float(r.fun)}


def static_best(dmax_rho: int = 6, dmax_cost: int = 4) -> dict:
    out = {"rho": {}, "cost_x10": {}, "cost_x01": {}, "stationary": {}}
    for d in range(1, dmax_rho + 1):
        out["rho"][str(d)] = best_rho(d)
    for d in range(1, dmax_cost + 1):
        out["cost_x10"][str(d)] = best_cost(d, (1.0, 0.0))
        out["cost_x01"][str(d)] = best_cost(d, (0.0, 1.0))
        out["stationary"][str(d)] = best_stationary(d)
    return out


def vieta_random(n: int = 20000) -> dict:
    """n zufällige Gewinne (k1 ∈ [0, 2], k2 ∈ [0, 4]) und d = 1 … 8: Verletzungen von ρ ≥ 2/(d + 2); dazu die Abweichung der Wurzelsumme von 2 an 200 Fällen."""
    rng = SplitMix64(SEEDS["vieta"])
    viol = 0
    for _ in range(n):
        d = 1 + rng.below(8)
        k1, k2 = _uniform(rng, 0, 2), _uniform(rng, 0, 4)
        if D.rho_poly(k1, k2, d) < 2.0 / (d + 2) - 1e-9:
            viol += 1
    maxs = 0.0
    for _ in range(200):
        d = 1 + rng.below(8)
        k1, k2 = _uniform(rng, 0, 2), _uniform(rng, 0, 4)
        maxs = max(maxs, abs(float(np.sum(D.roots_wagen(k1, k2, d)).real) - 2.0))
    return {"n": n, "violations": viol, "max_sum_dev": maxs, "seed": SEEDS["vieta"]}


# --- C4/C5 -----------------------------------------------------------------------------------------------------------------------------

def falsch_dh_random(n: int = 2000) -> dict:
    """n zufällige (R, d, dh): R = 10^U(−3, 0) (R ≤ 1), d ∈ 1 … 4, dh ≠ d mit |dh − d| ≤ 2 und dh ≥ 0: Zahl der instabilen Schleifen."""
    rng = SplitMix64(SEEDS["falsch_dh"])
    unstable = tot = 0
    while tot < n:
        R = 10.0 ** _uniform(rng, -3, 0)
        d = 1 + rng.below(4)
        dh = d + (rng.below(5) - 2)
        if dh == d or dh < 0:
            continue
        k1, k2 = D.gain(R)
        unstable += int(D.rho_loop(k1, k2, d, dh) >= 1.0)
        tot += 1
    return {"n": n, "unstable": unstable, "seed": SEEDS["falsch_dh"]}


def tolerance_random(n_R: int = 60) -> dict:
    """n_R zufällige R = 10^U(−2, 1): Breite des stabilen b-Bereichs mit Vorhersage für d = 1 … 5 (Gitter 0,01); gezählt, bei wie vielen R die Breite mit d streng fällt."""
    rng = SplitMix64(SEEDS["toleranz_b"])
    falling = 0
    for _ in range(n_R):
        R = 10.0 ** _uniform(rng, -2, 1)
        k1, k2 = D.gain(R)
        widths = []
        for d in range(1, 6):
            lo, hi, _, _ = D.b_interval(D.pred_gain(k1, k2, d), d, step=0.01)
            widths.append(hi - lo)
        falling += int(all(widths[i + 1] < widths[i] for i in range(4)))
    return {"n_R": n_R, "falling": falling, "seed": SEEDS["toleranz_b"]}


def m9_table() -> dict:
    out = {}
    for R in (1.0, 0.1):
        K = D.lqr(R)[1]
        dare = max(float(np.max(np.abs(D.riccati_gain_n(R, N) - K))) for N in range(1, 13))
        q = {str(N): float(np.max(np.abs(D.riccati_gain_n(R, N, "q") - K))) for N in (1, 2, 5, 12)}
        out[repr(R)] = {"max_dev_dare_terminal": dare, "dev_q_terminal": q}
    return out


def zoh_krit() -> dict:
    return {str(d): {"zoh": D.r_krit(d, "ZOH"), "linie": D.r_krit(d)} for d in (1, 2)}


def main() -> dict:
    t0 = time.time()
    data = {
        "m5_diag": m5_counts("diag"),
        "m5_voll": m5_counts("voll"),
        "r_grid": r_grid_crossings(),
        "r_random": r_random_crossings(),
        "theta": theta_crossings(),
        "largest_stable_d": largest_stable_d(),
        "static_best": static_best(),
        "vieta": vieta_random(),
        "falsch_dh": falsch_dh_random(),
        "tolerance_random": tolerance_random(),
        "m9": m9_table(),
        "zoh_krit": zoh_krit(),
        "r_krit": {str(d): D.r_krit(d) for d in range(1, 9)},
    }
    data["_meta"] = {"seconds": round(time.time() - t0, 1), "hinweis": "vorgerechnet mit tools/make_frozen.py; Nelder-Mead-Optima sind keine Beweise und im letzten Bit plattformabhängig"}
    return data


if __name__ == "__main__":
    result = main()
    text = json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True)
    if "--check" in sys.argv:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        old.pop("_meta", None)
        new = json.loads(text)
        new.pop("_meta", None)
        print("gleich" if old == new else "ABWEICHUNG")
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"geschrieben: {OUT} ({result['_meta']['seconds']} s)")
