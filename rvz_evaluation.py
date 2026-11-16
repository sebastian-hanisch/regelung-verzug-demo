"""Kennzahlen, Verdikte und Auswertungen der sechs Experimente.

Jedes Verdikt ist eine reine Funktion `verdict_*(…) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an
handgesetzten Werten. Alle Aussagen gelten für den Wagen mit Q = I, ganzzahlig bekannten Verzug (Experiment 3: Bruchteil der Periode), ohne Stellgrenzen, in diesem Modell.
Vorgerechnete Zahlen (Nelder–Mead-Optima, Zufallszählungen) kommen aus data/rvz_frozen.json und sind in der App als „vorgerechnet“ gekennzeichnet."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

import numpy as np

import rvz_constants as C
import rvz_dynamics as K
import rvz_exact as X

FROZEN_PATH = Path(__file__).resolve().parent / "data" / "rvz_frozen.json"
NBSP = " "


# --- Formatierung ----------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit geschütztem Leerzeichen als Tausendertrenner (1722 -> 1 722)."""
    return f"{n:,}".replace(",", NBSP)


def pct(x: float, nd: int = 1, sign: bool = False) -> str:
    """Prozent aus einem Anteil (0,339 -> 33,9 %)."""
    return de(100.0 * x, nd, sign) + f"{NBSP}%"


def fmt_R(R: float) -> str:
    """R mit sinnvollen Stellen: 6,85 / 82,8 / 370 / 1 092 / 0,1 / 0,001."""
    if R >= 1000:
        return fmt_int(int(round(R)))
    if R >= 100:
        return f"{R:.0f}"
    if R >= 1:
        s = de(R, 1 if R >= 10 else 2)
        return s.rstrip("0").rstrip(",") if "," in s else s
    return f"{R:.4g}".replace(".", ",")


def fmt_rho(r: float) -> str:
    return de(r, 3)


def R_of(logR: float) -> float:
    return float(10.0 ** round(float(logR), 4))


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


# --- Einstellungen ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    logR: float = C.DEFAULT_LOGR
    d: int = C.DEFAULT_D
    theta: float = C.DEFAULT_THETA
    law: str = C.DEFAULT_LAW
    x0: str = C.DEFAULT_X0
    b: float = C.DEFAULT_B
    dh_off: int = C.DEFAULT_DH_OFF
    w: float = C.DEFAULT_W

    @property
    def R(self) -> float:
        return R_of(self.logR)

    @property
    def dh(self) -> int:
        """Angenommener Verzug des Reglers: d + Versatz, nicht unter 0."""
        return max(self.d + self.dh_off, 0)


# --- Experiment 1: ein Takt Verzug -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Takt:
    R: float
    d: int
    k1: float
    k2: float
    rho0: float                    # ohne Verzug
    rho: float                     # mit d Perioden Verzug
    rhos: tuple[float, ...]        # d = 0 … D_MAX
    rkrit: float | None
    decay: int | None              # Abklingzeit nach x0 = (1, 0); None: instabil oder länger als der Horizont
    peak: float
    norms: tuple[float, ...]       # ‖x_k‖ der ersten PLOT_HORIZON + 1 Perioden
    roots: tuple[complex, ...]


def decay_of(k1: float, k2: float, d: int, dh: int, rho: float, x0=(1.0, 0.0)) -> tuple[int | None, float]:
    """Abklingzeit (letzte Periode über 1 % der Spitze) einer stabilen Schleife; None bei ρ ≥ 1 oder wenn sie den Horizont überschreitet."""
    if rho >= 1.0 - C.TOL:
        return None, float("nan")
    T = int(min(C.DECAY_HORIZON, 14.0 / -math.log(max(rho, 1e-6)) + 200 + 4 * d))
    xs, _ = K.simulate(k1, k2, d, x0, T, dh=dh)
    n, peak = K.decay_time(xs, C.DECAY_FRAC)
    if n >= T - 1:
        return None, peak
    return n, peak


@lru_cache(maxsize=512)
def takt(R: float, d: int) -> Takt:
    k1, k2 = K.gain(R)
    rhos = tuple(K.rho_poly(k1, k2, dd) for dd in range(C.D_MAX + 1))
    rho = K.rho_poly(k1, k2, d)
    decay, peak = decay_of(k1, k2, d, 0, rho)
    xs, _ = K.simulate(k1, k2, d, (1.0, 0.0), C.PLOT_HORIZON, dh=0)
    norms = tuple(float(v) for v in np.linalg.norm(xs, axis=1))
    return Takt(R, d, k1, k2, rhos[0], rho, rhos, K.r_krit(d), decay, peak, norms, tuple(K.roots_wagen(k1, k2, d)))


def largest_stable_d(rhos) -> int:
    """Größtes d ohne Lücke ab 0 mit ρ < 1 (bis zum letzten gemessenen d)."""
    n = 0
    while n + 1 < len(rhos) and rhos[n + 1] < 1.0 - C.TOL:
        n += 1
    return n


def verdict_takt(t: Takt) -> tuple[str, str]:
    if abs(t.rho - 1.0) <= C.TOL and t.d > 0:
        return "info", f"Mit d = {t.d} Perioden Verzug liegt der Entwurf R = {fmt_R(t.R)} auf der Stabilitätsgrenze (ρ = 1 im Band 10⁻⁹)."
    if t.d == 0:
        dec = f", ein Stoß x₀ = (1, 0) klingt in {t.decay} Perioden ab" if t.decay is not None else ""
        return "success", f"Ohne Verzug hat der Entwurf R = {fmt_R(t.R)} den Spektralradius ρ = {fmt_rho(t.rho)}{dec}. Mit d ≥ 1 sehen Sie den Verzug (Seitenleiste)."
    if t.rho < 1.0:
        dec = f", Abklingzeit {fmt_int(t.decay)} Perioden" if t.decay is not None else ", Abklingzeit länger als der Horizont"
        return "success", (f"Mit d = {t.d} Perioden Verzug hält der Entwurf R = {fmt_R(t.R)}: ρ = {fmt_rho(t.rho)} statt {fmt_rho(t.rho0)} ohne Verzug{dec}. "
                           f"Er hält ab R_krit(d) = {fmt_R(t.rkrit)}.")
    rk = f" Erst ab R = {fmt_R(t.rkrit)} hält d = {t.d}." if t.rkrit is not None else ""
    return "warning", f"Mit d = {t.d} Perioden Verzug ist der Entwurf R = {fmt_R(t.R)} instabil: ρ = {fmt_rho(t.rho)} statt {fmt_rho(t.rho0)} ohne Verzug.{rk}"


@lru_cache(maxsize=16)
def hand_example(d: int) -> dict:
    """Handbeispiel in Brüchen: gerundeter Gewinn K = (2/5, 6/5), Polynom P_d, Schur–Cohn, Bahnen mit und ohne Vorhersage, Gleichgewicht bei w = 1."""
    k1, k2 = F(2, 5), F(6, 5)
    poly = K.poly_wagen(k1, k2, d)
    xs_n, _ = X.trajectory(k1, k2, d, (1, 0), 4, predict=False)
    xs_p, _ = X.trajectory(k1, k2, d, (1, 0), 4, predict=True)
    return {
        "poly": poly,
        "stable": X.schur_stable(poly),
        "rho": float(np.max(np.abs(np.roots([float(c) for c in poly])))),
        "roots_sum": X.poly_roots_sum(poly) if d >= 1 else None,
        "x_nopred": [tuple(v) for v in xs_n],
        "x_pred": [tuple(v) for v in xs_p],
        "x1_pred": X.equilibrium_x1(k1, k2, d, True),
        "x1_nopred": X.equilibrium_x1(k1, k2, d, False),
    }


def frac_str(v) -> str:
    v = F(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def poly_str(coeffs) -> str:
    """Polynom aus Bruchkoeffizienten, höchste Potenz zuerst, als Text mit Unicode-Exponenten."""
    sup = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
    n = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        c = F(c)
        if c == 0:
            continue
        p = n - i
        mono = "" if p == 0 else ("z" if p == 1 else "z" + "".join(sup[ch] for ch in str(p)))
        mag = abs(c)
        coef = "" if (mag == 1 and p > 0) else frac_str(mag)
        term = f"{coef}{mono}" if coef or mono else "1"
        sign = "−" if c < 0 else "+"
        parts.append((sign, term))
    s = ""
    for k, (sign, term) in enumerate(parts):
        s += (("−" if sign == "−" else "") if k == 0 else f" {sign} ") + term
    return s


# --- Skalar ----------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=16)
def scalar_rows(dmax: int = 5) -> list[dict]:
    Kg = K.scalar_gain()
    rows = []
    for dd in range(dmax + 1):
        rho = K.scalar_rho(Kg, dd)
        lg = K.lager_grenze(dd + 1)
        rows.append({"d": dd, "rho": rho, "lager_grenze": lg, "ratio": Kg / lg, "on_boundary": abs(rho - 1.0) <= C.TOL,
                     "stable_35": X.scalar_stable(F(3, 5), dd), "stable_58": X.scalar_stable(F(5, 8), dd)})
    return rows


def verdict_scalar(row: dict) -> tuple[str, str]:
    d, rho = row["d"], row["rho"]
    if row["on_boundary"]:
        return "info", (f"Skalar, d = {d}: der LQR-Gewinn K = 1/φ = 0,618 liegt genau auf der Stabilitätsgrenze (ρ = 1 im Band 10⁻⁹); sie ist die des Lagers ohne Leitung mit L = {d + 1} "
                        f"(2 sin(π/10) = 0,618), eine Gleichheit zweier Zahlen, keine Erklärung.")
    if rho < 1.0:
        return "success", f"Skalar, d = {d}: ρ = {fmt_rho(rho)}, die Schleife hält. Die Lager-Grenze für L = {d + 1} ist {de(row['lager_grenze'], 3)}, K = 0,618 liegt darunter."
    return "warning", f"Skalar, d = {d}: ρ = {fmt_rho(rho)}, die Schleife ist instabil. Die Lager-Grenze für L = {d + 1} ist {de(row['lager_grenze'], 3)}, K = 0,618 liegt darüber."


# --- Experiment 2: weich genug, aber langsam -------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def krit_rows() -> list[dict]:
    """d = 1 … 8: R_krit, der gerade stabile Entwurf R = 1,2 · R_krit mit seiner Abklingzeit und die Vorhersage mit K(R = 1) (x₀ = (1, 0))."""
    k1p, k2p = K.gain(C.EVAL_R)
    rows = []
    for d in range(1, C.D_MAX + 1):
        rk = K.r_krit(d)
        Rs = C.SOFT_FACTOR * rk
        k1, k2 = K.gain(Rs)
        rho = K.rho_poly(k1, k2, d)
        dec, _ = decay_of(k1, k2, d, 0, rho)
        rho_p = K.rho_loop(k1p, k2p, d, d)
        dec_p, _ = decay_of(k1p, k2p, d, d, K.rho_poly(k1p, k2p, 0))   # die Vorhersage-Schleife hat die Wurzeln des Entwurfs ohne Verzug
        rows.append({"d": d, "rkrit": rk, "R_soft": Rs, "rho_soft": rho, "decay_soft": dec, "decay_pred": dec_p, "rho_pred": rho_p})
    return rows


def verdict_krit(d: int, rows: list[dict]) -> tuple[str, str]:
    if d == 0:
        return "info", "Ohne Verzug (d = 0) gibt es keine Grenze: alle 4 001 gemessenen Entwürfe (R = 10⁻³ bis 10⁶, Q = I) sind stabil. Wählen Sie d ≥ 1."
    r = rows[d - 1]
    soft = f"braucht {fmt_int(r['decay_soft'])} Perioden" if r["decay_soft"] is not None else "klingt innerhalb des Horizonts nicht ab"
    return "info", (f"d = {d}: R_krit = {fmt_R(r['rkrit'])}. Der gerade stabile Entwurf (R = {fmt_R(r['R_soft'])}) {soft}, mit Vorhersage (K aus R = 1) {r['decay_pred']} "
                    f"Perioden (Wagen, Q = I, Stoß x₀ = (1, 0), Abklingzeit bis 1 % der Spitze).")


# --- Experiment 3: Rechenzeit als Bruchteil der Periode -------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def theta_curve(R: float) -> dict:
    ths = np.round(np.arange(C.THETA_MIN, C.THETA_MAX + 1e-9, 0.01), 10)
    K1 = K.lqr(R, "ZOH")[1]
    rhos = [K.rho_eig(K.theta_matrix(K1, float(t))) for t in ths]
    return {"theta": ths, "rho": np.array(rhos)}


@lru_cache(maxsize=1)
def theta_star_curve() -> dict:
    logs = np.arange(-3.0, 2.0 + 1e-9, 0.25)
    return {"logR": logs, "theta_star": np.array([K.theta_krit(R_of(lg)) or float("nan") for lg in logs])}


def theta_info(R: float, theta: float) -> dict:
    ts = K.theta_krit(R)
    K1 = K.lqr(R, "ZOH")[1]
    rho = K.rho_eig(K.theta_matrix(K1, theta))
    ang = None
    if ts is not None:
        ev = np.linalg.eigvals(K.theta_matrix(K1, ts))
        dom = ev[int(np.argmax(np.abs(ev)))]
        ang = float(np.degrees(np.angle(dom)))
    return {"R": R, "theta": theta, "rho": rho, "theta_star": ts, "angle": ang}


def verdict_theta(info: dict) -> tuple[str, str]:
    R, th, rho, ts = info["R"], info["theta"], info["rho"], info["theta_star"]
    if ts is None:
        return "success", f"R = {fmt_R(R)}: bis θ = 3 Perioden Rechenzeit stabil (ρ = {fmt_rho(rho)} bei θ = {de(th, 2)})."
    if rho < 1.0:
        return "success", (f"θ = {de(th, 2)} Perioden Rechenzeit hält der Entwurf R = {fmt_R(R)} (ρ = {fmt_rho(rho)}); die Grenze liegt bei θ* = {de(ts, 2)} Perioden "
                           f"(Modell B = [½, 1]ᵀ, exakt abgetastet).")
    return "warning", (f"θ = {de(th, 2)} Perioden Rechenzeit hält der Entwurf R = {fmt_R(R)} nicht (ρ = {fmt_rho(rho)}); die Grenze liegt bei θ* = {de(ts, 2)} Perioden "
                       f"(Modell B = [½, 1]ᵀ, exakt abgetastet).")


# --- Experiment 4: mit Vorhersage ------------------------------------------------------------------------------------------------------

def static_entry(kind: str, d: int) -> dict | None:
    return load_frozen()["static_best"][kind].get(str(d))


@dataclass(frozen=True)
class LawRow:
    key: str
    label: str
    k1: float | None
    k2: float | None
    dh: int
    rho: float | None
    decay: int | None
    cost: float | None
    extra: float | None            # Mehrkosten gegen den Entwurf ohne Verzug (Anteil)
    note: str = ""


def law_rows(R: float, d: int, x0name: str) -> tuple[float, list[LawRow]]:
    """Die vier Gesetze bei Verzug d (Auswertungsgewicht = R): Spektralradius, Abklingzeit, Kosten Σ x'Qx + R u² und Mehrkosten gegen den Entwurf ohne Verzug (J0 = x₀'P x₀)."""
    x0 = C.X0_VALUES[x0name]
    J0 = K.cost_ideal(R, x0)
    k1, k2 = K.gain(R)
    rows: list[LawRow] = []

    # Vorhersage
    rho_p = K.rho_loop(k1, k2, d, d)
    dec_p, _ = decay_of(k1, k2, d, d, rho_p)
    Jp = K.cost_pred_closed(R, d, x0)
    rows.append(LawRow(C.LAWS[0], C.LAWS[0], k1, k2, d, rho_p, dec_p, Jp, Jp / J0 - 1.0))
    # ohne Vorhersage, gleicher Gewinn
    rho_n = K.rho_poly(k1, k2, d)
    dec_n, _ = decay_of(k1, k2, d, 0, rho_n)
    Jn = K.cost_loop([k1, k2], d, x0, R) if rho_n < 1.0 - C.TOL else None
    rows.append(LawRow(C.LAWS[1], C.LAWS[1], k1, k2, 0, rho_n, dec_n, Jn, None if Jn is None else Jn / J0 - 1.0,
                       "" if Jn is not None else "instabil: Kosten unendlich"))
    if d >= 1:
        rk = K.r_krit(d)
        k1l, k2l = K.gain(C.SOFT_FACTOR * rk)
        rho_l = K.rho_poly(k1l, k2l, d)
        dec_l, _ = decay_of(k1l, k2l, d, 0, rho_l)
        Jl = K.cost_loop([k1l, k2l], d, x0, R)
        rows.append(LawRow(C.LAWS[2], f"Weicher Entwurf (R = {fmt_R(C.SOFT_FACTOR * rk)})", k1l, k2l, 0, rho_l, dec_l, Jl, Jl / J0 - 1.0))
        ent_rho = static_entry("rho", d)
        if ent_rho is not None:
            k1s, k2s = ent_rho["k1"], ent_rho["k2"]
            rho_s = K.rho_poly(k1s, k2s, d)
            dec_s, _ = decay_of(k1s, k2s, d, 0, rho_s)
            ent_c = static_entry("cost_x10" if x0name == "(1, 0)" else "cost_x01", d)
            if ent_c is not None and abs(R - C.EVAL_R) < 1e-12:
                rows.append(LawRow(C.LAWS[3], C.LAWS[3], k1s, k2s, 0, rho_s, dec_s, ent_c["cost"], ent_c["cost"] / J0 - 1.0,
                                   "vorgerechnet (je Kennzahl neu abgestimmt): ρ und Abklingzeit vom ρ-optimalen Gewinn, Kosten vom kostenoptimalen"))
            else:
                rows.append(LawRow(C.LAWS[3], C.LAWS[3], k1s, k2s, 0, rho_s, dec_s, None, None,
                                   "vorgerechnet: Kosten nur für R = 1 und d ≤ 4 (ρ und Abklingzeit vom ρ-optimalen Gewinn)"))
        else:
            rows.append(LawRow(C.LAWS[3], C.LAWS[3], None, None, 0, None, None, None, None, "nicht vorgerechnet (d ≤ 6)"))
    return J0, rows


def law_trajectory(R: float, d: int, row: LawRow, x0name: str, T: int = C.PLOT_HORIZON):
    """Bahn x_0 … x_T der gewählten Zeile und die des Entwurfs ohne Verzug (d = 0)."""
    x0 = C.X0_VALUES[x0name]
    k1, k2 = K.gain(R)
    ideal, _ = K.simulate(k1, k2, 0, x0, T)
    if row.k1 is None:
        return ideal, None
    xs, us = K.simulate(row.k1, row.k2, d, x0, T, dh=row.dh)
    return ideal, xs


def verdict_vorhersage(d: int, x0name: str, rows: list[LawRow], R: float, decay0: int | None = None) -> tuple[str, str]:
    """decay0: Abklingzeit ohne Verzug bei demselben R (Stoß x₀ = (1, 0)); None: nicht genannt."""
    if d == 0:
        return "info", "Ohne Verzug (d = 0) sind alle vier Gesetze dasselbe Gesetz u = −K x. Wählen Sie d ≥ 1 (Seitenleiste)."
    pred, nop = rows[0], rows[1]
    ref = "" if decay0 is None else f" (ohne Verzug {decay0} bei R = {fmt_R(R)})"
    s = (f"d = {d}, x₀ = {x0name}: Mit Vorhersage kostet der Verzug {pct(pred.extra, 1, True)} gegenüber dem Entwurf ohne Verzug, die Abklingzeit ist {pred.decay} Perioden{ref}. ")
    if nop.rho >= 1.0 - C.TOL:
        s += f"Derselbe Gewinn K ohne Vorhersage ist instabil (ρ = {fmt_rho(nop.rho)}). "
    else:
        s += f"Derselbe Gewinn K ohne Vorhersage hält (ρ = {fmt_rho(nop.rho)}), kostet aber {pct(nop.extra, 1, True)}. "
    stat = next((r for r in rows if r.key == C.LAWS[3]), None)
    if stat is not None and stat.extra is not None:
        s += f"Die beste statische Verstärkung braucht {stat.decay} Perioden und kostet {pct(stat.extra, 1, True)}. "
    s += f"Jede statische Verstärkung ohne Vorhersage hat ρ ≥ 2/(d + 2) = {fmt_rho(2.0 / (d + 2))}; Deadbeat (ρ = 0) gibt es nur ohne Verzug."
    return "info", s


# --- Experiment 5: wenn das Modell nicht stimmt ----------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def tolerance_bands(R: float) -> list[dict]:
    """Stabiler b-Bereich mit Vorhersage (dh = d) für d = 0 … D_MAX bei diesem R (Gitter 0,01 … 6, Schritt 0,002, Ränder per Bisektion)."""
    k1, k2 = K.gain(R)
    rows = []
    for d in range(C.D_MAX + 1):
        bmin, bmax, runs, edge = K.b_interval(K.pred_gain(k1, k2, d), d)
        rows.append({"d": d, "bmin": bmin, "bmax": bmax, "runs": runs, "edge": edge})
    return rows


@lru_cache(maxsize=64)
def dh_map(R: float) -> dict:
    """Spektralradius bei falsch angenommenem Verzug: d = 1 … 4 gegen dh = d − 2 … d + 2 (dh ≥ 0); Zahl der stabilen Zellen mit dh ≠ d."""
    k1, k2 = K.gain(R)
    ds = (1, 2, 3, 4)
    offs = C.DH_OFFSETS
    grid = np.full((len(ds), len(offs)), np.nan)
    stable = cells = 0
    for i, d in enumerate(ds):
        for j, o in enumerate(offs):
            dh = d + o
            if dh < 0:
                continue
            r = K.rho_loop(k1, k2, d, dh)
            grid[i, j] = r
            if o != 0:
                cells += 1
                stable += int(r < 1.0)
    return {"d": ds, "off": offs, "rho": grid, "stable_off_diagonal": stable, "cells_off_diagonal": cells}


@lru_cache(maxsize=1)
def static_b_rows() -> list[dict]:
    """b-Bereich der besten statischen Verstärkung ohne Vorhersage (vorgerechnete Gewinne, d = 1 … 4)."""
    rows = []
    for d in range(1, 5):
        e = static_entry("rho", d)
        bmin, bmax, _, _ = K.b_interval([e["k1"], e["k2"]], d)
        rows.append({"d": d, "bmin": bmin, "bmax": bmax, "rho": e["rho"]})
    return rows


def modell_info(R: float, d: int, dh: int, b: float) -> dict:
    k1, k2 = K.gain(R)
    rho = K.rho_loop(k1, k2, d, dh, b)
    band = tolerance_bands(R)[d]
    return {"R": R, "d": d, "dh": dh, "b": b, "rho": rho, "bmin": band["bmin"], "bmax": band["bmax"]}


def verdict_modell(info: dict) -> tuple[str, str]:
    R, d, dh, b, rho = info["R"], info["d"], info["dh"], info["b"], info["rho"]
    if dh != d:
        kind = "success" if rho < 1.0 else "warning"
        return kind, (f"Der Regler nimmt dh = {dh} an, wahr ist d = {d} (b = {de(b, 2)}): ρ = {fmt_rho(rho)}, die Schleife {'hält' if rho < 1.0 else 'ist instabil'} "
                      f"(R = {fmt_R(R)}).")
    lo, hi = info["bmin"], info["bmax"]
    rng = "" if lo is None else f" Stabil für b von {de(lo, 2)} bis {de(hi, 2)} (Gitter 0,01 bis 6)."
    kind = "success" if rho < 1.0 else "warning"
    return kind, f"Mit Vorhersage, d = {d}, wahrem b = {de(b, 2)} (angenommen 1): ρ = {fmt_rho(rho)}, die Schleife {'hält' if rho < 1.0 else 'ist instabil'}.{rng}"


# --- Experiment 6: Störkraft und Rauschen ----------------------------------------------------------------------------------------------

def stoer_info(R: float, d: int, w: float) -> dict:
    k1, k2 = K.gain(R)
    G = K.pred_gain(k1, k2, d)
    x1_ideal = K.equilibrium_formula(k1, k2, 0, w, predict=False)
    x1_pred = K.equilibrium_x1(G, d, w) if w != 0 else 0.0
    out = {"R": R, "d": d, "w": w, "x1_ideal": x1_ideal, "x1_pred": x1_pred, "x1_pred_formula": K.equilibrium_formula(k1, k2, d, w, True),
           "factor": (x1_pred / x1_ideal) if x1_ideal else None}
    if d >= 1:
        k1l, k2l = K.gain(C.SOFT_FACTOR * K.r_krit(d))
        out["x1_soft"] = w / k1l
        e = static_entry("rho", d)
        out["x1_static"] = w / e["k1"] if e else None
    else:
        out["x1_soft"] = out["x1_static"] = None
    out["stat_ideal"] = K.stationary_cost(K.pred_gain(k1, k2, 0), 0, R)
    out["stat_pred"] = K.stationary_cost(G, d, R)
    es = static_entry("stationary", d) if d >= 1 else None
    out["stat_static"] = es["cost"] if (es is not None and abs(R - C.EVAL_R) < 1e-12) else None
    return out


def verdict_stoer(s: dict) -> tuple[str, str]:
    d, w, R = s["d"], s["w"], s["R"]
    if w == 0:
        return "info", (f"Ohne Störkraft (w = 0) gibt es keinen Dauerfehler. Streuung je Periode unter Rauschen N(0, 1): {de(s['stat_pred'], 2)} mit Vorhersage "
                        f"(d = {d}) gegen {de(s['stat_ideal'], 2)} ohne Verzug (R = {fmt_R(R)}).")
    if d == 0:
        return "info", (f"Ohne Verzug (d = 0): Gleichgewichtslage x₁* = {de(s['x1_ideal'], 2)} bei w = {de(w, 1)}, Streuung {de(s['stat_ideal'], 2)} je Periode (R = {fmt_R(R)}). "
                        f"Mit d ≥ 1 sehen Sie den Dauerfehler der Vorhersage.")
    txt = (f"d = {d}, R = {fmt_R(R)}, w = {de(w, 1)}: Gleichgewichtslage {de(s['x1_pred'], 2)} mit Vorhersage gegen {de(s['x1_ideal'], 2)} ohne Verzug"
           f" (Faktor {de(s['factor'], 2)})")
    if s["x1_static"] is not None:
        txt += f", {de(s['x1_static'], 1)} bei der besten statischen Verstärkung"
    txt += ". Die Vorhersage kennt keine Störung (kein Störungsmodell): gegenüber d = 0 vergrößert sie den Dauerfehler. "
    txt += f"Streuung je Periode unter Rauschen N(0, 1): {de(s['stat_pred'], 2)} mit Vorhersage gegen {de(s['stat_ideal'], 2)} ohne Verzug."
    return "info", txt


@lru_cache(maxsize=32)
def mc_run(R: float, d: int, periods: int, seed: int) -> dict:
    k1, k2 = K.gain(R)
    G = K.pred_gain(k1, k2, d)
    mc = K.monte_carlo_cost(G, d, R, periods, seed)
    lyap = K.stationary_cost(G, d, R)
    return {"mc": mc, "lyap": lyap, "dev": mc / lyap - 1.0, "periods": periods, "seed": seed, "d": d, "R": R}


def verdict_mc(m: dict) -> tuple[str, str]:
    dev = abs(m["dev"])
    base = (f"Monte-Carlo mit Zufalls-Seed {m['seed']}, {fmt_int(m['periods'])} Perioden, d = {m['d']}, R = {fmt_R(m['R'])}: {de(m['mc'], 2)} gegen {de(m['lyap'], 2)} aus der "
            f"Lyapunov-Rechnung (Abstand {de(100 * dev, 2)}{NBSP}%).")
    if dev <= C.MC_BAND:
        return "success", base + f" Innerhalb des Bandes von {int(C.MC_BAND * 100)}{NBSP}%."
    return "warning", base + f" Außerhalb des Bandes von {int(C.MC_BAND * 100)}{NBSP}%: mehr Perioden wählen."


# --- Zufallsentwürfe (vorgerechnet) ---------------------------------------------------------------------------------------------------

def m5_text() -> str:
    f = load_frozen()
    dg, vl = f["m5_diag"], f["m5_voll"]
    return (f"Zu Q = I: bei zufälligem Q (diagonal, Einträge 10^U(−2, 2)) und R = 10^U(−3, 4) hält d = 1 in {fmt_int(dg['stable'])} von {fmt_int(dg['total'])} Entwürfen "
            f"(volles zufälliges Q: {fmt_int(vl['stable'])} von {fmt_int(vl['total'])}); vorgerechnet, ganzzahliger SplitMix64.")


# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------

def verdict_kern1(R: float, rhos, theta_star) -> tuple[str, str]:
    n = largest_stable_d(rhos)
    last = len(rhos) - 1
    ints = (f"ganzzahlig bis d = {n} Perioden (ab d = {n + 1} instabil, bis d = {last} geprüft)" if n < last
            else f"ganzzahlig bis d = {last} Perioden (Reglergrenze; weiter nicht gezeigt)")
    ts = "bis θ = 3 Perioden stabil" if theta_star is None else f"als Bruchteil der Periode bis θ* = {de(theta_star, 2)} (Modell B = [½, 1]ᵀ)"
    return "info", f"Entwurf R = {fmt_R(R)} hält {ints}; {ts}. Wagen, Q = I, ohne Stellgrenzen."


def verdict_kern2(d: int, rows: list[dict]) -> tuple[str, str]:
    if d == 0:
        return "info", "Ohne Verzug sind alle gemessenen Entwürfe stabil (R = 10⁻³ bis 10⁶, Q = I); wählen Sie d ≥ 1, um den Preis des weichen Entwurfs zu sehen."
    r = rows[d - 1]
    soft = f"{fmt_int(r['decay_soft'])} Perioden" if r["decay_soft"] is not None else "mehr als den Horizont"
    return "info", (f"Für d = {d} muss R mindestens {fmt_R(r['rkrit'])} sein; der gerade stabile Entwurf (R = {fmt_R(r['R_soft'])}) braucht {soft} bis zur Ruhe, ohne Verzug "
                    f"(R = 1) sind es 5.")


def verdict_kern3(d: int, J_extra: float | None, lo: float | None, hi: float | None, lo0: float | None, hi0: float | None, R: float) -> tuple[str, str]:
    if d == 0 or J_extra is None or lo is None:
        return "info", "Ohne Verzug braucht es keine Vorhersage; wählen Sie d ≥ 1."
    return "info", (f"Vorhersage bei d = {d} (R = {fmt_R(R)}, x₀ = (1, 0)): Mehrkosten {pct(J_extra, 1, True)}; sie verlangt ein genaues Modell: stabil nur für b von {de(lo, 2)} bis "
                    f"{de(hi, 2)} (ohne Verzug von {de(lo0, 2)} bis {de(hi0, 2)}).")
