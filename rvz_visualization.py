"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rvz_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_roots(roots):
    """Wurzeln des Polynoms P_d in der z-Ebene mit Einheitskreis; Nullwurzeln liegen im Ursprung."""
    t = np.linspace(0, 2 * math.pi, 181)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=C.COLOR_GREY, width=2), hoverinfo="skip", name="Einheitskreis"))
    re = [float(z.real) for z in roots]
    im = [float(z.imag) for z in roots]
    colors = [C.COLOR_BAD if abs(z) >= 1.0 - C.TOL else C.COLOR_MAIN for z in roots]
    fig.add_trace(go.Scatter(x=re, y=im, mode="markers", marker=dict(size=10, color=colors, line=dict(width=1, color="white")), name="Wurzeln",
                             hovertemplate="%{x:.3f} %{y:+.3f}i<extra></extra>"))
    lim = max(1.25, 1.1 * max([abs(z) for z in roots] + [1.0]))
    fig.update_xaxes(title_text="Realteil", range=[-lim, lim], zeroline=True)
    fig.update_yaxes(title_text="Imaginärteil", range=[-lim, lim], zeroline=True)
    return _base(fig, 330)


def build_decay(norms, rho):
    """‖x_k‖ nach dem Stoß x₀ = (1, 0) über den ersten Perioden, logarithmische Achse."""
    ks = list(range(len(norms)))
    ys = [max(v, 1e-12) for v in norms]
    fig = go.Figure(go.Scatter(x=ks, y=ys, mode="lines+markers", line=dict(color=C.COLOR_BAD if rho >= 1 else C.COLOR_MAIN, width=2), marker=dict(size=4),
                               hovertemplate="k = %{x}: %{y:.3g}<extra></extra>", name="‖x_k‖"))
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="‖x_k‖ (logarithmisch)", type="log")
    return _base(fig, 330)


def build_rho_vs_d(rhos, d_now):
    """Spektralradius gegen den Verzug d = 0 … D_MAX bei dem gewählten R; Linie bei ρ = 1."""
    ds = list(range(len(rhos)))
    colors = [C.COLOR_BAD if r >= 1.0 - C.TOL else C.COLOR_MAIN for r in rhos]
    line_w = [3 if d == d_now else 0 for d in ds]
    fig = go.Figure(go.Bar(x=ds, y=rhos, marker=dict(color=colors, line=dict(color="black", width=line_w)), hovertemplate="d = %{x}: ρ = %{y:.3f}<extra></extra>", name="ρ"))
    fig.add_hline(y=1.0, line=dict(color="black", dash="dash"))
    fig.update_xaxes(title_text="Verzug d (Perioden)", dtick=1)
    fig.update_yaxes(title_text="Spektralradius ρ", range=[0, max(1.4, max(rhos) * 1.08)])
    return _base(fig, 300)


def build_krit(rows):
    """R_krit gegen d (logarithmisch) und R des gerade stabilen Entwurfs."""
    ds = [r["d"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=[r["rkrit"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_REF, width=3), name="R_krit(d)",
                             hovertemplate="d = %{x}: R_krit = %{y:.4g}<extra></extra>"))
    fig.update_xaxes(title_text="Verzug d (Perioden)", dtick=1)
    fig.update_yaxes(title_text="kleinstes stabiles R (logarithmisch)", type="log")
    return _base(fig, 320)


def build_decay_bars(rows):
    """Abklingzeit des gerade stabilen Entwurfs (R = 1,2 · R_krit) gegen die Vorhersage (K aus R = 1), logarithmisch."""
    ds = [r["d"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ds, y=[r["decay_soft"] or 0 for r in rows], marker_color=C.COLOR_WARN, name="weicher Entwurf (R = 1,2 · R_krit)",
                         hovertemplate="d = %{x}: %{y} Perioden<extra></extra>"))
    fig.add_trace(go.Bar(x=ds, y=[r["decay_pred"] or 0 for r in rows], marker_color=C.COLOR_MAIN, name="Vorhersage (K aus R = 1)",
                         hovertemplate="d = %{x}: %{y} Perioden<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Verzug d (Perioden)", dtick=1)
    fig.update_yaxes(title_text="Abklingzeit (Perioden, logarithmisch)", type="log")
    return _base(fig, 320)


def build_theta(curve, info):
    """ρ gegen die Rechenzeit θ (Perioden) mit Marke bei θ* und dem gewählten θ."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["theta"], y=curve["rho"], mode="lines", line=dict(color=C.COLOR_REF, width=3), name="ρ(θ)",
                             hovertemplate="θ = %{x:.2f}: ρ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="black", dash="dash"))
    if info["theta_star"] is not None:
        fig.add_trace(go.Scatter(x=[info["theta_star"]], y=[1.0], mode="markers", marker=dict(size=12, color=C.COLOR_BAD, symbol="diamond"), name="θ*",
                                 hovertemplate="θ* = %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[info["theta"]], y=[info["rho"]], mode="markers", marker=dict(size=11, color="black"), name="gewähltes θ",
                             hovertemplate="θ = %{x:.2f}: ρ = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Rechenzeit θ (Perioden)")
    fig.update_yaxes(title_text="Spektralradius ρ", range=[0, 1.6])
    return _base(fig, 320)


def build_theta_star(curve, logR_now):
    """Kritische Rechenzeit θ* gegen log₁₀ R mit der Marke des gewählten R."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["logR"], y=curve["theta_star"], mode="lines+markers", line=dict(color=C.COLOR_REF, width=3), marker=dict(size=5), name="θ*(R)",
                             hovertemplate="log₁₀R = %{x:.2f}: θ* = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=logR_now, line=dict(color="black", dash="dot"))
    fig.update_xaxes(title_text="log₁₀ R")
    fig.update_yaxes(title_text="θ* (Perioden)", rangemode="tozero")
    return _base(fig, 320)


def build_traj(ideal, xs, d, label):
    """Lage und Geschwindigkeit über den Perioden: Entwurf ohne Verzug (gestrichelt) und das gewählte Gesetz; die d blinden Perioden sind schraffiert."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Lage x₁", "Geschwindigkeit x₂"))
    ks = list(range(len(ideal)))
    for r, col in ((1, 0), (2, 1)):
        fig.add_trace(go.Scatter(x=ks, y=ideal[:, col], mode="lines", line=dict(color=C.COLOR_GREY, dash="dash", width=2), name="ohne Verzug (d = 0)", showlegend=(r == 1),
                                 hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"), row=r, col=1)
        if xs is not None:
            fig.add_trace(go.Scatter(x=ks, y=xs[:, col], mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name=label, showlegend=(r == 1),
                                     hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"), row=r, col=1)
    if d > 0:
        for r in (1, 2):
            fig.add_vrect(x0=0, x1=d, fillcolor="rgba(200,120,0,0.15)", line_width=0, row=r, col=1)
    fig.update_xaxes(title_text="Periode k", row=2, col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_bands(rows, b_now):
    """Stabiler b-Bereich mit Vorhersage je d (waagerechte Balken); Linie bei der angenommenen b = 1 und beim gewählten b."""
    fig = go.Figure()
    ds = [r["d"] for r in rows if r["bmin"] is not None]
    fig.add_trace(go.Bar(y=ds, x=[r["bmax"] - r["bmin"] for r in rows if r["bmin"] is not None], base=[r["bmin"] for r in rows if r["bmin"] is not None], orientation="h",
                         marker_color=C.COLOR_MAIN, name="stabil", hovertemplate="d = %{y}: b von %{base:.3f} bis %{x:.3f} (Breite)<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color="black", dash="dash"))
    fig.add_vline(x=b_now, line=dict(color=C.COLOR_BAD, dash="dot"))
    fig.update_yaxes(title_text="Verzug d", dtick=1, autorange="reversed")
    fig.update_xaxes(title_text="wahre Verstärkung b (angenommen: 1)", range=[0, 2.5])
    return _base(fig, 340)


def build_dh_map(m):
    """Spektralradius bei falsch angenommenem Verzug (Zeilen d, Spalten dh − d); über 2 abgeschnitten, Beschriftung mit dem echten Wert."""
    z = np.where(np.isnan(m["rho"]), np.nan, np.minimum(m["rho"], 2.0))
    text = [["" if math.isnan(v) else f"{v:.2f}".replace(".", ",") for v in row] for row in m["rho"]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{o:+d}" if o else "0" for o in m["off"]], y=[f"d = {d}" for d in m["d"]], text=text, texttemplate="%{text}",
                               colorscale=[[0.0, "#2F6B65"], [0.5, "#f3e9b0"], [1.0, "#c44e52"]], zmin=0.0, zmax=2.0, zmid=1.0, showscale=False,
                               hovertemplate="%{y}, dh − d = %{x}: ρ = %{text}<extra></extra>"))
    fig.update_xaxes(title_text="angenommener minus wahrer Verzug (dh − d)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 280)


def build_equilibrium(s):
    """Gleichgewichtslage bei konstanter Störkraft: ohne Verzug, mit Vorhersage, weicher Entwurf und beste statische Verstärkung."""
    labels, vals, cols = ["ohne Verzug (d = 0)", "mit Vorhersage"], [s["x1_ideal"], s["x1_pred"]], [C.COLOR_GREY, C.COLOR_MAIN]
    if s["x1_soft"] is not None:
        labels.append("weicher Entwurf")
        vals.append(s["x1_soft"])
        cols.append(C.COLOR_WARN)
    if s["x1_static"] is not None:
        labels.append("beste statische Verstärkung")
        vals.append(s["x1_static"])
        cols.append(C.COLOR_REF)
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=cols, hovertemplate="%{x}: x₁* = %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title_text="Gleichgewichtslage x₁*", rangemode="tozero")
    return _base(fig, 320)
