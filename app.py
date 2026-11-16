"""Verzug: wann ein Regelgesetz den Takt nicht mehr hält - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Thema „Regelung (von LQR bis MPC)“: ein Wagen (Doppelintegrator, Q = I) mit einem LQR-Gesetz, dessen Rechenzeit d Perioden beträgt. Wie viel Verzug hält der Entwurf,
was kostet es, ihn weich genug zu machen, und was bringt die Vorhersage des Zustands? Alles eigene Herleitung und Messung in diesem Modell (rvz_dynamics.py).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import rvz_constants as C
import rvz_evaluation as ev
from rvz_evaluation import Settings, de, fmt_int, fmt_R, fmt_rho, pct, poly_str, frac_str
from rvz_layout import dunkles_design, kopf_css
from rvz_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from rvz_visualization import (
    build_bands,
    build_decay,
    build_decay_bars,
    build_dh_map,
    build_equilibrium,
    build_krit,
    build_roots,
    build_rho_vs_d,
    build_theta,
    build_theta_star,
    build_traj,
)

st.set_page_config(page_title="Verzug – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _takt(R, d):
    return ev.takt(R, d)


@st.cache_data(show_spinner="Rechne die Grenzen R_krit(d) für d = 1 bis 8 …")
def _krit_rows():
    return ev.krit_rows()


@st.cache_data(show_spinner=False)
def _hand(d):
    return ev.hand_example(d)


@st.cache_data(show_spinner=False)
def _scalar():
    return ev.scalar_rows(5)


@st.cache_data(show_spinner="Rechne ρ(θ) …")
def _theta(R, theta):
    return ev.theta_curve(R), ev.theta_info(R, theta)


@st.cache_data(show_spinner="Rechne θ*(R) über R …")
def _theta_star_curve():
    return ev.theta_star_curve()


@st.cache_data(show_spinner=False)
def _laws(R, d, x0):
    return ev.law_rows(R, d, x0)


@st.cache_data(show_spinner="Rechne die stabilen Bereiche von b für d = 0 bis 8 …")
def _bands(R):
    return ev.tolerance_bands(R)


@st.cache_data(show_spinner=False)
def _dh_map(R):
    return ev.dh_map(R)


@st.cache_data(show_spinner=False)
def _static_b():
    return ev.static_b_rows()


@st.cache_data(show_spinner=False)
def _stoer(R, d, w):
    return ev.stoer_info(R, d, w)


@st.cache_data(show_spinner="Monte-Carlo-Lauf …")
def _mc(R, d, periods, seed):
    return ev.mc_run(R, d, periods, seed)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rvz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏱️ Verzug: wann ein Regelgesetz den Takt nicht mehr hält")
st.markdown(
    """
Ein Regelgesetz, das **ohne Verzug** gerechnet wurde, läuft hier mit **d Perioden Rechenzeit**: Die Stellgröße steht erst d Takte nach der Messung bereit. Die Demo misst am **Wagen**
(Doppelintegrator, Q = I, ohne Stellgrenzen), **wie viel Verzug ein LQR-Entwurf hält**, was es kostet, den Entwurf dafür **weich** zu machen, und was die **Vorhersage** des Zustands bringt und
vom Modell verlangt. Die Demo gehört zum Thema „Regelung (von LQR bis MPC)“; Verzug am Lager steht in
[`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo), dort ohne Reglerentwurf.
Alle Formeln sind **eigene Herleitung und Messung in diesem Modell, bei diesen Parametern**; eine Neuheit wird nicht behauptet.

Weiter unten: die **Kernfragen** mit Verdikt, sechs Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/regelung-verzug-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich einen Wagen vor, den ein Regelgesetz an eine Markierung bringen soll. Das Gesetz schaut auf Lage und Geschwindigkeit und legt fest, wie stark geschoben wird. Entworfen wurde es, als dauerte das Rechnen nichts. In der Wirklichkeit steht der Schub erst einige Perioden nach der Messung bereit: Das ist der **Verzug**. Die Demo misst an diesem Wagen (ohne Grenzen für den Schub, ohne Messrauschen), wie viel Verzug das Gesetz hält, was es kostet, es zurückhaltend genug einzustellen, und was die **Vorhersage** bringt: Der Regler rechnet aus dem verspäteten Bild und seinen eigenen Schüben den heutigen Stand voraus. Das hilft nur, wenn Modell und Verzug stimmen. Die Aussagen gelten für dieses Rechenmodell, nicht für eine echte Anlage.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/regelung-verzug-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert Verzug", expanded=True):
    st.markdown(
        """
1. **Entwurf ohne Verzug:** Der Gewinn K kommt aus der diskreten Riccati-Gleichung (DARE) des Wagens mit Q = I und dem Kostengewicht R; das Gesetz ist u = −K x.
2. **Verzug:** Mit d Perioden Rechenzeit gilt **u_k = −K x_(k−d)**; bis k = d − 1 liegt keine Messung vor, dort ist u = 0. **d = 0 heißt „Rechnen dauert nichts“**; u_k wirkt immer auf x_(k+1)
   (das ist die „eine Periode Wirkverzug“ der Lager-Demos, dort L = 1; d Perioden Verzug entsprechen dort L = d + 1). Stellverzug und Messverzug haben ohne Vorhersage dieselben Wurzelbeträge (gemessen, siehe README).
3. **Wurzeln:** Die Schleife hat das Polynom **P_d(z) = z^d (z − 1)² + k₂ (z − 1) + k₁** (K = (k₁, k₂), eigene Herleitung); sie hält, wenn alle Wurzeln im Einheitskreis liegen (Spektralradius ρ < 1).
4. **Vorhersage:** Der Regler rechnet den Zustand um die d Perioden voraus (aus x_(k−d) und seinen eigenen Stellgrößen) und setzt u = −K x̂. Im exakten Modell ist die Schleife dann wieder die ohne Verzug,
   übrig bleiben d blinde Perioden. Sie verlangt dafür das richtige Modell und den richtigen Verzug (Experiment 5).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wagen und Entwurf**")
    logR = st.slider("Kostengewicht R (log₁₀)", *bounds("logR_slider"), step=C.LOGR_STEP, key="logR_slider",
                     help="Entwurf ohne Verzug: Kosten Σ x'Qx + R u² mit Q = I. Großes R heißt weicher Eingriff (kleiner Gewinn K), kleines R heißt kräftiger.")
    st.caption(f"R = {fmt_R(ev.R_of(logR))}")
    d = st.slider("Verzug d (Perioden Rechenzeit)", *bounds("d_slider"), step=1, key="d_slider",
                  help="d = 0: das Rechnen dauert nichts. d = 1: die Stellgröße steht eine Periode nach der Messung bereit.")
    st.caption("Experimentbezogene Regler stehen bei ihrem Experiment.")

# Regler der Experimente (im Hauptbereich) werden unten gezeichnet; ihre Werte stehen schon im Session State (init_session_state_defaults).
S = Settings(
    logR=float(logR), d=int(d), theta=float(st.session_state["theta_slider"]), law=st.session_state["law_select"], x0=st.session_state["x0_select"],
    b=float(st.session_state["b_slider"]), dh_off=int(st.session_state["dh_select"]), w=float(st.session_state["w_slider"]),
)
R = S.R
t = _takt(R, S.d)
krit = _krit_rows()
tinfo_curve, tinfo = _theta(R, S.theta)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Spektralradius ρ bei d = {S.d}", fmt_rho(t.rho), delta=f"{de(t.rho - t.rho0, 3, True)} gegen d = 0", delta_color="inverse",
          help="Größter Wurzelbetrag der Schleife; unter 1 hält sie. Delta: mein Wert minus der Wert ohne Verzug.")
m2.metric("Kleinstes stabiles R bei diesem d", fmt_R(t.rkrit) if t.rkrit is not None else "—", delta="Wagen, Q = I" if t.rkrit is not None else "ohne Verzug keine Grenze", delta_color="off")
m3.metric("Kritische Rechenzeit θ* (Perioden)", de(tinfo["theta_star"], 2) if tinfo["theta_star"] is not None else "über 3", delta=f"bei R = {fmt_R(R)}, B = [½, 1]ᵀ", delta_color="off")
laws_J0, laws_rows = _laws(R, S.d, "(1, 0)")
if S.d >= 1:
    m4.metric("Mehrkosten mit Vorhersage", pct(laws_rows[0].extra, 1, True), delta="Stoß x₀ = (1, 0), gegen d = 0", delta_color="off")
else:
    m4.metric("Mehrkosten mit Vorhersage", "—", delta="ohne Verzug keine", delta_color="off")
st.plotly_chart(build_rho_vs_d(list(t.rhos), S.d), width="stretch", key="main_chart")
st.caption(f"Spektralradius des Regelkreises gegen den Verzug d bei R = {fmt_R(R)} (Wagen, Q = I); rot: ρ ≥ 1, die Schleife hält nicht. Der gewählte Verzug ist umrandet.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie viel Rechenzeit lässt ein Entwurf zu?**")
kind, text = ev.verdict_kern1(R, t.rhos, tinfo["theta_star"])
getattr(st, kind)(text)
st.markdown("**2. Was kostet es, weich genug zu sein?**")
kind, text = ev.verdict_kern2(S.d, krit)
getattr(st, kind)(text)
st.markdown("**3. Was bringt die Vorhersage, und was verlangt sie vom Modell?**")
bands = _bands(R)
if S.d >= 1:
    kind, text = ev.verdict_kern3(S.d, laws_rows[0].extra, bands[S.d]["bmin"], bands[S.d]["bmax"], bands[0]["bmin"], bands[0]["bmax"], R)
else:
    kind, text = ev.verdict_kern3(0, None, None, None, None, None, R)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Takt Verzug")
st.caption("Wurzeln des Polynoms P_d in der z-Ebene und die Antwort auf einen Stoß x₀ = (1, 0). Verzug d und R stehen in der Seitenleiste.")
tab_wagen, tab_skalar, tab_hand = st.tabs(["Wagen", "Skalar", "Von Hand (Brüche)"])
with tab_wagen:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Wurzeln der Schleife**")
        st.plotly_chart(build_roots(list(t.roots)), width="stretch", key="roots_chart")
        st.caption(f"{S.d + 2} Wurzeln von P_d; rot: Betrag ≥ 1. Die d Nullwurzeln der Zustandsmatrix kommen dazu und sind nicht gezeichnet. K = ({de(t.k1, 3)}; {de(t.k2, 3)}).")
    with c2:
        st.markdown("**Antwort auf den Stoß**")
        st.plotly_chart(build_decay(list(t.norms), t.rho), width="stretch", key="decay_chart")
        st.caption("‖x_k‖ über den ersten 80 Perioden (logarithmisch); ohne Verzug: ρ = " + fmt_rho(t.rho0) + ".")
    kind, text = ev.verdict_takt(t)
    getattr(st, kind)(text)
    st.caption(ev.m5_text())
with tab_skalar:
    st.markdown("Skalar x⁺ = x + u, q = r = 1: K = 1/φ = 0,618 (φ: Goldener Schnitt); mit Verzug hat die Schleife das Polynom z^(d+1) − z^d + K, **dasselbe wie das Lager ohne Leitung** mit L = d + 1 und α = K "
                "([`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo)); diese Demo wiederholt dessen Grenze nicht, sie nutzt sie als Gegenprobe.")
    srows = _scalar()
    sel = srows[min(S.d, len(srows) - 1)]
    kind, text = ev.verdict_scalar(sel)
    getattr(st, kind)(text)
    st.dataframe(
        [{"d": r["d"], "ρ": de(r["rho"], 4), "Lager-Grenze (L = d + 1)": de(r["lager_grenze"], 4), "K / Grenze": de(r["ratio"], 3),
          "K = 3/5 stabil": "ja" if r["stable_35"] else "nein", "K = 5/8 stabil": "ja" if r["stable_58"] else "nein"} for r in srows],
        hide_index=True, width="stretch",
    )
    st.caption("Spalten 5 und 6: exakte Schur–Cohn-Prüfung in Brüchen (K = 3/5 unterhalb, K = 5/8 oberhalb von 0,618). Bei d = 2 liegt K = 1/φ genau auf der Grenze; das ist eine Gleichheit zweier Zahlen, "
               "keine Erklärung. Bei d > 5 zeigt die Tabelle nichts Neues; der Skalar steht für d aus der Seitenleiste bis 5.")
with tab_hand:
    h = _hand(S.d)
    st.markdown(f"Gerundeter Entwurf K = (2/5, 6/5) (nahe R = 1), Verzug d = {S.d}, alles in Brüchen (`fractions`), ohne Gleitkomma:")
    st.markdown(f"- Polynom **P_{S.d}(z) = {poly_str(h['poly'])}**; Schur–Cohn: **{'stabil' if h['stable'] else 'instabil'}** (ρ = {de(h['rho'], 4)} numerisch).")
    if h["roots_sum"] is not None:
        st.markdown(f"- Summe der Wurzeln nach Vieta: {frac_str(h['roots_sum'])} (der Koeffizient von z^(d+1)); für jede statische Verstärkung auf dem verspäteten Zustand allein (ohne Vorhersage) folgt ρ ≥ 2/(d + 2).")
    xs_n = ", ".join(f"({frac_str(a)}; {frac_str(b)})" for a, b in h["x_nopred"][1:])
    xs_p = ", ".join(f"({frac_str(a)}; {frac_str(b)})" for a, b in h["x_pred"][1:])
    st.markdown(f"- Bahn x₁ … x₄ ohne Vorhersage (Stoß x₀ = (1; 0), blinde Perioden u = 0): {xs_n}")
    st.markdown(f"- Bahn x₁ … x₄ mit Vorhersage: {xs_p}")
    ohne_hinweis = "" if h["stable"] else "; diese Schleife ist ohne Vorhersage instabil und erreicht diese Lage nie, es ist nur der Fixpunkt der Gleichungen"
    st.markdown(f"- Konstante Störkraft w = 1: Gleichgewichtslage x₁* = **{frac_str(h['x1_pred'])}** mit Vorhersage, {frac_str(h['x1_nopred'])} ohne (exakte Lösung des Gleichungssystems{ohne_hinweis}).")
    st.caption("Handbeispiel: bei d = 1 ist P₁ = z³ − 2z² + (11/5) z − 4/5 nach Schur–Cohn instabil (ρ = 1,1725), mit Vorhersage x₂ = (1; −2/5), x₃ = (3/5; −8/25), und x₁* = 11/2. "
               "Der Entwurf K = (2/5, 6/5) ist gerundet und nur grob nahe R = 1: bei R = 1 selbst ist ρ = 1,193 (mit d = 1), nicht 1,1725. "
               "Die Herleitung der Gleichgewichtslage mit Vorhersage steht im Expander 📐 unter „Dauerfehler“.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Weich genug, aber langsam")
st.caption("Das kleinste R, ab dem der LQR-Entwurf mit d Perioden Verzug hält (Bisektion, Q = I), und die Abklingzeit des gerade stabilen Entwurfs (R = 1,2 · R_krit) gegen die der Vorhersage.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_krit(krit), width="stretch", key="krit_chart")
with c2:
    st.plotly_chart(build_decay_bars(krit), width="stretch", key="decay_bars_chart")
st.dataframe(
    [{"d": r["d"], "R_krit": fmt_R(r["rkrit"]), "R (1,2 · R_krit)": fmt_R(r["R_soft"]), "ρ bei 1,2 · R_krit": de(r["rho_soft"], 4),
      "Abklingzeit weich": fmt_int(r["decay_soft"]) if r["decay_soft"] is not None else "über Horizont",
      "Abklingzeit ohne Verzug (gleicher R)": r["decay_d0"], "Abklingzeit Vorhersage (R = 1)": r["decay_pred"]} for r in krit],
    hide_index=True, width="stretch",
)
kind, text = ev.verdict_krit(S.d, krit)
getattr(st, kind)(text)
st.caption("Abklingzeit: letzte Periode mit ‖x_k‖ über 1 % der Spitze, Stoß x₀ = (1, 0). Die Spalte „ohne Verzug (gleicher R)“ zeigt, was der große R allein kostet; den Rest macht der Verzug "
           "(der Entwurf R = 1 ohne Verzug braucht 5). Warum R_krit so wächst, ist nicht erklärt; gemessen ist nur, bei welchem R und d die Schleife hält.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Rechenzeit als Bruchteil der Periode")
st.caption("Hier gilt die exakte Abtast-Halte-Fassung des Wagens, **B = [½, 1]ᵀ** (nicht B = [0, 1]ᵀ der übrigen Experimente: dort hat ein Bruchteil keine Bedeutung). Der Regler misst zur Taktzeit, "
           "rechnet θ Perioden und stellt dann; K kommt aus der DARE dieses Modells.")
st.slider("Rechenzeit θ (Perioden)", *bounds("theta_slider"), step=C.THETA_STEP, key="theta_slider",
          help="θ = m + τ: m ganze Perioden und der Bruchteil τ. In der Periode wirkt erst die alte, dann die neue Stellgröße.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_theta(tinfo_curve, tinfo), width="stretch", key="theta_chart")
with c2:
    st.plotly_chart(build_theta_star(_theta_star_curve(), float(logR)), width="stretch", key="theta_star_chart")
kind, text = ev.verdict_theta(tinfo)
getattr(st, kind)(text)
if tinfo["angle"] is not None:
    st.caption(f"Bei θ* liegt die dominante Wurzel bei einem Winkel von {de(tinfo['angle'], 1)}° (Gitter 0,01 Perioden, dann Bisektion). Rechts: θ* gegen log₁₀ R; die gestrichelte Linie ist das gewählte R. "
               "Eine Faustregel aus dem Schrifttum (Verhältnis Verzug zu Abtastzeit deutlich unter 1) wird damit nicht geprüft, die Zahl liegt nur in derselben Richtung.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Mit Vorhersage")
st.caption("Vier Gesetze bei demselben Verzug d: die Vorhersage, derselbe Gewinn ohne Vorhersage, der weichste stabile Entwurf und die beste statische Verstärkung ohne Vorhersage (vorgerechnet).")
cc1, cc2 = st.columns(2)
with cc1:
    st.selectbox("Gesetz in der Abbildung", C.LAWS, key="law_select")
with cc2:
    st.radio("Anfangsabweichung x₀", C.X0_OPTIONS, key="x0_select", horizontal=True, help="(1, 0): Lage um 1 daneben, in Ruhe. (0, 1): Geschwindigkeit 1, die der Wagen in den blinden Perioden weiterfährt.")
S = Settings(logR=S.logR, d=S.d, theta=S.theta, law=st.session_state["law_select"], x0=st.session_state["x0_select"], b=S.b, dh_off=S.dh_off, w=S.w)
J0, rows = _laws(R, S.d, S.x0)
chosen = next(r for r in rows if r.key == S.law) if any(r.key == S.law for r in rows) else rows[0]
ideal, xs = ev.law_trajectory(R, S.d, chosen, S.x0)
st.plotly_chart(build_traj(ideal, xs, S.d, chosen.label), width="stretch", key="traj_chart")
note = f" {chosen.note}" if chosen.note else ""
st.caption(f"Schraffiert: die d blinden Perioden (keine Messung, u = 0). Gewähltes Gesetz: {chosen.label}.{note}")


def _fmt_opt(v, f):
    return "—" if v is None else f(v)


st.dataframe(
    [{"Gesetz": r.label, "ρ": _fmt_opt(r.rho, lambda v: de(v, 3)), "Abklingzeit": _fmt_opt(r.decay, lambda v: fmt_int(v)) if r.rho is None or r.rho < 1 - C.TOL else "instabil",
      "Kosten": _fmt_opt(r.cost, lambda v: de(v, 3)), "Mehrkosten": _fmt_opt(r.extra, lambda v: pct(v, 1, True))}
     for r in rows],
    hide_index=True, width="stretch",
)
st.caption(f"Kosten Σ x'Qx + R u² nach der Anfangsabweichung x₀ = {S.x0} bei R = {fmt_R(R)} (Auswertung = Entwurf), Mehrkosten gegen den Entwurf ohne Verzug ({de(J0, 3)}). "
           "Die Zeile „Beste statische Verstärkung“ ist vorgerechnet (Nelder–Mead, kein Beweis für das globale Minimum) und nur für R = 1 mit Kosten hinterlegt. "
           "Mit Vorhersage hat die Schleife (bei b = 1, dh = d) die beiden Wurzeln von A − BK und 3d Nullwurzeln. Bei d = 0 sind alle Gesetze gleich, die Auswahl „Gesetz“ hat dann keine Wirkung.")
kind, text = ev.verdict_vorhersage(S.d, S.x0, rows, R, _takt(R, 0).decay)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wenn das Modell nicht stimmt")
st.caption("Die Vorhersage rechnet mit dem Modell des Reglers: Beschleunigungsverstärkung 1 und angenommenem Verzug dh. Die Strecke hat die wahre Verstärkung b und den wahren Verzug d.")
cc1, cc2 = st.columns(2)
with cc1:
    st.slider("Wahre Beschleunigungsverstärkung b (angenommen 1)", *bounds("b_slider"), step=C.B_STEP, key="b_slider", help="x⁺ = A x + b B u; der Regler rechnet mit b = 1.")
with cc2:
    st.selectbox("Angenommener Verzug dh gegenüber dem wahren d", C.DH_OFFSETS, key="dh_select", format_func=lambda o: "dh = d" if o == 0 else f"dh = d {o:+d}".replace("-", "−"),
                 help="Der Regler nimmt dh = d + Versatz an (nicht unter 0).")
S = Settings(logR=S.logR, d=S.d, theta=S.theta, law=S.law, x0=S.x0, b=float(st.session_state["b_slider"]), dh_off=int(st.session_state["dh_select"]), w=S.w)
minfo = ev.modell_info(R, S.d, S.dh, S.b)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Stabiler Bereich von b mit Vorhersage (je d)**")
    st.plotly_chart(build_bands(bands, S.b), width="stretch", key="bands_chart")
    st.caption("Strichlinie: angenommenes b = 1; gepunktet: das gewählte b. Gitter 0,01 bis 6, Ränder per Bisektion; bei d = 0 reicht der stabile Bereich bis zum Gitterrand 0,01 (untere Grenze nicht erreicht).")
with c2:
    dm = _dh_map(R)
    st.markdown("**Falsch angenommener Verzug: ρ je Zelle**")
    st.plotly_chart(build_dh_map(dm), width="stretch", key="dh_chart")
    st.caption(f"Stabil in {dm['stable_off_diagonal']} von {dm['cells_off_diagonal']} Zellen mit dh ≠ d (R = {fmt_R(R)}, b = 1).")
kind, text = ev.verdict_modell(minfo)
getattr(st, kind)(text)
srow = _static_b()
st.caption("Zum Vergleich ohne Vorhersage, beste statische Verstärkung (vorgerechnet): stabil bis b = " + ", ".join(f"{de(r['bmax'], 2)} (d = {r['d']}, ρ = {de(r['rho'], 3)})" for r in srow)
           + " – größere Toleranz gegen b, aber langsamer.")

st.markdown("---")

# --- Experiment 6 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Störkraft und Rauschen")
st.caption("Konstante Störkraft w auf die Beschleunigung (Dauerfehler) und Rauschen N(0, 1) je Periode (Streuung). Die Vorhersage kennt keine Störung; ein Störungsmodell mit Beobachter steht in "
           "[`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo).")
st.slider("Konstante Störkraft w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", help="Wirkt wie eine zusätzliche Beschleunigung w in jeder Periode.")
S = Settings(logR=S.logR, d=S.d, theta=S.theta, law=S.law, x0=S.x0, b=S.b, dh_off=S.dh_off, w=float(st.session_state["w_slider"]))
sinfo = _stoer(R, S.d, S.w)
st.plotly_chart(build_equilibrium(sinfo), width="stretch", key="eq_chart")
st.caption("Gleichgewichtslage x₁* (der Wagen bleibt dort mit u = −w stehen). Mit Vorhersage x₁* = (w/k₁)(1 + k₂ d + k₁ d(d − 1)/2), ohne Vorhersage w/k₁; weicher Entwurf: R = 1,2 · R_krit; "
           "beste statische Verstärkung: vorgerechnet.")
kind, text = ev.verdict_stoer(sinfo)
getattr(st, kind)(text)
tab_lyap, tab_mc = st.tabs(["Streuung (Lyapunov)", "Monte-Carlo"])
with tab_lyap:
    m1, m2, m3 = st.columns(3)
    m1.metric("Ohne Verzug (d = 0)", de(sinfo["stat_ideal"], 2), delta="Kosten je Periode", delta_color="off")
    m2.metric(f"Mit Vorhersage (d = {S.d})", de(sinfo["stat_pred"], 2), delta=f"{de(sinfo['stat_pred'] - sinfo['stat_ideal'], 2, True)} gegen d = 0", delta_color="inverse")
    m3.metric("Beste lineare Verstärkung ohne Vorhersage", de(sinfo["stat_static"], 2) if sinfo["stat_static"] is not None else "—",
              delta="vorgerechnet, R = 1 und d ≤ 4" if sinfo["stat_static"] is not None else "nur für R = 1 und d ≤ 4 vorgerechnet", delta_color="off")
    st.caption("Kosten je Periode E[x'Qx + R u²] unter Störkraft N(0, 1), stationär aus der Lyapunov-Gleichung; die Vorhersage ist lokal kostenminimal unter linearen Gesetzen der verfügbaren Information "
               "(gemessen: keine zufällige Störung des Gewinns senkt die Kosten).")
with tab_mc:
    st.markdown("Gegenprobe mit einem Zufallslauf: Störkraft aus dem ganzzahligen SplitMix64 (Zufalls-Seed), Mittel über die Perioden, die ersten 1 000 ausgenommen.")
    cc1, cc2 = st.columns(2)
    with cc1:
        seed = st.number_input("Zufalls-Seed des Laufs", *bounds("mc_seed_input"), key="mc_seed_input", step=1)
    with cc2:
        st.selectbox("Perioden", C.MC_LENGTHS, key="mc_len_select", format_func=lambda v: fmt_int(v))
    if st.button("▶ Monte-Carlo-Lauf starten", key="mc_button"):
        st.session_state["mc_result"] = _mc(R, S.d, int(st.session_state["mc_len_select"]), int(seed))
    res = st.session_state.get("mc_result")
    if res is not None:
        kind, text = ev.verdict_mc(res)
        getattr(st, kind)(text)
    else:
        st.info("Noch kein Lauf. Der Lauf dauert bei 20 000 Perioden etwa 1 bis 2 Sekunden.")

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zustand wird exakt gemessen, nur um d Perioden verspätet** | Mit Messrauschen und Vorhersage nicht gemessen; ein Beobachter für den verspäteten Zustand ist nicht gebaut. | [`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo) (ohne Verzug) |
| **Der Verzug ist ganzzahlig, fest und bekannt** | Mit falsch angenommenem Verzug (um höchstens 2 Perioden daneben) und R ≤ 1 ist die Vorhersage in 2 000 von 2 000 Zufallszellen instabil (Experiment 5), getestet für d = 1 bis 4 und bei großem R nicht; Schwankung der Rechenzeit (Jitter, Überspringen eines Taktes) ist nicht gemessen. | – |
| **Das Streckenmodell stimmt (b = 1)** | Die Toleranz gegen b schrumpft mit d von beiden Seiten (R = 1: von 0,01 bis 1,94 bei d = 0 auf 0,77 bis 1,25 bei d = 5). | – |
| **Keine Stellgrenzen** | Ein MPC mit Beschränkungen und Rechenzeit ist nicht gemessen. | [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo) |
| **Keine Störung** | Bei konstanter Störkraft wächst der Dauerfehler mit der Vorhersage (Faktor 2,24 bei d = 1, R = 1, gegenüber d = 0). | [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) |
| **Q = I** | R_krit hängt an Q; die Zahlen dieser Demo gelten für Q = I (bei zufälligem Q hält d = 1 nur in einem Teil der Entwürfe, Experiment 1). | – |
| **Die Rechenzeit ist vorgegeben** | Wie viele Perioden eine Optimierung mit N Schritten braucht, ist nicht gemessen; hier ist d ein Regler. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/regelung-verzug-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Wagen mit Zustand $x = (x_1, x_2)$ (Lage, Geschwindigkeit), Kosten, die Abweichung und Stellaufwand abwägen, und ein Verzug von $d$ Perioden zwischen Messung und Stellgröße. Gesucht ist, ob das Gesetz $u = -Kx$ bei $d$ Perioden Rechenzeit die Schleife stabil hält und was die Vorhersage ändert. Die Formeln sind eigene Herleitung in diesem Modell; das Modell ist vollständig mit Strecke, Entwurf, Verzug, Vorhersage, Kosten und Dauerfehler beschrieben (Stellgrenzen und Messrauschen gehören nicht dazu).

| Symbol | Bedeutung |
|---|---|
| $x = (x_1, x_2)$ | Zustand des Wagens: Lage und Geschwindigkeit |
| $u_k$ | Stellgröße (Schub) in Periode $k$ |
| $A, B$ | Strecke $x_{k+1} = A x_k + B(b\,u_k + w)$ mit $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$, $B = (0, 1)^\top$ bzw. $B = (\tfrac12, 1)^\top$ (Rechenzeit als Bruchteil der Periode) |
| $Q, R$ | Kostengewichte, $Q = I$, $R > 0$ |
| $K = (k_1, k_2)$ | Gewinn des Entwurfs ohne Verzug aus der DARE |
| $d, \theta = m + \tau$ | Verzug in ganzen Perioden bzw. Rechenzeit mit Bruchteil $\tau$ |
| $b, w$ | wahre Beschleunigungsverstärkung (Regler nimmt 1 an) bzw. konstante Störkraft |
| $\hat d$ | vom Regler angenommener Verzug |
| $P_d(z)$, $\rho$ | charakteristisches Polynom der Schleife und sein Spektralradius (größter Wurzelbetrag); die Schleife hält, wenn $\rho$ unter 1 liegt |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/regelung-verzug-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Entwurf ohne Verzug.** Die DARE und der Gewinn:

$$P = Q + A^\top P A - A^\top P B\,(R + B^\top P B)^{-1} B^\top P A$$

$$K = (R + B^\top P B)^{-1} B^\top P A$$

**Verzug.** Das Gesetz mit $d$ Perioden Rechenzeit (blind bis $k = d - 1$):

$$u_k = -K\,x_{k-d}\quad (k \ge d),\qquad u_k = 0\quad (k < d)$$

Mit $B = (0, 1)^\top$ folgt für die Lage $x_{k+2} - 2x_{k+1} + x_k = u_k$ und damit das charakteristische Polynom (Grad $d + 2$, dazu $d$ Nullwurzeln der Zustandsmatrix):

$$P_d(z) = z^d (z-1)^2 + k_2 (z - 1) + k_1$$

Für $d \ge 1$ ist die Summe der Wurzeln $2$ (Koeffizient von $z^{d+1}$), also gilt für jede statische Verstärkung $K'$ auf dem verspäteten Zustand allein (ohne Vorhersage) $\rho \ge 2/(d+2)$; Deadbeat ($\rho = 0$) ist damit nicht erreichbar. Mit der Vorhersage ist das anders: Für $K = (1, 2)$ wird das Polynom der Schleife unten $z^{3d+2}$, also $\rho = 0$ (numerisch nur bis auf die Rechengenauigkeit).

**Vorhersage.** Mit angenommenem Verzug $\hat d$ rechnet der Regler den Zustand voraus:

$$\hat x_k = A^{\hat d} x_{k-d} + \sum_{j=0}^{\hat d - 1} A^{\hat d - 1 - j} B\, u_{k-\hat d + j}, \qquad u_k = -K \hat x_k$$

Für $\hat d = d$, $b = 1$, $w = 0$ gilt $x_{d+m} = (A - BK)^m A^d x_0$; das charakteristische Polynom der Schleife (Dimension $3d + 2$) ist $z^{3d}\,(z^2 + (k_2 - 2) z + 1 - k_2 + k_1)$.

**Kosten.** Mit Vorhersage und blinden Perioden nach der Anfangsabweichung $x_0$:

$$J = \sum_{k=0}^{d-1} x_k^\top Q x_k + x_d^\top P x_d, \qquad x_k = A^k x_0 \ (k \le d)$$

Für $x_0 = (1, 0)$ und $Q = I$ ist $J = P_{11} + d$, denn der Wagen bleibt in den blinden Perioden stehen.

**Dauerfehler.** Bei konstanter Störkraft $w$ liegt die Gleichgewichtslage mit Vorhersage bei

$$x_1^* = \frac{w}{k_1}\left(1 + k_2\, d + k_1\,\frac{d(d-1)}{2}\right),$$

ohne Vorhersage bei $x_1^* = w/k_1$.

*Herleitung mit Vorhersage* (eigene Rechnung): Im Gleichgewicht ist $x_2 = 0$ und $u^* = -w$, damit $b\,u^* + w = 0$ mit $b = 1$. Die Vorhersage liefert dann $\hat x_1 = x_1 + u^*\,d(d-1)/2$ und $\hat x_2 = d\,u^*$. Aus $u^* = -(k_1 \hat x_1 + k_2 \hat x_2)$ folgt $k_1 x_1 = w\,(1 + k_2 d + k_1 d(d-1)/2)$, also die Formel oben. Ohne Vorhersage ist $\hat x = x$ und $x_1^* = w/k_1$. Die Formel sagt nur, wo der Fixpunkt der Gleichungen liegt; erreicht wird er nur, wenn die Schleife stabil ist (ohne Vorhersage ist das nicht immer so, siehe das Handbeispiel).

**Abklingzeit und Streuung.** Die Abklingzeit für den Stoß $x_0 = (1, 0)$ ist die letzte Periode, in der die Norm (Lage und Geschwindigkeit zusammen) über 1 % ihrer Spitze liegt:

$$T_{1\%} = \max\{\,k : \lVert x_k \rVert_2 > 0{,}01\,\max_j \lVert x_j \rVert_2\,\}$$

Die Streuung unter einer Störkraft $N(0, 1)$ je Periode sind die stationären Kosten je Periode. Mit dem Zustand $z$ der Schleife (die letzten $d+1$ Zustände $x$ und die Stellgrößen-Vergangenheit), $z_{k+1} = M z_k + g\,\xi_k$, $\xi_k \sim N(0, 1)$, $g$ wirkt auf die Geschwindigkeit, und $u_k = -r^\top z_k$ gilt:

$$S = M S M^\top + g g^\top, \qquad \text{Streuung} = \operatorname{tr}(Q\,S_{xx}) + R\; r^\top S r$$

Dabei ist $S_{xx}$ der Block von $S$ für den aktuellen Zustand $x_k$ (Lage und Geschwindigkeit).

**Rechenzeit als Bruchteil der Periode** (Modell $B = (\tfrac12, 1)^\top$, $\theta = m + \tau$): in einer Periode wirkt zuerst $u_{k-m-1}$, nach dem Bruchteil $\tau$ dann $u_{k-m}$:

$$x_{k+1} = A x_k + \Gamma_1(\tau)\,u_{k-m-1} + \Gamma_0(\tau)\,u_{k-m}$$

$$\Gamma_0 = \begin{pmatrix}(1-\tau)^2/2\\ 1-\tau\end{pmatrix},\qquad \Gamma_1 = \begin{pmatrix}\left(1-(1-\tau)^2\right)/2\\ \tau\end{pmatrix}$$

**Literatur (nur so, wie gelesen).** Findeisen und Allgöwer (ADCHEM 2003, Wortlaut gelesen): Die Rechenzeit werde in Stabilitätsbeweisen oft vernachlässigt, führe aber zu einem Verzug zwischen Zustandsinformation und Stellsignal, der
die Regelgüte stark mindern oder die Schleife instabil machen könne (nichtlineares MPC; hier wird davon nichts nachgerechnet). Wolf, Würth und Marquardt (CDC 2011, Wortlaut gelesen): ein ignorierter Rechenverzug gefährdet die Stabilität,
als Faustregel solle das Verhältnis von Verzug zu Abtastzeit deutlich unter 1 liegen. Orzechowska (Dissertation Coventry 2014, nur das Abstract gelesen) löst eine Lieferzeit im Lager mit einem Smith-Prädiktor; Åström und Murray, Übung 12.8
(nur der Aufgabentext gelesen). Diese Demo sagt „Vorhersage“, nicht „Smith-Prädiktor“, und behauptet keine Gleichheit mit dem dortigen Regler. Rawlings, Mayne und Diehl, Abschnitt 2.5.1 (S. 132, in eigenen Worten; Wortlaut nach einem
früheren Quellenbericht gelesen, die Stelle wurde beim Bau nicht erneut geöffnet): beim unbeschränkten
Problem mit Terminalkosten aus der DARE hängt der Gewinn der ersten Stufe nicht vom Horizont ab; hier gemessen für N = 1 bis 12 (README, Zeile „MPC-Brücke“).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html)."
)
