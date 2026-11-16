"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Alle Regler stehen mit ihren Grenzen hier; die Seitenleiste (R, d) und die Experimente (θ, Gesetz, x₀, b, dh, w, Monte-Carlo) lesen sie über rvz_presets.bounds().
Die App rechnet deterministisch; Zufall gibt es nur im Monte-Carlo-Tab (Zufalls-Seed, ganzzahliger SplitMix64) und in tools/make_frozen.py."""

LOGR_MIN, LOGR_MAX, LOGR_STEP, DEFAULT_LOGR = -3.0, 4.0, 0.25, 0.0     # R = 10^log₁₀R
D_MIN, D_MAX, DEFAULT_D = 0, 8, 1
THETA_MIN, THETA_MAX, THETA_STEP, DEFAULT_THETA = 0.0, 3.0, 0.05, 0.25
B_MIN, B_MAX, B_STEP, DEFAULT_B = 0.2, 2.5, 0.01, 1.0
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.0, 3.0, 0.1, 1.0
SEED_MAX, DEFAULT_SEED = 999999, 1814

LAWS = (
    "Mit Vorhersage",
    "Ohne Vorhersage (gleicher Gewinn K)",
    "Weicher Entwurf (R = 1,2 · R_krit)",
    "Beste statische Verstärkung",
)
DEFAULT_LAW = LAWS[0]
X0_OPTIONS = ("(1, 0)", "(0, 1)")
X0_VALUES = {"(1, 0)": (1.0, 0.0), "(0, 1)": (0.0, 1.0)}
DEFAULT_X0 = X0_OPTIONS[0]
DH_OFFSETS = (-2, -1, 0, 1, 2)
DEFAULT_DH_OFF = 0
MC_LENGTHS = (10000, 20000, 50000)
DEFAULT_MC_LEN = 20000

SOFT_FACTOR = 1.2            # weicher Entwurf: R = 1,2 · R_krit(d)
EVAL_R = 1.0                 # Auswertungsgewicht der vorgerechneten besten statischen Verstärkungen (tools/make_frozen.py)
MC_BAND = 0.03               # Band des Monte-Carlo-Laufs gegen die Lyapunov-Rechnung (relativ)
DECAY_FRAC = 0.01            # Abklingzeit: letzte Periode mit ‖x_k‖ über 1 % der Spitze
DECAY_HORIZON = 30000        # Perioden, nach denen die Suche nach der Abklingzeit endet
PLOT_HORIZON = 80            # Perioden der Bahn-Abbildungen
TOL = 1e-9                   # Grenzfälle auf dem Einheitskreis gelten nicht als stabil/instabil, sondern als „Grenze“

COLOR_MAIN = "#2F6B65"
COLOR_BAD = "#c44e52"
COLOR_REF = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.55)"
COLOR_WARN = "#dd8452"

PRESETS = {
    "Eine Periode Verzug (R = 1)": {},   # Voreinstellung: log₁₀R = 0, d = 1, alle übrigen Regler auf Standard
    "Skalar auf der Grenze (d = 2)": {"d": 2},
    "Weich genug (R = 10)": {"logR": 1.0, "d": 1},
    "Eine halbe Periode Rechenzeit": {"theta": 0.5},
    "Mit Vorhersage (d = 3)": {"d": 3},
    "Falsche Masse (b = 0,5)": {"d": 2, "b": 0.5},
    "Störkraft (w = 2)": {"w": 2.0},
}
PRESET_HELP = {
    "Eine Periode Verzug (R = 1)": "Wagen, Q = I, R = 1, ein Takt Rechenzeit: der Entwurf ohne Verzug (ρ = 0,422) wird mit d = 1 instabil (ρ = 1,193).",
    "Skalar auf der Grenze (d = 2)": "d = 2: im Skalarfall x⁺ = x + u liegt der Entwurf K = 1/φ genau auf der Stabilitätsgrenze (Tab Skalar); der Wagen mit R = 1 ist bei d = 2 instabil.",
    "Weich genug (R = 10)": "Ein weicherer Entwurf (R = 10) hält einen Takt Verzug (ρ = 0,961), klingt aber nur langsam ab.",
    "Eine halbe Periode Rechenzeit": "Rechenzeit θ = 0,5 Perioden bei R = 1: der Entwurf hält (ρ = 0,918); bei R = 0,1 hielte er nicht (Experiment 3).",
    "Mit Vorhersage (d = 3)": "Drei Takte Verzug, der Zustand wird vorhergerechnet: die Schleife hat wieder die Wurzeln des Entwurfs ohne Verzug, nur d blinde Perioden bleiben.",
    "Falsche Masse (b = 0,5)": "Die wahre Beschleunigungsverstärkung ist b = 0,5, der Regler nimmt 1 an: mit Vorhersage und d = 2 instabil, ohne Verzug stabil.",
    "Störkraft (w = 2)": "Konstante Störkraft w = 2: die Gleichgewichtslage mit Vorhersage liegt weiter vom Soll als ohne Verzug (Dauerfehler, Experiment 6).",
}
