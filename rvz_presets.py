"""Permalink (Begrenzen, Einrasten) und Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben).

Zwei Regler stehen in der Seitenleiste (log₁₀R, d), die übrigen bei ihrem Experiment im Hauptbereich; alle sind in SETTING_SPECS eingetragen, damit jede Änderung
die Hervorhebung des Presets aufhebt. Ein Preset legt nur fest, was von der Voreinstellung abweicht; alles Übrige gilt als Standardwert und wird beim Anklicken gesetzt."""

import math
from dataclasses import dataclass

import streamlit as st

import rvz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Auswahlregler: erlaubte Werte (Einrasten); None: Schieberegler mit lo … hi


SETTING_SPECS = {
    "logR_slider": SettingSpec("logR", float, C.DEFAULT_LOGR, C.LOGR_MIN, C.LOGR_MAX),
    "d_slider": SettingSpec("d", int, C.DEFAULT_D, C.D_MIN, C.D_MAX),
    "theta_slider": SettingSpec("theta", float, C.DEFAULT_THETA, C.THETA_MIN, C.THETA_MAX),
    "law_select": SettingSpec("gesetz", int, C.DEFAULT_LAW, options=C.LAWS),
    "x0_select": SettingSpec("x0", int, C.DEFAULT_X0, options=C.X0_OPTIONS),
    "b_slider": SettingSpec("b", float, C.DEFAULT_B, C.B_MIN, C.B_MAX),
    "dh_select": SettingSpec("dh", int, C.DEFAULT_DH_OFF, options=C.DH_OFFSETS),
    "w_slider": SettingSpec("w", float, C.DEFAULT_W, C.W_MIN, C.W_MAX),
    "mc_seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "mc_len_select": SettingSpec("perioden", int, C.DEFAULT_MC_LEN, options=C.MC_LENGTHS),
}
PRESET_KEYS = {"logR": "logR_slider", "d": "d_slider", "theta": "theta_slider", "law": "law_select", "x0": "x0_select", "b": "b_slider",
               "dh_off": "dh_select", "w": "w_slider", "mc_seed": "mc_seed_input", "mc_len": "mc_len_select"}
SIDEBAR_KEYS = ("logR_slider", "d_slider")


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _from_url(state_key, raw):
    """Wert aus der Adresszeile: Zahl begrenzt, Auswahl eingerastet (Gesetz und x₀ stehen als Index in der Adresse); None bei Unbrauchbarem."""
    spec = SETTING_SPECS[state_key]
    if spec.options is not None:
        if state_key in ("law_select", "x0_select"):
            i = int(raw)
            return spec.options[max(0, min(len(spec.options) - 1, i))]
        v = float(raw)
        return min(spec.options, key=lambda o: abs(o - v))
    value = spec.caster(raw)
    if not math.isfinite(value):
        return None
    value = max(spec.lo, min(spec.hi, value))
    if state_key == "logR_slider":
        value = round(round(value / C.LOGR_STEP) * C.LOGR_STEP, 4)
    elif state_key == "theta_slider":
        value = round(round(value / C.THETA_STEP) * C.THETA_STEP, 4)
    elif state_key == "b_slider":
        value = round(value, 2)
    elif state_key == "w_slider":
        value = round(value, 1)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf die erlaubten Stufen eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _from_url(state_key, qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration (Gesetz und x₀ als Index)."""
    try:
        for state_key, value in values.items():
            spec = SETTING_SPECS[state_key]
            if state_key in ("law_select", "x0_select"):
                value = spec.options.index(value)
            st.query_params[spec.url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt jeden Regler: den Wert des Presets, sonst den Standardwert (ein Preset legt nur Abweichungen fest)."""
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = preset.get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )

