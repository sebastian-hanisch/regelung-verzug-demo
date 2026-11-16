"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätzliche Tests (unter der Vorlage) ---------------------------------------------------------------------------------------------------

def test_monte_carlo_button_runs_and_shows_the_verdict():
    """Der Monte-Carlo-Knopf hat das Zeichen ▶ und wird vom Smoke-Test oben ausgelassen: hier eigens geklickt."""
    at = _fresh_app()
    assert any("noch kein lauf" in i.value.lower() for i in at.info)
    at.button(key="mc_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Monte-Carlo mit Zufalls-Seed 1814" in (s.value if hasattr(s, "value") else "") for s in list(at.success) + list(at.warning))


def test_every_verzug_value_and_every_select_option_renders():
    for d in range(0, 9):
        at = _fresh_app()
        at.sidebar.slider(key="d_slider").set_value(d).run(timeout=TIMEOUT)
        assert not at.exception, (d, [str(e) for e in at.exception])
    at = _fresh_app()
    for key in ("law_select", "x0_select", "dh_select", "mc_len_select"):
        art = "radio" if key == "x0_select" else "selectbox"
        for option in list(getattr(at, art)(key=key).options):
            getattr(at, art)(key=key).set_value(option)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_permalink_sets_the_controls_and_bad_values_are_ignored_or_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["d"] = "3"
    at.query_params["logR"] = "1.0"
    at.query_params["gesetz"] = "1"
    at.query_params["b"] = "abc"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="d_slider").value == 3 and at.sidebar.slider(key="logR_slider").value == 1.0
    assert at.selectbox(key="law_select").value == "Ohne Vorhersage (gleicher Gewinn K)"
    assert at.slider(key="b_slider").value == 1.0                                   # unbrauchbarer Wert: Standard bleibt
