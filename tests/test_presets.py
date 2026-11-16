"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, eindeutige Hervorhebung, Permalink-Begrenzung."""

import rvz_constants as C
import rvz_evaluation as ev
import rvz_presets as P


def _full(preset):
    """Alle Regler eines Presets; was fehlt, ist der Standardwert."""
    return {key: preset.get(key, P.SETTING_SPECS[state_key].default) for key, state_key in P.PRESET_KEYS.items()}


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, value in _full(preset).items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(**{k: v for k, v in _full(first).items() if k in ev.Settings.__dataclass_fields__}) == ev.Settings()


def test_all_presets_are_distinct_so_the_highlight_is_unambiguous():
    fulls = [tuple(sorted(_full(p).items())) for p in C.PRESETS.values()]
    assert len(set(fulls)) == len(fulls)


def test_setting_specs_cover_every_preset_key_and_defaults_are_inside_their_bounds():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        if spec.options is not None:
            assert spec.default in spec.options
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_log_r_grid_hits_round_values_exactly():
    """Die Schieberegler-Stufen 0,25 geben für R = 1, 10 genau 1,0 und 10,0 (kein 9,999…, das Presets und Verdikte verschöbe)."""
    assert ev.R_of(0.0) == 1.0 and ev.R_of(1.0) == 10.0 and ev.R_of(2.0) == 100.0
    assert P._from_url("logR_slider", "0.9") == 1.0 and P._from_url("logR_slider", "99") == C.LOGR_MAX and P._from_url("d_slider", "-4") == C.D_MIN
    assert P._from_url("law_select", "2") == C.LAWS[2] and P._from_url("law_select", "9") == C.LAWS[-1]
    assert P._from_url("dh_select", "1.4") == 1 and P._from_url("mc_len_select", "21000") == 20000


def test_slider_defaults_are_multiples_of_their_steps():
    assert abs(C.DEFAULT_LOGR / C.LOGR_STEP - round(C.DEFAULT_LOGR / C.LOGR_STEP)) < 1e-12
    for p in C.PRESETS.values():
        for key, step in (("logR", C.LOGR_STEP), ("theta", C.THETA_STEP), ("b", C.B_STEP), ("w", C.W_STEP)):
            if key in p:
                assert abs(p[key] / step - round(p[key] / step)) < 1e-9, key
