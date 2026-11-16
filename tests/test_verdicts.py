"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds. Der Test prüft die Grenzen (ρ < 1, Band 10⁻⁹ um 1, 3 %-Band, w = 0, d = 0) und jede genannte Zahl direkt."""

import numpy as np

import rvz_constants as C
import rvz_evaluation as ev

NB = " "


def _takt(R=1.0, d=1, rho=1.193, rho0=0.422, decay=None, rkrit=6.854):
    return ev.Takt(R, d, 0.42, 1.24, rho0, rho, tuple([rho0, rho] + [1.3] * 7), rkrit, decay, 1.085, (1.0, 1.0), ())


def test_formatting_helpers():
    assert ev.de(1.5) == "1,50" and ev.de(-0.5, 1) == "−0,5" and ev.de(2.0, 1, True) == "+2,0"
    assert ev.fmt_int(1722) == f"1{NB}722" and ev.fmt_int(12) == "12"
    assert ev.pct(0.339, 1, True) == f"+33,9{NB}%"
    assert [ev.fmt_R(x) for x in (0.001, 0.1, 1.0, 6.854, 8.2249, 10.0, 82.79, 369.87, 1092.09)] == ["0,001", "0,1", "1", "6,85", "8,22", "10", "82,8", "370", f"1{NB}092"]
    assert ev.R_of(0.0) == 1.0 and ev.R_of(1.0) == 10.0 and ev.R_of(-1.0) == 0.1


def test_settings_dh_is_clipped_at_zero():
    assert ev.Settings(d=1, dh_off=-2).dh == 0 and ev.Settings(d=3, dh_off=2).dh == 5 and ev.Settings(d=2).dh == 2


def test_verdict_takt_unstable_names_rho_and_critical_r():
    kind, text = ev.verdict_takt(_takt())
    assert kind == "warning"
    assert "instabil: ρ = 1,193 statt 0,422" in text and "Erst ab R = 6,85 hält d = 1" in text and "R = 1 " in text


def test_verdict_takt_stable_names_decay_and_critical_r():
    kind, text = ev.verdict_takt(_takt(R=10.0, rho=0.961, rho0=0.65, decay=213, rkrit=6.854))
    assert kind == "success"
    assert "hält der Entwurf R = 10: ρ = 0,961 statt 0,650" in text and "Abklingzeit 213 Perioden" in text and "ab R_krit(d) = 6,85" in text


def test_verdict_takt_threshold_at_one_and_the_boundary_band():
    assert ev.verdict_takt(_takt(rho=1.0 - 1e-6))[0] == "success"                  # knapp unter 1, außerhalb des Bandes 10⁻⁹
    assert ev.verdict_takt(_takt(rho=1.0 + 1e-6))[0] == "warning"
    kind, text = ev.verdict_takt(_takt(rho=1.0 + 5e-10))
    assert kind == "info" and "auf der Stabilitätsgrenze" in text                  # Band 10⁻⁹


def test_verdict_takt_without_delay_and_unknown_decay():
    kind, text = ev.verdict_takt(_takt(d=0, rho=0.422, decay=5, rkrit=None))
    assert kind == "success" and "Ohne Verzug hat der Entwurf R = 1 den Spektralradius ρ = 0,422" in text and "in 5 Perioden ab" in text
    kind, text = ev.verdict_takt(_takt(rho=0.99, decay=None))
    assert kind == "success" and "länger als der Horizont" in text


def test_verdict_scalar_three_branches():
    on = {"d": 2, "rho": 1.0 + 1e-16, "lager_grenze": 0.618, "ratio": 1.0, "on_boundary": True}
    kind, text = ev.verdict_scalar(on)
    assert kind == "info" and "genau auf der Stabilitätsgrenze" in text and "L = 3" in text and "keine Erklärung" in text
    kind, text = ev.verdict_scalar({"d": 1, "rho": 0.786, "lager_grenze": 1.0, "ratio": 0.618, "on_boundary": False})
    assert kind == "success" and "ρ = 0,786" in text and "Grenze für L = 2 ist 1,000" in text and "darunter" in text
    kind, text = ev.verdict_scalar({"d": 3, "rho": 1.07, "lager_grenze": 0.445, "ratio": 1.39, "on_boundary": False})
    assert kind == "warning" and "ρ = 1,070" in text and "L = 4 ist 0,445" in text and "darüber" in text


def test_verdict_krit_names_critical_r_and_both_decay_times():
    rows = [{"d": 1, "rkrit": 6.854, "R_soft": 8.225, "decay_soft": 213, "decay_pred": 6, "rho_soft": 0.99, "rho_pred": 0.42},
            {"d": 2, "rkrit": 82.79, "R_soft": 99.3, "decay_soft": None, "decay_pred": 7, "rho_soft": 0.999, "rho_pred": 0.42}]
    kind, text = ev.verdict_krit(1, rows)
    assert kind == "info" and "R_krit = 6,85" in text and "(R = 8,22) braucht 213 Perioden" in text and "mit Vorhersage (K aus R = 1) 6 Perioden" in text
    assert "klingt innerhalb des Horizonts nicht ab" in ev.verdict_krit(2, rows)[1]
    assert f"alle 4{NB}001 gemessenen Entwürfe (R = 10⁻³ bis 10⁶, Q = I) sind stabil" in ev.verdict_krit(0, rows)[1]


def test_verdict_theta_branches():
    kind, text = ev.verdict_theta({"R": 1.0, "theta": 0.5, "rho": 0.918, "theta_star": 0.621, "angle": 59.2})
    assert kind == "success" and "θ = 0,50 Perioden Rechenzeit hält der Entwurf R = 1 (ρ = 0,918)" in text and "θ* = 0,62" in text
    kind, text = ev.verdict_theta({"R": 0.1, "theta": 0.5, "rho": 1.031, "theta_star": 0.464, "angle": 69.3})
    assert kind == "warning" and "hält der Entwurf R = 0,1 nicht (ρ = 1,031)" in text and "θ* = 0,46" in text
    kind, text = ev.verdict_theta({"R": 100.0, "theta": 3.0, "rho": 0.9, "theta_star": None, "angle": None})
    assert kind == "success" and "bis θ = 3 Perioden Rechenzeit stabil" in text


def _row(key, extra, rho, decay, k=(0.4, 1.2), dh=0, cost=1.0):
    return ev.LawRow(key, key, k[0], k[1], dh, rho, decay, cost, extra)


def test_verdict_vorhersage_with_unstable_and_stable_plain_gain_and_static_row():
    rows = [_row(C.LAWS[0], 0.339, 0.422, 6, dh=1), _row(C.LAWS[1], None, 1.193, None), _row(C.LAWS[2], 2.9, 0.99, 213), _row(C.LAWS[3], 0.576, 0.667, 21)]
    kind, text = ev.verdict_vorhersage(1, "(1, 0)", rows, 1.0, 5)
    assert kind == "info"
    assert "d = 1, x₀ = (1, 0): Mit Vorhersage kostet der Verzug +33,9" in text and "Abklingzeit ist 6 Perioden (ohne Verzug 5 bei R = 1)" in text
    assert "(ohne Verzug" not in ev.verdict_vorhersage(1, "(1, 0)", rows, 1.0)[1]
    assert "ohne Vorhersage ist instabil (ρ = 1,193)" in text
    assert "braucht 21 Perioden und kostet +57,6" in text and "ρ ≥ 2/(d + 2) = 0,667" in text
    rows[1] = _row(C.LAWS[1], 0.8, 0.9, 40)
    assert "hält (ρ = 0,900), kostet aber +80,0" in ev.verdict_vorhersage(1, "(1, 0)", rows, 1.0)[1]
    assert "alle vier Gesetze dasselbe" in ev.verdict_vorhersage(0, "(1, 0)", rows[:2], 1.0)[1]
    rows3 = rows[:3]
    assert "beste statische" not in ev.verdict_vorhersage(2, "(0, 1)", rows3, 2.0)[1]


def test_verdict_modell_wrong_delay_and_wrong_gain():
    kind, text = ev.verdict_modell({"R": 1.0, "d": 1, "dh": 0, "b": 1.0, "rho": 1.193, "bmin": 0.335, "bmax": 1.531})
    assert kind == "warning" and "nimmt dh = 0 an, wahr ist d = 1" in text and "ist instabil" in text
    kind, text = ev.verdict_modell({"R": 1.0, "d": 1, "dh": 1, "b": 1.0, "rho": 0.422, "bmin": 0.335, "bmax": 1.531})
    assert kind == "success" and "ρ = 0,422, die Schleife hält" in text and "von 0,34 bis 1,53" in text
    kind, text = ev.verdict_modell({"R": 1.0, "d": 2, "dh": 2, "b": 0.5, "rho": 1.05, "bmin": 0.536, "bmax": 1.383})
    assert kind == "warning" and "wahrem b = 0,50" in text and "von 0,54 bis 1,38" in text
    assert ev.verdict_modell({"R": 1.0, "d": 1, "dh": 3, "b": 1.0, "rho": 0.9, "bmin": None, "bmax": None})[0] == "success"


def test_verdict_stoer_branches():
    s = {"R": 1.0, "d": 1, "w": 1.0, "x1_ideal": 2.369, "x1_pred": 5.316, "factor": 2.244, "x1_static": 27.0, "x1_soft": 4.53, "stat_ideal": 4.613, "stat_pred": 13.299, "stat_static": 21.6}
    kind, text = ev.verdict_stoer(s)
    assert kind == "info"
    assert "Gleichgewichtslage 5,32 mit Vorhersage gegen 2,37 ohne Verzug (Faktor 2,24)" in text and "27,0 bei der besten statischen" in text and "13,30 mit Vorhersage gegen 4,61" in text
    s2 = dict(s, x1_static=None)
    assert "beste statische" not in ev.verdict_stoer(s2)[1]
    assert "keinen Dauerfehler" in ev.verdict_stoer(dict(s, w=0.0))[1]
    assert "Ohne Verzug (d = 0)" in ev.verdict_stoer(dict(s, d=0))[1]


def test_verdict_mc_band_is_inclusive_at_3_percent():
    m = {"mc": 29.0, "lyap": 28.8784, "dev": 0.03, "periods": 20000, "seed": 1814, "d": 2, "R": 1.0}
    kind, text = ev.verdict_mc(m)
    assert kind == "success" and f"20{NB}000 Perioden" in text and "Zufalls-Seed 1814" in text and "Innerhalb des Bandes" in text
    kind, text = ev.verdict_mc(dict(m, dev=-0.0301))
    assert kind == "warning" and "Außerhalb" in text


def test_kernfragen_texts():
    kind, text = ev.verdict_kern1(1.0, (0.422, 1.193, 1.3, 1.3), 0.621)
    assert kind == "info" and "ganzzahlig bis d = 0 Perioden (ab d = 1 instabil, bis d = 3 geprüft)" in text and "θ* = 0,62" in text
    assert "Reglergrenze" in ev.verdict_kern1(1e4, (0.9, 0.9, 0.9), None)[1] and "bis θ = 3 Perioden stabil" in ev.verdict_kern1(1e4, (0.9, 0.9, 0.9), None)[1]
    rows = [{"d": 1, "rkrit": 6.854, "R_soft": 8.225, "decay_soft": 213}]
    assert "mindestens 6,85" in ev.verdict_kern2(1, rows)[1] and "braucht 213 Perioden" in ev.verdict_kern2(1, rows)[1]
    assert "alle gemessenen Entwürfe stabil (R = 10⁻³ bis 10⁶, Q = I)" in ev.verdict_kern2(0, rows)[1]
    kind, text = ev.verdict_kern3(1, 0.339, 0.335, 1.531, 0.01, 1.936, 1.0)
    assert "Mehrkosten +33,9" in text and "von 0,34 bis 1,53" in text and "von 0,01 bis 1,94" in text
    assert "braucht es keine Vorhersage" in ev.verdict_kern3(0, None, None, None, None, None, 1.0)[1]


def test_poly_and_fraction_strings():
    from fractions import Fraction as F

    assert ev.frac_str(F(4, 2)) == "2" and ev.frac_str(F(-4, 5)) == "-4/5"
    assert ev.poly_str([F(1), F(-2), F(11, 5), F(-4, 5)]) == "z³ − 2z² + 11/5z − 4/5"
    assert ev.poly_str([F(1), F(0), F(0)]) == "z²"


def test_largest_stable_d_counts_from_zero_without_gaps():
    assert ev.largest_stable_d((0.4, 0.9, 1.2, 0.9)) == 1
    assert ev.largest_stable_d((0.4, 1.2)) == 0
    assert ev.largest_stable_d(tuple(np.full(5, 0.5))) == 4
