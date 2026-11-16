# Verzug: wann ein Regelgesetz den Takt nicht mehr hält – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-regelung-verzug-demo.streamlit.app/)**

Demo zum Thema **Regelung (von LQR bis MPC)** auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Jede Demo dazu hebt eine Annahme auf; die Schwäche, an der diese ansetzt: **Entwurf und
Anwendung geschehen im selben Takt.** Das LQR-Gesetz ([`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo)), der endliche Horizont ([`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo)),
das QP mit Stellgrenzen ([`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo)) und der Beobachter ([`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo)) werden gerechnet, als dauerte das Rechnen nichts.
Hier ergibt sich das Ergebnis erst **d Perioden später**. Weitere Nachbarn zum selben Thema: [`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo), [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo),
[`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) (Totzeit in der Strecke eines PI-Reglers) und [`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo) (Zulässigkeit mit Verzug ist dort nicht gemessen).
Als Kontrast gibt es das Thema „Rückkopplung und Verzögerung“: dasselbe Thema am Lager, dort ohne Reglerentwurf ([`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo), [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo),
[`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo), [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo)); die stetige Gleichung mit Verzögerung rechnet die
[`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo), den Zeitaktualisierungsschritt ohne Rauschmodell hat das [`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo).
Vehikel ist der **Wagen** (Doppelintegrator, Q = I) und als Handbeispiel der Skalar x⁺ = x + u, **nicht das Lager**.

## Warum dieses Problem

Ein LQR-Gesetz u = −K x wird ohne Verzug entworfen. Braucht das Rechnen d Perioden, steht die Stellgröße erst d Takte nach der Messung bereit: u_k = −K x_(k−d). **Wie viel Verzug hält der Entwurf, was kostet es, ihn weich genug zu machen, und was bringt die Vorhersage des Zustands?**
Am Wagen mit Q = I ist der Entwurf mit d = 1 für R = 1, 0,1, 0,01 und 0,001 instabil (Spektralradius 1,193 / 1,321 / 1,346 / 1,349 statt 0,422 / 0,362 / 0,380 / 0,382). Ab R = 6,854 (d = 1), 82,79 (d = 2) und 369,9 (d = 3) hält er wieder,
dann aber mit Abklingzeiten von 213 bis 548 Perioden (d = 1 bis 4, R = 1,2 · R_krit); derselbe große R ohne Verzug klingt nach 11 bis 40 Perioden ab, der Entwurf R = 1 ohne Verzug nach 5. Rechnet man den Zustand vor (Vorhersage), ist die Schleife im exakten Modell wieder die ohne Verzug und kostet nur d blinde Perioden; das Gegenstück ist die Modellabhängigkeit:
mit falsch angenommenem Verzug (um höchstens 2 Perioden daneben) und R ≤ 1 war die Schleife in 2 000 von 2 000 zufälligen Fällen instabil (R = 10^U(−3, 0), d = 1 bis 4; bei großem R nicht, B12), und die Toleranz gegen eine falsche Beschleunigungsverstärkung schrumpft mit d (R = 1: b von 0,01 bis 1,94 bei d = 0, von 0,34 bis 1,53 bei d = 1, von 0,77 bis 1,25 bei d = 5).
**Alle Aussagen gelten für den Wagen mit Q = I, bei exakt bekanntem ganzzahligem Verzug, ohne Stellgrenzen und ohne Messrauschen.** Ausnahmen nennt jede Befundzeile selbst: Rechenzeit als Bruchteil der Periode (Experiment 3), zufälliges Q (B4), der Skalar (B6) und das absichtlich falsch angenommene b und dh (B11, B12). Der Satz „jeder LQR-Entwurf wird mit Verzug instabil“ wäre falsch: siehe B3 und B4.
Alles ist **eigene Herleitung und Messung in diesem Modell**; die Idee, eine Totzeit durch Vorhersage aus der Schleife zu nehmen, ist als Smith-Prädiktor bekannt (siehe Quellen). Diese Demo sagt „Vorhersage“, nicht „Smith-Prädiktor“, und behauptet keine Gleichheit mit dem dortigen Regler. Eine Neuheit wird nicht behauptet.

## Modell

**Zeitlage (gilt für jede Zahl):** x_(k+1) = A x_k + B (b u_k + w) mit A = [[1, 1], [0, 1]], Ts = 1 Periode, Q = I, Kostengewicht R. Eine Stellgröße u_k wirkt immer auf x_(k+1); **d = 0 heißt „Rechnen dauert nichts“** und entspricht L = 1 der Lager-Demos (dort „eine Periode Wirkverzug“); jedes weitere d verlängert um eine Periode, d Perioden Verzug entsprechen L = d + 1 (so auch bei B6).
b ist die wahre Beschleunigungsverstärkung (der Regler nimmt 1 an), w eine konstante Störkraft (nur in Experiment 6 ungleich 0).

- **Zwei Fassungen von B.** Standardfassung B = [0, 1]ᵀ für alle ganzzahligen Verzüge (Experimente 1, 2, 4, 5, 6); exakte Abtast-Halte-Fassung B = [½, 1]ᵀ der stetigen Strecke x″ = u nur in **Experiment 3** (Rechenzeit als Bruchteil der Periode: dort hat ein Bruchteil bei B = [0, 1]ᵀ keine Bedeutung) und in B3 (Gegenprobe). Wie weit die Aussagen am Eintrag ½ hängen, ist nur für R_krit(1), R_krit(2) und ρ bei d = 1 gemessen (B3).
- **Entwurf:** K = (k₁, k₂) aus der diskreten Riccati-Gleichung (`scipy.linalg.solve_discrete_are`), u = −K x ohne Faktor ½ (die Kostenfunktion ist Σ x'Qx + R u²).
- **Verzug:** u_k = −K x_(k−d) für k ≥ d, bis k = d − 1 liegt keine Messung vor (blind, u = 0). Charakteristisches Polynom (eigene Herleitung durch Determinante und über x_(k+2) − 2x_(k+1) + x_k = u_k, zwei Rechenwege im Orakel-Test):
  **P_d(z) = z^d (z − 1)² + k₂ (z − 1) + k₁**, Grad d + 2, dazu d Nullwurzeln der Zustandsmatrix. Die Schleife hält, wenn der Spektralradius ρ (größter Wurzelbetrag) unter 1 liegt; ρ kommt aus `np.roots` von P_d, nicht aus der Zustandsmatrix (deren Nullwurzeln rauschen).
- **Stellverzug und Messverzug** (die Stellgröße wirkt d Perioden später, bzw. sie rechnet mit dem d Perioden alten Zustand) haben ohne Vorhersage dieselben Wurzelbeträge (B5, 28 Fälle gemessen).
- **Vorhersage:** x̂_k = A^dh x_(k−d) + Σ_(j<dh) A^(dh−1−j) B u_(k−dh+j), u_k = −K x̂_k; dh ist der vom Regler angenommene, d der wahre Verzug. Ohne Störung und mit b = 1, dh = d ist die Schleife wieder die ohne Verzug (B8).
- **Rechenzeit als Bruchteil (Experiment 3):** θ = m + τ Perioden; der Regler misst zur Taktzeit, rechnet θ und stellt dann. In der Periode wirkt erst die alte, dann die neue Stellgröße: x_(k+1) = A x_k + Γ₁(τ) u_(k−m−1) + Γ₀(τ) u_(k−m) mit Γ₀ = [(1−τ)²/2, 1−τ]ᵀ, Γ₁ = [(1−(1−τ)²)/2, τ]ᵀ.
- **Abklingzeit** heißt: letzte Periode mit ‖x_k‖₂ über 1 % der Spitze, Stoß x₀ = (1, 0), Zähleinheit Periode (wie in den Lager-Demos, dort mit |i_t|). Die Schwelle ist eine Wahl, keine Naturgröße.
- **Kosten** Σ x'Qx + R u² nach der Anfangsabweichung, blinde Perioden mit u = 0; **Dauerfehler** x₁* = Gleichgewichtslage bei konstanter Störkraft w; **Streuung** = stationäre Kosten je Periode unter Störkraft N(0, 1) je Periode.
- **Herleitung des Dauerfehlers mit Vorhersage** (eigene Rechnung): Im Gleichgewicht ist x₂ = 0 und u* = −w, damit b u* + w = 0 mit b = 1. Die Vorhersage liefert dann x̂₁ = x₁ + u* d(d − 1)/2 und x̂₂ = d u*. Aus u* = −(k₁ x̂₁ + k₂ x̂₂) folgt k₁ x₁ = w (1 + k₂ d + k₁ d(d − 1)/2), also x₁* = (w/k₁)(1 + k₂ d + k₁ d(d − 1)/2); ohne Vorhersage ist x̂ = x und x₁* = w/k₁. Das ist die Lage, in der die Gleichungen im Gleichgewicht sind; erreicht wird sie nur, wenn die Schleife stabil ist (Beispiel: B14, Handbeispiel in der App).
- Die Formeln stehen im Expander 📐 der App (auch die Herleitung, die Abklingzeit und die Streuung). **Zufall** gibt es nur im Monte-Carlo-Tab von Experiment 6 (Zufalls-Seed, ganzzahliger SplitMix64 aus `rvz_rng.py`, `normal()` ohne `math.log`/`math.cos`, damit die Folge auf jeder Plattform dieselbe ist) und in der Vorrechnung `tools/make_frozen.py` (die Zufallszählungen der Befunde) sowie in einigen Tests (Zufalls-Seed jeweils im Test).

## Methodik

- **Zwei Rechenwege je Kern:** Wurzeln von P_d und Eigenwerte der Zustandsmatrix; Stellverzug und Messverzug; Γ in geschlossener Form und per Van-Loan-Matrixexponential; Kosten per Lyapunov-Gleichung und per Simulation; Gleichgewicht per Formel, Gleichungssystem (in Brüchen) und Simulation.
- **Brüche (`fractions`)** für die Handbeispiele: Schur–Cohn-Prüfung und charakteristisches Polynom (Faddeev–LeVerrier) exakt, Bahnen und Gleichgewichtslagen exakt. **Gleitkomma nur mit Toleranz** (Eigenwerte, DARE, `expm`).
- **R_krit** und **θ\*** durch Bisektion (R_krit: auf log R, grobes Gitter mit 10 Punkten je Dekade, dann 60 Halbierungen; θ\*: Gitter 0,01, dann 60 Halbierungen); die Annahme „genau ein Wechsel stabil/instabil“ ist an einem feinen Gitter und an Zufallspunkten geprüft (B3, B7), nicht bewiesen.
- **Beste statische Verstärkung** (ohne Vorhersage) ist **vorgerechnet** (`tools/make_frozen.py` → `data/rvz_frozen.json`, Gitter 101 × 151, dann Nelder–Mead): kein Beweis für das globale Minimum, im letzten Bit plattformabhängig; in der App als „vorgerechnet“ gekennzeichnet und nur für R = 1 mit Kosten hinterlegt. Je Kennzahl neu abgestimmt, also ein günstiger Gegner.
- **Live-Rechnung** in der App: alles bis auf die vorgerechneten Zeilen und den Monte-Carlo-Lauf. Erster Aufruf rund 2,5 Sekunden, Änderung von R rund 1,3 Sekunden, jede andere Eingabe etwa 0,1 Sekunden (Messung dieser README mit dem AppTest-Rahmen, ohne Netz; kein Garantiewert, nicht im Test); der Monte-Carlo-Lauf braucht bei 20 000 Perioden etwa 1,2 Sekunden. Die Experimente sind gecacht (`st.cache_data`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Wagen, Q = I, B = [0, 1]ᵀ (außer B3-Gegenprobe und B7), ohne Stellgrenzen, bekannter ganzzahliger Verzug, Zeitlage wie oben. Die Spalte „Test“ nennt die Funktion in `tests/test_claims.py`, die genau diese Form prüft („Messung“ heißt: Zahl dieser README aus `tools/make_frozen.py`
oder aus früheren Messläufen vor dem Bau, im Test nur auf Konsistenz oder als Band; Bänder dort, wo Plattformen abweichen können). Nelder–Mead-Zahlen sind nie bitgleich zugesagt.

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Spektralradius gegen d | d = 0 / 1 / 2 / 3: R = 1: 0,422 / 1,193 / 1,329 / 1,339; R = 0,1: 0,362 / 1,321 / 1,416 / 1,403; R = 10: 0,650 / 0,961 / 1,160 / 1,211; R = 100: 0,795 / 0,764 / 0,987 / 1,072; R = 1 000: 0,881 / 0,857 / 0,858 / 0,951. Mit d = 1 sind R = 1, 0,1, 0,01 und 0,001 instabil (1,193 / 1,321 / 1,346 / 1,349). Ohne Verzug (d = 0) sind alle 4 001 Gitterpunkte R = 10⁻³ bis 10⁶ stabil. | `test_b1_*` |
| B2 | Größtes stabiles d je R | R = 1 und kleiner sowie R = 3: d = 0; R = 10 und 30: 1; R = 100 und 300: 2; R = 1 000: 3; R = 3 000: 5; R = 10 000: 7; R = 100 000: 12. In allen 13 gemessenen R bilden die stabilen d (bis d = 119 geprüft) eine zusammenhängende Reihe von 0 an. | `test_b2_*` (Messung vorgerechnet; Test: 6 R bis d = 40) |
| B3 | Kleinstes stabiles R je d | R_krit = 6,854 / 82,79 / 369,9 / 1 092 / 2 555 / 5 147 / 9 336 / 15 673 für d = 1 bis 8. Genau ein Wechsel instabil → stabil im Gitter von 4 001 Punkten (R = 10⁻³ bis 10⁶) in 8 von 8 d; 160 000 Zufallsprüfungen (20 000 zufällige R, d = 1 bis 8, Zufalls-Seed 1820): 0 Abweichungen von „stabil genau dann, wenn R > R_krit“. Mit B = [½, 1]ᵀ: ρ bei d = 1 für R = 1 / 0,1 / 0,01 / 0,001 ist 1,198 / 1,339 / 1,370 / 1,374 (instabil), R_krit = 6,888 (d = 1) und 82,71 (d = 2). Warum R_krit so wächst, ist nicht erklärt. | `test_b3_*` (Messung vorgerechnet; Test: Gitter 4 001 Punkte und 2 000 Zufallspunkte neu gerechnet) |
| B4 | Nicht nur Q = I | Zufällige Entwürfe nur für d = 1, R = 10^U(−3, 4) (der Exponent gleichverteilt zwischen −3 und 4), ganzzahliger SplitMix64: Q diagonal mit zwei unabhängigen Einträgen 10^U(−2, 2): **1 702 von 4 000** stabil (je Dekade von R, also R ab 10^k bis unter 10^(k+1), stabil/gesamt: 10⁻³: 0/574, 10⁻²: 0/571, 10⁻¹: 63/601, 10⁰: 231/583, 10¹: 342/546, 10²: 520/579, 10³: 546/546); volles zufälliges Q (G Gᵀ + 10⁻³ I mit einer 2 × 2-Matrix G aus vier unabhängigen Normalzahlen N(0, 1)): **1 725 von 4 000** (Zufalls-Seed 1810 bzw. 1811). Bei diagonalem Q hält d = 1 für R unter 10⁻¹ in keinem der 1 145 Entwürfe (0/574 und 0/571); bei vollem Q in 1 von 1 199. **R_krit hängt an Q; die Zahlen der übrigen Befunde gelten für Q = I.** | `test_b4_*` (Messung vorgerechnet; Test rechnet neu, Zählungen ± 2) |
| B5 | Stellverzug = Messverzug | Die Beträge der Wurzeln beider Modelle stimmen überein (28 Fälle: R = 10, 1, 0,1, 0,01; d = 0 bis 6; Abweichung in einem früheren Rechenlauf 2,9·10⁻¹⁵, im Test Band 10⁻⁶ wegen der rauschenden Nullwurzeln). | `test_b5_*` |
| B6 | Skalar x⁺ = x + u, q = r = 1 | p = φ (Goldener Schnitt), K = 1/φ = 0,618. Spektralradius d = 0 bis 5: 0,382 / 0,786 / 1,000 / 1,070 / 1,097 / 1,108. Das Polynom z^(d+1) − z^d + K ist das des Lagers ohne Leitung mit L = d + 1 und α = K; **bei d = 2 liegt K genau auf der Lager-Grenze 2 sin(π/10) = (√5 − 1)/2** (ρ liegt um weniger als 10⁻⁹ neben 1; „genau“ nur im Band, nie ρ == 1). Exakt in Brüchen: K = 3/5 stabil für d = 0, 1, 2 und instabil für d = 3, 4; K = 5/8 bei d = 2 instabil. Das ist eine Gleichheit zweier Zahlen, keine Erklärung. | `test_b6_*` |
| B7 | Rechenzeit als Bruchteil (B = [½, 1]ᵀ) | Kritische Rechenzeit θ\* = 2,114 / 1,130 / 0,800 / 0,621 / 0,464 / 0,437 / 0,434 Perioden für R = 100 / 10 / 3 / 1 / 0,1 / 0,01 / 0,001; in 7 von 7 R genau ein Wechsel im Gitter (3 001 Punkte, θ = 0 bis 3), und 5 000 Zufallspunkte (500 zufällige R = 10^U(−3, 2) mit je 10 gleichverteilt gezogenen θ zwischen 0 und 3, Zufalls-Seed 1821): 0 Abweichungen. Spektralradius für R = 1 bei θ = 0 / 0,25 / 0,5 / 0,75 / 1: 0,434 / 0,693 / 0,918 / 1,076 / 1,198. **Eine halbe Periode Rechenzeit hält R = 1 (ρ = 0,918) und R = 0,1 nicht (1,031).** Die dominante Wurzel liegt bei θ\* im Winkel 41,5° (R = 10), 59,2° (R = 1), 69,3° (R = 0,1). | `test_b7_*` |
| B8 | Vorhersage im exakten Modell | Ohne Störung, b = 1 und dh = d gilt in Brüchen x_(d+m) = (A − BK)^m A^d x₀ (Test: K = (2/5, 6/5), d = 1 bis 5, zwei Startwerte, m = 0 bis 9: 10 von 10; früherer Rechenlauf: 15 von 15 mit drei Startwerten) und das charakteristische Polynom der Schleife (Dimension 3d + 2) ist z^(3d) (z² + (k₂ − 2) z + 1 − k₂ + k₁): 12 von 12 (d = 1 bis 4, drei Gewinne; früherer Rechenlauf: 15 von 15 bis 17 × 17). **Die Vorhersage nimmt den Verzug aus den Wurzeln; übrig bleiben d blinde Perioden.** | `test_b8_*` |
| B9 | Ist die Vorhersage der optimale Gewinn? | Der Gewinn der Vorhersage [K A^d, K A^(d−1) B, …, K B] stimmt mit dem DARE-Gewinn des um die Stellgrößen-Vergangenheit erweiterten Systems (Stellverzug, Zustand (x_k, u_(k−d), …, u_(k−1))) überein: R = 10, 1, 0,1, 0,01 und d = 1 bis 6, größte Abweichung unter 10⁻⁸ (früherer Rechenlauf 6,6·10⁻¹³). Unter Prozessrauschen (R = 1) senkt keine der 2 100 zufälligen Störungen des Gewinns die stationären Kosten (700 je d = 1 bis 3; jede Komponente des Gewinns wird mit 1 + 0,1·N(0, 1) multipliziert, ganzzahliger SplitMix64, Zufalls-Seed 1826; 2 002 der 2 100 gestörten Schleifen sind stabil und nur diese zählen), die Vorhersage ist also lokal kostenminimal unter den gemessenen Störungen (im Test neu gerechnet; die Verteilung der Störungen eines früheren Rechenlaufs ist nicht erhalten). | `test_b9_*` |
| B10 | MPC-Brücke | Der Gewinn der ersten Stufe des unbeschränkten N-Schritt-Problems mit Terminalkosten P_f = DARE ist K für N = 1 bis 12 (Abweichung unter 10⁻⁹; Messung 4,4·10⁻¹⁶ bei R = 1, 6,7·10⁻¹⁶ bei R = 0,1). Mit P_f = Q weicht er um 0,744 / 0,244 / 0,0006 ab (N = 1 / 2 / 5, R = 1) und für N = 12 um weniger als 10⁻⁵. Wendet ein unbeschränktes MPC mit DARE-Terminalkosten, das d Perioden zum Rechnen braucht, seine erste Stufe auf den verspäteten Zustand an, ist es in diesem Modell die Schleife u_k = −K x_(k−d) (Modellierungsannahme dieser Demo, nicht gemessen). Mit Stellgrenzen nicht gemessen. | `test_b10_*` |
| B11 | Modellfehler im Eingang | Der Regler nimmt b = 1 an, die Strecke hat b. Stabiler Bereich von b mit Vorhersage (dh = d), R = 1: d = 0: bis 1,936 (untere Grenze nicht erreicht: bis zum Gitterrand 0,01 stabil); d = 1: 0,335 bis 1,531; d = 2: 0,536 bis 1,383; d = 3: 0,659 bis 1,319; d = 4: 0,728 bis 1,286; d = 5: 0,769 bis 1,250. R = 0,1: d = 0: bis 1,592; d = 1: 0,595 bis 1,405; d = 5: 0,822 bis 1,179. In allen 60 zufälligen R = 10^U(−2, 1) (Zufalls-Seed 1824) fällt die Breite des stabilen Bereichs von d = 1 bis 5 streng. | `test_b11_*` (Zufallszählung vorgerechnet) |
| B12 | Falsch angenommener Verzug | Spektralradius bei d = 1 bis 4 und dh = d − 2 bis d + 2 (dh ≥ 0): stabil in **24 von 75** Zellen mit dh ≠ d; je R (je 15 Zellen): R = 1 000: 15, R = 100: 8, R = 10: 1, **R = 1: 0, R = 0,1: 0**. Beispiel R = 1, d = 1: dh = 0 (keine Vorhersage) 1,193, dh = 2: 1,671, dh = 3: 1,855. Gegenprobe: 2 000 zufällige (R, d, dh) mit R = 10^U(−3, 0), d = 1 bis 4 und dh gleichverteilt von d − 2 bis d + 2 (dh ≠ d, dh ≥ 0, also der angenommene Verzug um höchstens 2 Perioden daneben): **in 2 000 von 2 000 instabil** (Zufalls-Seed 1823). | `test_b12_*` |
| B13 | Beste statische Verstärkung ohne Vorhersage | Die Summe der Wurzeln von P_d ist 2 für d ≥ 1 (Koeffizient von z^(d+1)), also ρ ≥ 2/(d + 2) für jede statische Verstärkung auf dem verspäteten Zustand allein (ohne Vorhersage; Vieta; 20 000 Zufallsgewinne ohne Verletzung, Zufalls-Seed 1822). Kleinster gefundener ρ: d = 1: 0,66667 bei K = (1/27, 1/3), **dreifache Wurzel 2/3, die Schranke genau erreicht (exakt in Brüchen)**; d = 2: 0,78868; 3: 0,84495; 4: 0,87749; 5: 0,89871; 6: 0,91366 (numerisch, nicht als globales Minimum bewiesen). **Deadbeat (ρ = 0) ist mit Verzug und einer statischen Verstärkung auf dem verspäteten Zustand allein (ohne Vorhersage) nicht erreichbar** (ohne Verzug mit K = (1, 2): (A − BK)² = 0). Mit Vorhersage ist es erreichbar: für K = (1, 2) ist das charakteristische Polynom der Schleife z^(3d+2), exakt in Brüchen für d = 1 bis 5 geprüft (numerisch ρ höchstens 0,07 für d = 1 bis 5, die Abweichung von 0 ist die schlechte Konditionierung der Mehrfachwurzel). Ohne Vorhersage hält mit d = 1 bis 4 der Gewinn bis b = 2,95 / 3,07 / 3,10 / 3,11 (ρ = 0,667 / 0,789 / 0,845 / 0,877); die letzte Stelle der Obergrenzen hängt am Gewinn, weil die Mehrfachwurzel schlecht konditioniert ist. | `test_b13_*` (Optima als Band 10⁻³, „nicht kleiner als 2/(d + 2)“ als Absolutaussage) |
| B14 | Dauerfehler bei konstanter Störkraft | Mit Vorhersage x₁* = (w/k₁)(1 + k₂ d + k₁ d(d − 1)/2), ohne Vorhersage w/k₁ (exakt in Brüchen, drei Gewinne, d = 0 bis 6 bzw. 0 bis 4; für den DARE-Gewinn gegen Gleichungssystem und Simulation). Zahlen (w = 1, R = 1), d = 0 bis 5, mit Vorhersage: 2,369 / 5,316 / 9,264 / 14,21 / 20,16 / 27,10 (Faktor 1 / 2,24 / 3,91 / 6,00 / 8,51 / 11,44). Beste statische Verstärkung (d = 1 bis 5, vorgerechnet): 27,0 / 77,6 / 153 / 255 / 381. Weicher Entwurf (R = 1,2 · R_krit, d = 1 bis 5): 4,53 / 12,6 / 24,6 / 40,8 / 60,9. **Die Vorhersage ohne Störmodell vergrößert den Dauerfehler gegenüber d = 0; bei d = 1 ist er größer als beim weichen Entwurf (5,32 gegen 4,53), ab d = 2 kleiner (9,26 gegen 12,6).** Warum (die Vorhersage rechnet mit einer Bewegung, die die Störkraft aufhebt): Herleitung im Abschnitt „Modell“ dieser README und im Expander 📐 der App, nicht als Ursache einer Messung geprüft. Handbeispiel (K = (2/5, 6/5), d = 1, w = 1): 11/2 mit Vorhersage; die 5/2 ohne Vorhersage ist nur der Fixpunkt einer instabilen Schleife und wird nie erreicht (ρ siehe Handbeispiel in der App). | `test_b14_*` |
| B15 | Kosten und Abklingzeit nach dem Stoß | Kosten Σ x'Qx + R u² mit R = 1, x₀ = (1, 0), ohne Verzug 2,947 (= P₁₁). Mit Vorhersage 3,947 / 4,947 / 5,947 / 6,947 für d = 1 bis 4 (**Mehrkosten 33,9 / 67,9 / 101,8 / 135,7 %; genau P₁₁ + d**, denn der Wagen bleibt in den blinden Perioden stehen); weicher Entwurf R = 1,2 · R_krit: 11,58 / 13,13 / 16,70 / 20,61 (+293 bis +599 %); beste statische Verstärkung (vorgerechnet): 4,646 / 6,855 / 9,193 / 11,57 (+58 / +133 / +212 / +293 %). Bei x₀ = (0, 1) (der Wagen fährt in den blinden Perioden weiter): ohne Verzug 4,613; Vorhersage 13,30 / 28,88 / 53,35 / 88,72; beste statische 21,61 / 74,86 / 189,0 / 388,4; weicher Entwurf 102,6 / 276,1 / 663,3 / 1 336. **Auch mit Vorhersage kostet der Verzug etwas (die blinden Perioden), die Vorhersage ist aber in jeder gezeigten Zeile (R = 1, d ≤ 4, Q = I) die kleinste Zahl unter den Gesetzen mit Verzug.** Abklingzeit (Stoß x₀ = (1, 0), d = 0 bis 4): Vorhersage 5 / 6 / 7 / 8 / 9 Perioden; weicher Entwurf (R = 1,2 · R_krit) 213 / 314 / 434 / 548 (d = 1 bis 4; bis d = 8: 996); beste statische Verstärkung 21 / 37 / 52 / 67 (ρ = 0,667 / 0,789 / 0,845 / 0,877). Mit Vorhersage wächst die Abklingzeit um genau 1 je d, bei der besten statischen Verstärkung um 15 bis 16. **Fairer Vergleich für den weichen Entwurf:** derselbe große R (R = 1,2 · R_krit, je d ein anderer) klingt ohne Verzug nach 11 / 21 / 30 / 40 Perioden ab (d = 1 bis 4; 49 / 58 / 68 / 77 für d = 5 bis 8); der Verzug allein macht daraus 213 / 314 / 434 / 548 (d = 8: 996). Der Entwurf R = 1 ohne Verzug braucht 5, aber das ist ein anderer, kräftigerer Entwurf. | `test_b15_*` |
| B16 | Streuung unter Prozessrauschen | Kosten je Periode E[x'Qx + R u²] bei Störkraft N(0, 1) je Periode, stationär per Lyapunov (R = 1): ohne Verzug 4,613; mit Vorhersage 13,30 / 28,88 / 53,35 / 88,72 (d = 1 bis 4); beste lineare Verstärkung ohne Vorhersage (Nelder–Mead, vorgerechnet) 21,61 / 74,86 / 189,0 / 388,4. Das sind dieselben Zahlen wie die Kosten für x₀ = (0, 1) in B15 (Zeile „Kosten und Abklingzeit nach dem Stoß“; ein Rauschstoß setzt die Geschwindigkeit um 1; Identität durch Linearität). Monte-Carlo (d = 2, Zufalls-Seed 1814, Mittel ohne die ersten 1 000 Perioden): 29,45 bei 20 000, 29,00 bei 50 000 Perioden gegen 28,88 aus Lyapunov (Abstand 1,97 % bzw. 0,43 %; bei 10 000 Perioden 3,96 %, außerhalb des Bandes von 3 %); im Orakel-Test 40 000 Perioden, Band 3 %. | `test_b16_*` |

### Korrekturen gegenüber früheren Annahmen

- **„Jeder gemessene Entwurf ist mit einer Periode Verzug instabil“ (frühere Messung)** gilt für R = 1, 0,1, 0,01, 0,001 bei Q = I (die vier gemessenen) und bleibt dort richtig. Als allgemeiner Satz wäre er **falsch**: Mit Q = I hält d = 1 ab R = 6,854 (B3), mit zufälligem Q hält d = 1 in 1 702 bzw. 1 725 von 4 000 Entwürfen (B4). Die Demo nennt Zählungen statt „jeder“.
- **Zufallszählung B4 weicht von der früheren Messung ab:** Die frühere Messung (NumPy-Zufallsstrom) zählte 1 722 von 4 000 (Q diagonal) und 1 761 von 3 999 (Q voll, eine Instanz ohne DARE-Lösung übersprungen). Im Bau mit dem ganzzahligen SplitMix64 neu gemessen: 1 702 von 4 000 und 1 725 von 4 000 (keine übersprungen). Die Größenordnung (rund 43 bis 44 % der Entwürfe stabil: 42,6 und 43,1 % im Bau, 43,1 und 44,0 % in der früheren Messung; die früher genannten „43 bis 45 %“ stimmten nicht, 45 % erreicht keine Zählung) und der Verlauf je Dekade von R (bei diagonalem Q unter R = 10⁻¹ nie, bei 10³ in allen gezählten Entwürfen) bleiben; die Zahl hängt am Zufallsstrom (schon die Binomialstreuung beträgt bei 4 000 Entwürfen etwa ± 30) und ist mit ihm angegeben.
- **„Vorhersage noch nicht gemessen“ (frühere Annahme)** ist erledigt: Das Ergebnis ist zweischneidig, exakt im Modell (B8), aber empfindlich gegen falschen Verzug (B12) und falsche Verstärkung (B11).
- **Wechselzahlen:** Die frühere Messung zählte beim b-Bereich „2 Wechsel“ des Gitters (Anfang und Ende des stabilen Intervalls); hier steht „ein stabiles Intervall“ (derselbe Befund, anders gezählt). Bei d = 0 reicht der stabile Bereich bis zum Gitterrand 0,01.
- **Mehrkosten „genau d/2,947“:** Eine frühere Notiz schrieb die Mehrkosten als d/2,947; gemessen und im Test geprüft ist J = P₁₁ + d für x₀ = (1, 0) und Q = I, also Mehrkosten d/P₁₁ (P₁₁ = 2,947 bei R = 1).
- **Verdikt „ohne Störungsmodell hilft die Vorhersage dem Dauerfehler nicht“ (frühere Annahme zu Experiment 6)** ist zu pauschal: ab d = 2 liegt der Dauerfehler mit Vorhersage unter dem des weichen Entwurfs (B14). Gemessen und im Verdikt steht: gegenüber d = 0 vergrößert die Vorhersage den Dauerfehler.
- **Presets und Regler:** Die Experimente liegen auf einer Seite; die Seitenleiste hat nur R (log₁₀) und d, die Regler der Experimente stehen bei ihnen. Die Seitenleiste „Schnellstart“ setzt alle Regler (nicht angegebene auf den Standardwert).
- Messungen früherer Rechenläufe, die im Bau **nicht** neu gerechnet werden, sind in der Tabelle als „früherer Rechenlauf“ oder „frühere Messung“ gekennzeichnet (B5, B8, B9): im Test steht jeweils eine kleinere Fassung.

## Quellen

Nur so zitiert, wie gelesen (Stufen: W Wortlaut gelesen, A Abstract):

- Findeisen und Allgöwer, „Computational delay in NMPC“ (ADCHEM 2003, W): Die Rechenzeit werde in Stabilitätsbeweisen oft vernachlässigt, führe aber zu einem Verzug zwischen Zustandsinformation und Stellsignal, der die Regelgüte stark mindern oder die Schleife instabil machen könne. Nichtlineares MPC; **hier wird davon nichts nachgerechnet**.
- Wolf, Würth und Marquardt (CDC 2011, W): ein ignorierter Rechenverzug gefährdet die Stabilität; als Faustregel solle das Verhältnis von Verzug zu Abtastzeit deutlich unter 1 liegen, wenn man den Verzug vernachlässigt. B7 ist keine Prüfung dieser Faustregel, nur eine Zahl in derselben Richtung.
- Orzechowska, Dissertation Coventry 2014 (A, nur das Abstract): eine Lieferzeit im Lager wird mit einem Smith-Prädiktor behandelt, das Ergebnis sei kein MPC mehr. Åström und Murray, Übung 12.8 (nur der Aufgabentext gelesen; die Fassung ist nicht festgehalten, Übungsnummern können sich zwischen Auflagen verschieben).
- Rawlings, Mayne und Diehl, Abschnitt 2.5.1 (S. 132, Stufe W nach einem früheren Quellenbericht; die Stelle wurde beim Bau nicht erneut geöffnet): beim unbeschränkten Problem mit Terminalkosten aus der DARE hängt der Gewinn der ersten Stufe nicht vom Horizont ab, hier gemessen für N = 1 bis 12 (B10, Zeile „MPC-Brücke“). Das PDF hat den Vermerk „Weitergabe nur mit Erlaubnis des Verlags“: nur Gleichung und Befund mit Fundstelle, in eigenen Worten, nichts wörtlich.
- Alle Formeln (Polynom P_d, Vorhersage-Gewinn, Dauerfehler, Γ) sind eigene Herleitung in diesem Modell, bei diesen Parametern; die Quellen belegen sie nicht.

## Ehrliche Grenzen

- **Nur der Wagen mit Q = I** (zwei Zustände); für andere Q nur die Zählung bei d = 1 (B4); der Skalar steht nur als Handbeispiel. Doppelintegrator mit B = [½, 1]ᵀ nur in B3 und B7.
- **Kein Messrauschen mit Vorhersage:** nur Prozessrauschen auf der Kraft (B16); ein Beobachter für den verspäteten Zustand ist nicht gebaut (Verweis `kalman-filter-demo`, `offsetfrei-demo`).
- **Feste Verzüge:** ganzzahlig oder fester Bruchteil; kein Jitter, kein Überspringen eines Taktes mit Halten der alten Stellgröße. Die Vorhersage bei nicht ganzzahliger Rechenzeit ist nicht gemessen.
- **Modellfehler:** nur der Eingangsfaktor b und der angenommene Verzug dh; nicht der Fehler in A (Kopplung Lage–Geschwindigkeit), und nicht unbekannte Störungen mit Vorhersage außer der konstanten (B14).
- **Ursache:** dass Verzug „Phase kostet“, ist hier **nicht** als Ursache der Instabilität belegt; gemessen ist nur, bei welchem R und d sie eintritt. Warum R_krit(d) so wächst (6,854; 82,79; 369,9 …), ist nicht erklärt; eine geschlossene Formel gibt es nicht, die beste statische Rate für d ≥ 2 ist nur numerisch.
- **Aufwand in Perioden:** wie lange eine Optimierung mit N Schritten braucht und was d daraus folgt, ist nicht gemessen; hier ist d ein Regler.
- **Zusammenhang mit der Phasenreserve** (`lqr-reserven-demo`) ist nicht gemessen.
- **Nelder–Mead:** die beste statische Verstärkung ist ein gefundenes Minimum, kein bewiesenes; „ein günstiger Gegner“ heißt: je Kennzahl neu abgestimmt, aus ein bis zwei Startpunkten.
- Die Zahlen mit „vorgerechnet“ kommen aus `data/rvz_frozen.json`; die Zufallszählungen darin sind mit ihrem Zufalls-Seed angegeben und im Test neu gerechnet (B3, B4), die Nelder–Mead-Zeilen nur als Band.

## Verifikation

- **Handwerte** (`tests/test_dynamics.py`): die fünf Handbeispiele in Brüchen (P₀, P₁, (z − 2/3)³, Bahnen x₁ bis x₄ mit und ohne Vorhersage, x₁* = 11/2, Γ(½) = (1/8, 1/2) und (3/8, 1/2), Schur–Cohn für K = 3/5 und 5/8, Deadbeat); Schur–Cohn gegen `numpy.roots` an 400 Zufallspolynomen.
- **Orakel** (`tests/test_oracle_regelung_verzug.py`, anderer Rechenweg): Wurzeln von P_d gegen Eigenwerte der Zustandsmatrix; Stellverzug gegen Messverzug; charakteristisches Polynom der Vorhersage-Schleife exakt (Faddeev–LeVerrier mit Brüchen); Gewinn der Vorhersage gegen `solve_discrete_are` des erweiterten Systems;
  die direkt gebaute Zustandsmatrix gegen die durch Probieren gebaute; Γ₀, Γ₁ gegen Van Loan und Zweiteilung der Periode; Rechenzeit θ gegen Zeitschritt-Simulation mit `expm`; Kosten per Lyapunov gegen Simulation; Gleichgewicht gegen Gleichungssystem in Brüchen und Simulation;
  R_krit gegen eigene Bisektion über die Zustandsmatrix; K_N gegen gestapeltes Kleinste-Quadrate-Problem; Monte-Carlo gegen Lyapunov (Band 3 %); Skalar-Polynom gegen die Begleitmatrix der Lager-Rekursion.
- **Eigenschaften** (`tests/test_eigenschaften_regelung_verzug.py`): Wurzelsumme und Vieta-Schranke, Linearität in x₀ und w, d = 0 gleich A − BK, Vorhersage gleich Bahn ohne Verzug nach den blinden Perioden, Skalierung von Q und R ändert K nicht, J = P₁₁ + d, τ = 0 und ganzzahliges θ, einmaliges Kreuzen an Zufallspunkten, falscher Verzug bei R ≤ 1.
- **Randfälle** (`tests/test_edge_cases.py`): d = 0, d = 40, Gewinn auf der Grenze (nur als Band), dreifache Wurzel, R von 10⁻⁶ bis 10⁸, ungültige Eingaben abgewiesen, b = 0, dh = 0, θ = 0 und ganzzahlig, die größte Matrix der App (26 × 26).
- **Claims** (`tests/test_claims.py`): jede Zahl der Befunde B1 bis B16 mit Zählung oder Band; je Preset die erwartete Verdikt-Zeile.
- **Verdikte** (`tests/test_verdicts.py`): Schwellen (ρ < 1, Band 10⁻⁹, 3 %-Band, w = 0, d = 0) und jede Zahl im Text an handgesetzten Werten. **Presets, Layout, Oberfläche, Footer, Laufzeit-Importe** (`tests/test_presets.py`, `test_layout.py`, `test_app.py`, `test_footer.py`, `test_runtime_imports.py`).
- **Plattform:** Neu gerechnete Zählungen als Band (± 2), Floats relativ 10⁻⁹, Nelder–Mead und Monte-Carlo nur als Band; exakt verglichen wird nur, was in Brüchen gerechnet ist. Zufall ohne C-Bibliothek (`rvz_rng.py`). Der Linux-Check (`tools/demo_linux_check.py` des Website-Repos) war beim ersten Lauf in einem Test rot (Spektralradius an einer dreifachen Wurzel, Band 10⁻⁹ statt 10⁻⁶, Abweichung 6·10⁻⁹); mit der Toleranz grün, 135 Tests.
- Frühere Messläufe vor dem Bau liegen nicht im Repository und werden nicht importiert; die Messlogik ist neu geschrieben, ihre Zahlen sind als Erwartung behandelt und hier neu gemessen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfragen, sechs Experimente, Grenzen, Expander „In einfachen Worten“, Formeln |
| `data/rvz_frozen.json` | vorgerechnete Messreihen (beste statische Verstärkung, Zufallszählungen, Gitter-Gegenproben) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, SciPy, Plotly |
| `rvz_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `rvz_dynamics.py` | Kern: DARE-Entwurf, Wurzeln, Verzug, Vorhersage, Kosten, Dauerfehler, Rechenzeit, Skalar |
| `rvz_evaluation.py` | Auswertungen der Experimente und Verdikte (reine Funktionen) |
| `rvz_exact.py` | exakte Rechnung mit Brüchen (Schur–Cohn, Faddeev–LeVerrier, Bahnen, Gleichgewicht) |
| `rvz_layout.py` | Fester Kopf (Stylesheet) |
| `rvz_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) |
| `rvz_rng.py` | ganzzahliger SplitMix64, `normal()` ohne C-Bibliothek (Kopie der geprüften Vorlage) |
| `rvz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/make_frozen.py` | erzeugt `data/rvz_frozen.json` (und `--check`) |
| `tests/` | Handwerte, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Stellgrenzen und QP** (ein MPC mit Beschränkungen und Rechenzeit): ein anderes Problem, Verweis auf [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo). Was bei aktiven Beschränkungen mit der Optimalität der Vorhersage geschieht, ist offen.
- **Beobachter und Störungsmodell:** B14 ist die Motivation (ohne Störmodell vergrößert die Vorhersage den Dauerfehler gegenüber d = 0); die Lösung steht in [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo).
- **Schwankende Rechenzeit** (Jitter, übersprungener Takt) und die **Vorhersage bei Bruchteilen der Periode**: nicht gemessen.
- **Eine Brücke zur Phasenreserve** (`lqr-reserven-demo`): nicht gemessen (offene Messlücke); der Name „Smith-Prädiktor“ kommt nur in den Quellen vor.
- **PDF-Export** (nicht gebaut).
- **Messwerte früherer Messläufe für andere Q** über die Zählung B4 hinaus.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Zahlen neu erzeugen: `python tools/make_frozen.py` (etwa 50 Sekunden).

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html).
