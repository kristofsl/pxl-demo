# Topic 2 — Explainable AI

Eén tabulaire dataset, één model, één predictie, en **twee manieren om die te verklaren**:
**SHAP** en **LIME**.

| | |
|---|---|
| **Notebook** | [`explainable-ai-shap-lime.ipynb`](explainable-ai-shap-lime.ipynb) |
| **Dataset** | Breast Cancer Wisconsin (569 × 30, zit in scikit-learn — geen download) |
| **Model** | Random forest (300 bomen, 95,8% accuracy op de testset) |
| **Duur** | ~2 uur les · notebook draait volledig in ± 15 seconden |
| **Voorkennis** | Python, pandas, basis machine learning (train/test, classificatie) |

---

## 1. Eenmalige installatie

Vanuit deze map:

```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

Dit maakt een virtual environment in `.venv/` die **alleen bij dit topic hoort**.

> Wil je exact dezelfde versies als de docent (aanrader als iets niet werkt)?
> Gebruik dan `requirements-lock.txt` in plaats van `requirements.txt`.

## 2. De notebook starten

```bash
.venv/bin/jupyter lab explainable-ai-shap-lime.ipynb
```

JupyterLab opent in je browser. Kies rechtsboven de kernel **Python 3** — dat is automatisch
de kernel uit `.venv`, omdat je JupyterLab vanuit die venv start.

Liever met de venv geactiveerd?

```bash
source .venv/bin/activate
jupyter lab explainable-ai-shap-lime.ipynb
```

En achteraf weer eruit met `deactivate`.

> ⚠️ **Belangrijkste valkuil:** start Jupyter **altijd** via `.venv/bin/jupyter` of met de venv
> geactiveerd. Typ je gewoon `jupyter lab`, dan pakt macOS de systeem-Jupyter en krijg je
> `ModuleNotFoundError: No module named 'shap'` — de pakketten zitten immers in `.venv`.

## 3. Controleren of alles werkt

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace explainable-ai-shap-lime.ipynb
```

Dit voert alle cellen van boven naar onder uit. Geen foutmelding = alles in orde.
Duurt ongeveer 20 seconden.

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 0–1 | Setup, dataset, en de keuze van **de patiënt** die we twee keer gaan verklaren |
| 2 | Het black-box model, waarom `feature_importances_` tekortschiet, en permutation importance van nul opgebouwd |
| **3** | **SHAP** — speltheorie, globale importance, waterfall en force plot |
| **4** | **LIME** — het idee getekend, één uitleg lezen, waarom je LIME níét globaal kunt aggregeren, R² als kwaliteitscontrole, en een demo van de instabiliteit |
| 5 | De twee uitleggen naast elkaar: rangcorrelatie, tekens, en wanneer je welke gebruikt |

Beide methodesecties hebben **dezelfde vijfdelige opbouw**, zodat studenten ze kunnen vergelijken:

> 1. Het idee → 2. De explainer opzetten → 3. Globale uitleg → 4. Lokale uitleg van onze patiënt → 5. De rekening controleren

### De rode draad

De twee methodes leveren een ander soort uitleg op: SHAP geeft **een getal per feature**,
LIME geeft **regels met drempelwaarden**. En waar SHAP altijd hetzelfde antwoord geeft, moet je
bij LIME zelf controleren of de uitleg wel klopt — dat is wat §4.5 doet (R² ≈ 0,48, het
surrogaat zit er 0,13 naast).

En toch: beide methodes wijzen 8 van de 10 dezelfde features aan, met **100% hetzelfde teken**
(§5). Eens zijn ze het dus wel — alleen om verschillende redenen.

---

## Didactische aanknopingspunten

- **§2.1–2.2** — laat zien dat MDI- en permutation-importance een *andere* top 10 geven. Goede
  opener: "welke van de twee is juist?" In §2.2 wordt permutation importance eerst met de hand
  voorgedaan op één kolom (schudden en kijken wat je mist) voor sklearn het werk overneemt.
- **§2.2** — het verlies is klein, ook bij een belangrijke feature. Dat is geen fout: de
  "worst radius / perimeter / area"-kolommen zijn onderling vervangbaar. Mooie voorzet voor
  het correlatieprobleem dat bij SHAP terugkomt.
- **§3.3** — de globale SHAP-importance staat vol medisch plausibele features. Gebruik dat als
  sanity check: had er een patiëntnummer bovenaan gestaan, dan had je data leakage.
- **§3.4** — de waterfall plot is de beste plaat om aan een niet-technisch publiek te tonen.
- **§4.3** — hier leert de klas een LIME-regel lezen, met de uitleg er meteen naast: het getal
  is het verschil tussen twee groepen nepatiënten, geen helling.
- **§4.4** — de vraag die elke student stelt: "waarom middelen we die LIME-gewichten niet
  gewoon, zoals bij SHAP?" De notebook probeert het en laat zien dat de tabel maar 14% gevuld
  is en dat de rijen niet eens hetzelfde betekenen. Goed moment om het verschil tussen de twee
  methodes echt te laten landen.
- **§4.5** — de drie LIME-runs met verschillende uitkomsten. Ideaal om te tonen waarom je een
  seed vastzet vóór je iets aan een klant laat zien.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §1 | `i` | een andere patiënt verklaren — de hele notebook past zich aan |
| §4.4 | het bereik van `range(12)` | meer of minder patiënten in de aggregatiepoging |
| §4.3 | `num_samples` | LIME stabieler (hoog) of zichtbaar ruizig (laag) |
| §4.3 | `num_features` | hoeveel regels de LIME-uitleg mag bevatten |
| §2 | `min_samples_leaf` | een simpeler of complexer bos — verandert de hele uitleg mee |
| §2.2 | `kolom` in `schud_en_meet()` | welke feature je onbruikbaar maakt |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `ModuleNotFoundError: No module named 'shap'` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Kernel heet niet "Python 3" of vindt niets | Kies rechtsboven in JupyterLab de kernel, of herstart JupyterLab vanuit de venv. |
| `shap.plots.force` toont niets | Gebruik `matplotlib=True` (zo staat het in de notebook); de JavaScript-versie werkt niet in elke viewer. |
| sklearn-waarschuwing over feature names bij LIME | LIME geeft een numpy-array door. De notebook lost dit op met de wrapper `voorspel_kans()` in §4.2. |
| Notebook staat vol met oude output | `.venv/bin/jupyter nbconvert --clear-output --inplace explainable-ai-shap-lime.ipynb` |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · shap 0.52 · lime 0.2.0.1 · scikit-learn 1.9.1 ·
pandas 3.0.5. Exacte versies: `requirements-lock.txt`.

---

## `archief/`

Bevat een eerdere versie van deze notebook met een **derde methode** (Integrated Gradients).
Niet voor de les. Zie [`archief/README.md`](archief/README.md) — die map mag weg als je hem niet nodig hebt.
