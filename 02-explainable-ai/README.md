# Topic 2 — Explainable AI

Je hebt in topic 1 een random forest van 300 bomen gebouwd dat het beste presteerde — en dat
niemand kan overzien. Deze notebook haalt er per patiënt een verklaring uit met **SHAP**.

| | |
|---|---|
| **Notebook** | [`explainable-ai-shap.ipynb`](explainable-ai-shap.ipynb) |
| **Dataset** | Breast Cancer Wisconsin (569 × 30, zit in scikit-learn — geen download) |
| **Model** | Random forest (300 bomen, 95,8% accuracy op de testset) |
| **Duur** | ~2 uur les · notebook draait volledig in ± 15 seconden |
| **Voorkennis** | Topic 1, of elders geleerd wat een model, een split en een classifier zijn |

---

## Snel starten

```bash
./start.sh
```

Dat is alles. Het script maakt de virtual environment aan als die er nog niet is, installeert
wat nodig is, haalt de data op, en opent JupyterLab op **http://localhost:8882**.

Elk topic heeft zijn eigen poort, dus je kunt ze allemaal tegelijk open hebben.
Stoppen doe je met Ctrl-C in het venster waar je het script startte.

> Draai je het een tweede keer, dan slaat hij de installatie over en staat JupyterLab er
> binnen enkele seconden.

---

## Met de hand, als je liever ziet wat er gebeurt

### 1. Eenmalige installatie

```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

De dataset zit in scikit-learn ingebakken, dus er is geen downloadstap.

### 2. De notebook starten

```bash
.venv/bin/jupyter lab explainable-ai-shap.ipynb
```

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter`, anders vindt Python `shap` niet.

> ⚠️ Houd de notebook **gesloten in JupyterLab** terwijl je hem ergens anders bewerkt.

### 3. Controleren of alles werkt

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace explainable-ai-shap.ipynb
```

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 1 | De dataset en de codering van de doelvariabele |
| **2.1** | Waarom `feature_importances_` je op het verkeerde been zet |
| **2.2** | **Permutation importance** van nul opgebouwd: maak een kolom kapot en kijk wat je mist |
| **3.1** | **Het idee achter SHAP** — speltheorie, zonder formules |
| 3.2 | De explainer opzetten, en de valkuil van de derde dimensie |
| **3.3** | Globale uitleg: lokale waarden optellen tot een ranglijst |
| **3.4** | **Eén patiënt volledig ontleed** — waterfall, tabel en force plot |

### De kern

Een SHAP-waarde zegt: *hoeveel duwde deze featurewaarde de predictie weg van het gemiddelde,
voor deze ene rij.* Geen causaal effect, geen algemene belangrijkheid — een aandeel in één
beslissing.

---

## Didactische aanknopingspunten

- **§2.1** — laat zien dat MDI- en permutation-importance een *andere* top 10 geven. Goede
  opener: "welke van de twee is juist?"
- **§2.2** — permutation importance eerst met de hand: schud één kolom en kijk wat je mist. Het
  verlies is klein, ook bij een belangrijke feature, omdat de "worst radius / perimeter /
  area"-kolommen onderling vervangbaar zijn. Mooie voorzet voor het correlatieprobleem.
- **§3.3** — de globale SHAP-importance staat vol medisch plausibele features. Gebruik dat als
  sanity check: had er een patiëntnummer bovenaan gestaan, dan had je data leakage.
- **§3.4** — de waterfall plot is de beste plaat om aan een niet-technisch publiek te tonen.
  Laat iemand hem hardop voorlezen, van onder naar boven.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §1 | `i` | een andere patiënt verklaren — de hele notebook past zich aan |
| §2 | `min_samples_leaf` | een simpeler of complexer bos — verandert de hele uitleg mee |
| §2.2 | `kolom` in `schud_en_meet()` | welke feature je onbruikbaar maakt |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `ModuleNotFoundError: No module named 'shap'` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| `shap.plots.force` toont niets | Gebruik `matplotlib=True` (zo staat het in de notebook); de JavaScript-versie werkt niet in elke viewer. |
| Notebook staat vol met oude output | `.venv/bin/jupyter nbconvert --clear-output --inplace explainable-ai-shap.ipynb` |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · shap 0.52 · scikit-learn 1.9.1 · pandas 3.0.5.
Exacte versies: `requirements-lock.txt`.

---

## `archief/`

Bevat een eerdere versie met **LIME** en **Integrated Gradients** erbij. Niet voor de les.
Zie [`archief/README.md`](archief/README.md) — die map mag weg als je hem niet nodig hebt.
