# Topic 5 — Conformal prediction

Een model geeft één antwoord. Conformal prediction geeft er een **verzameling** bij, met een
garantie die je kunt natellen — en daarmee een werkstroom: wat gaat automatisch, en wat gaat
naar een mens?

| | |
|---|---|
| **Notebook** | [`conformal-prediction.ipynb`](conformal-prediction.ipynb) |
| **Dataset** | MNIST, 20.000 handgeschreven cijfers, lokaal gecacht |
| **Model** | Klein neuraal netje (sklearn), bewust maar 90,8% accuraat |
| **Duur** | ~2 uur les · notebook draait volledig in ± 10 seconden |
| **Voorkennis** | Python, basis machine learning. **Geen wiskunde nodig.** |

---

## Snel starten

```bash
./start.sh
```

Dat is alles. Het script maakt de virtual environment aan als die er nog niet is, installeert
wat nodig is, haalt de data op, en opent JupyterLab op **http://localhost:8885**.

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

```bash
.venv/bin/python haal_data.py
```

Daarna heeft de notebook geen internet meer nodig.

### 2. De notebook starten

```bash
.venv/bin/jupyter lab conformal-prediction.ipynb
```

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter`, anders vindt Python de pakketten niet.

> ⚠️ Houd de notebook **gesloten in JupyterLab** terwijl je hem ergens anders bewerkt.

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 1 | De data, en de **drie**-weg-split: train · kalibratie · test |
| 2 | Waarom "97% zeker" van een softmax geen garantie is |
| **3** | **Conformal in vier stappen** — meten, sorteren, grens, toepassen (met tekening) |
| **4** | **Klopt de garantie?** Beloofd versus gemeten, bij vier niveaus — en waarom 94,8% géén mislukking is |
| **5** | **De werkstroom**: wat automatiseren we, en wat gaat naar een mens? |
| 6 | Waar twijfelt het model over? De verwarringsparen, in beeld |

### Het idee in één zin

> Kijk op je kalibratiedata hoe ver het model er normaal naast zit, en gebruik die
> gebruikelijke misser als marge voor nieuwe voorspellingen.

Geen formules, geen aannames over je data, geen hertraining. Het werkt op elk model — ook op
een black box van een leverancier — en het kost ongeveer vijf regels code.

### Wat de notebook uitrekent

| | |
|---|---|
| Beloofde dekking | 95,0% |
| Gemeten dekking op 14.000 testcijfers | **94,8%** |
| Automatisch af te handelen (één label) | **86,9%**, waarvan 95,7% correct |
| Naar een mens (twee of meer labels) | 13,1% |
| Van alle modelfouten afgevangen | **59%** |

De verwarringsparen die het model zelf aanwijst: `{4,9}`, `{7,9}`, `{3,5}`, `{2,8}` — precies
de cijfers die een mens ook door elkaar haalt.

---

## Didactische aanknopingspunten

- **§1** — de drie-weg-split. Vraag de klas waaróm de kalibratieset apart moet blijven. Dat het
  model die nooit gezien heeft, is de hele basis van de garantie.
- **§2** — de tabel met "model zegt X%, klopt in Y%". Dit model valt mee, en dat is juist het
  punt: ook een redelijk gekalibreerd model geeft je geen *belofte*, en het roept altijd precies
  één antwoord.
- **§4** — beloofd naast gemeten, bij vier niveaus. Dit is het moment waarop conformal van
  "aardig idee" naar "dit werkt gewoon" gaat.
- **§4 slot** — de vraag die altijd komt: "94,8% is toch minder dan 95%?" De notebook meet het
  met veertig verschillende kalibratiesets: het gemiddelde landt op 94,99% en ruim de helft van
  de runs zit onder het doel. Dát is wat *marginale* dekking betekent, en het is de meest
  gemaakte denkfout bij conformal prediction.
- **§5** — de werkstroomtabel. Laat de klas een rij kiezen alsof zij de afdeling runnen. Wat
  kost een fout, en wat kost een mens die meekijkt? Dat is geen technische vraag.
- **§6** — de plaatjes van de twijfelgevallen. Laat studenten zelf kijken of ze het met de
  verzameling eens zijn. Meestal wel, en dat maakt indruk.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §3 | `ALPHA` | 0.20 tot 0.01: de hele afweging tussen garantie en vaagheid |
| §1 | `train_size=2000` | een beter model → kleinere verzamelingen |
| §1 | de grootte van de kalibratieset | hoe stabiel is de grens? |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `FileNotFoundError: Data niet gevonden` | Je bent `haal_data.py` vergeten. Zie stap 1. |
| `ModuleNotFoundError` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Dekking wijkt een paar tienden af | Normaal, en §4 meet het na: gemiddeld over 40 kalibratiesets komt het op 94,99% uit. |
| `ValueError: te weinig kalibratiepunten` | Geen bug: met n punten kun je geen garantie van meer dan ongeveer 1 − 1/n geven. Kies een lagere garantie of een grotere kalibratieset. |
| Lege verzamelingen | Geen fout: geen enkel label komt door de keuring. Zie de uitleg in §4. |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · scikit-learn 1.9.1 · pandas 3.0.5.
Exacte versies: `requirements-lock.txt`.
