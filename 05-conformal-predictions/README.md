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

Eén voorspelling, van begin tot eind uitgewerkt. Niet meer en niet minder.

| § | Onderwerp |
|---|---|
| 1 | Data en model, met de **drie**-weg-split: train · kalibratie · test |
| 2 | Wat het model zegt over één cijfer — en waarom dat percentage geen belofte is |
| **3** | **Kalibreren**: hoe ver zit het model er normaal naast, en waar leg je de grens? |
| **4** | **De verzameling** voor die ene voorspelling, cijfer voor cijfer opgebouwd |
| 5 | Klopt de belofte? Beloofd naast gemeten op 14.000 testcijfers |

### Het voorbeeld

Het gekozen cijfer is met opzet een lastig geval:

| | |
|---|---|
| In werkelijkheid | een **4** |
| Het model gokt | een **9**, met 80,2% zekerheid |
| De verzameling | **{4, 9}** — het juiste antwoord zit erin |

Daarmee staat de hele les in één voorbeeld: het model zit ernaast en klinkt er zeker bij, maar
de verzameling vangt het op. Conformal prediction geeft je geen beter model, maar een eerlijker
antwoord van hetzelfde model.

| Beloofd | 95,0% |
|---|---|
| Gemeten op 14.000 testcijfers | **94,8%** |
| Gemiddeld aantal antwoorden per verzameling | 1,15 |

### Het idee in één zin

> Kijk op apart gehouden data hoe ver het model er normaal naast zit, en gebruik die
> gebruikelijke misser als marge.

Geen formules, geen aannames over je data, geen hertraining. Het werkt op elk model — ook op
een black box van een leverancier.

---

## Didactische aanknopingspunten

- **§1** — vraag waaróm de kalibratieset apart moet blijven. Dat het model die nooit gezien
  heeft, is de hele basis van de garantie.
- **§2** — het model gokt fout én zelfverzekerd. Goed moment om te vragen wat die 80% nu
  eigenlijk waard is.
- **§4** — de tabel met tien regels, waar je cijfer voor cijfer ziet wie de keuring haalt. Laat
  de klas eerst zelf raden welke twee overblijven.
- **§5** — beloofd naast gemeten. Dit is het moment waarop conformal van "aardig idee" naar
  "dit werkt gewoon" gaat.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §2 | `i` | een ander cijfer uitwerken; 112, 125 en 144 zijn ook twijfelgevallen |
| §3 | `ALPHA` | 0.20 tot 0.01: de hele afweging tussen garantie en vaagheid |
| §1 | `train_size=2000` | een beter model → kleinere verzamelingen |

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
