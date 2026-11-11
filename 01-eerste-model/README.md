# Topic 1 — Van data naar model

Een compleet machine learning-project van begin tot eind, in twee delen: eerst klassieke
modellen, dan een neuraal netwerk op exact dezelfde data — en de eerlijke vergelijking.

| | |
|---|---|
| **Notebook** | [`eerste-model.ipynb`](eerste-model.ipynb) |
| **Dataset** | Adult / Census Income: 48.842 personen, gemengd numeriek en categorisch |
| **De vraag** | Verdient deze persoon meer dan 50.000 dollar per jaar? |
| **Duur** | ~4 uur les, of twee sessies van twee uur · notebook draait in ± 55 seconden |
| **Voorkennis** | Python en een beetje pandas. **Verder niets.** Dit is het startpunt van de cursus. |

---

## Snel starten

```bash
./start.sh
```

Dat is alles. Het script maakt de virtual environment aan als die er nog niet is, installeert
wat nodig is, haalt de data op, en opent JupyterLab op **http://localhost:8881**.

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

> De installatie trekt TensorFlow binnen (enkele honderden megabytes) omdat deel 2 daarop
> draait. Doe dit vóór de les, niet tijdens.

### 2. De notebook starten

```bash
.venv/bin/jupyter lab eerste-model.ipynb
```

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter`, anders vindt Python de pakketten niet.

> ⚠️ Houd de notebook **gesloten in JupyterLab** terwijl je hem ergens anders bewerkt.

---

## Inhoud

### Deel 1 — klassieke modellen

| § | Onderwerp |
|---|---|
| 1–2 | De data, en **kijken voordat je modelleert**: basislijn, ontbrekende waarden, grafieken |
| **3** | **Van tekst naar getallen** — one-hot versus volgnummers, en waarom dat verschil uitmaakt |
| **4** | **Splitsen** — train, validatie, test, en waarom je er drie nodig hebt |
| **5** | **Een beslisboom** die je uittekent en voorleest, plus twee evolutiegrafieken |
| 6 | Logistische regressie ernaast — en wat een verkeerde encodering kost |
| **7** | **Random forest**: dezelfde overfittende bomen, maar dan gemiddeld |
| 8 | Beoordelen: confusion matrix, precisie en recall |

### Deel 2 — een neuraal netwerk

| § | Onderwerp |
|---|---|
| 9 | Wat een neuraal netwerk is, uitgetekend — en waarom schalen hier verplicht is |
| 10 | Bouwen in Keras: lagen, verliesfunctie, optimizer |
| **11** | **Trainen en de evolutie per ronde volgen** — overfitting zie je gebeuren |
| 12 | Early stopping |
| **13** | **De eindvergelijking** |

### De drie evolutiegrafieken

Op drie plekken zie je train versus validatie naast elkaar lopen, en elke keer leert dat je
iets anders:

| waar | de as | wat je leert |
|---|---|---|
| §5 | hoe diep de boom mag worden | een onbeperkte boom haalt 97,8% op train en 82,3% op validatie |
| §5 | hoeveel trainingsdata het model krijgt | bij 500 rijen gaapt er 15 punten verschil; bij 29.000 nog één |
| §11 | hoeveel rondes het netwerk traint | de trainlijn daalt terwijl de validatielijn omhoog kruipt |

### Wat de notebook uitrekent

| model | op de testset |
|---|---|
| altijd `<=50K` gokken (basislijn) | 76,07% |
| netwerk, 60 rondes doorgetraind | 84,86% |
| netwerk met early stopping | 85,39% |
| één beslisboom | 86,20% |
| **random forest** | **86,70%** |

**Daar zit de clou van de notebook.** Het neurale netwerk — vierduizend parameters, verplichte
schaling, zestig trainingsrondes, een stopmechanisme — verslaat geen van beide boommodellen.
Op tabulaire data is dat eerder regel dan uitzondering, en het is de reden waarom je altijd met
het eenvoudige model begint.

---

## Didactische aanknopingspunten

- **§2.1** — laat de klas eerst uitrekenen dat "altijd nee zeggen" al 76,1% oplevert. Zonder die
  basislijn klinkt 85% indrukwekkend.
- **§3** — de val van de volgnummers. Vraag waarom "Sales = 1, Tech-support = 2" problematisch is
  vóór je het uitlegt.
- **§4** — waarom drie stapels en niet twee. Het argument "ik heb op de testset zitten kiezen" is
  overtuigender dan de regel zelf.
- **§5** — de uitgetekende boom. Laat iemand hem hardop voorlezen; dan landt wat een model is.
- **§5** — de overfit-curve. Dit is het belangrijkste plaatje van deel 1.
- **§6** — meten dat een verkeerde encodering de regressie 5 punten kost en de boom maar 1,6.
- **§7** — driehonderd overfittende bomen die samen beter zijn dan één. Goed moment voor de
  jury-vergelijking.
- **§11** — de trainlijn die daalt terwijl de validatielijn stijgt. Dit is het belangrijkste
  plaatje van deel 2.
- **§13** — de eindtabel. Laat de klas raden waar het netwerk belandt vóór je hem toont.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §5 | de lijst met dieptes | de overfit-curve fijner of grover |
| §7 | `n_estimators` | hoeveel bomen je middelt |
| §10 | het aantal neuronen per laag | groter netwerk, meer overfitting |
| §11 | `epochs` en `batch_size` | hoe snel en hoe grillig het netwerk leert |
| §12 | `patience` | hoe lang early stopping geduld heeft |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `FileNotFoundError: Data niet gevonden` | Je bent `haal_data.py` vergeten. Zie stap 1. |
| `ModuleNotFoundError` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Deel 2 traint heel traag | Normaal is ongeveer 40 seconden voor 60 rondes op een laptop zonder GPU. |
| Andere getallen dan hierboven | Controleer dat `RANDOM_STATE = 42` bovenaan staat en dat je van boven naar onder draait. |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · scikit-learn 1.9.1 · pandas 3.0.5 · keras 3.15.1 ·
tensorflow 2.21. Exacte versies: `requirements-lock.txt`.
