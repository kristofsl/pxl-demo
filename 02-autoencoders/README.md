# Topic 2 — Autoencoders: fraude opsporen als anomalie

Een autoencoder leert hoe **normale** creditcardtransacties eruitzien. Wat hij daarna niet kan
namaken, is verdacht.

| | |
|---|---|
| **Notebook** | [`autoencoder-fraudedetectie.ipynb`](autoencoder-fraudedetectie.ipynb) |
| **Dataset** | Credit Card Fraud (60.492 transacties, 0,81% fraude) — lokaal gecacht |
| **Model** | Autoencoder in PyTorch, 29 → 20 → **8** → 20 → 29 |
| **Duur** | ~2 uur les · notebook draait volledig in ± 20 seconden |
| **Voorkennis** | Python, pandas, basis machine learning. Neurale netwerken zijn een plus, geen vereiste. |

---

## 1. Eenmalige installatie

Vanuit deze map, twee stappen:

```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python haal_data.py
```

De tweede stap haalt de dataset één keer op bij OpenML en bewaart een compacte selectie in
`data/` (17 MB). **Daarna heeft de notebook geen internet meer nodig** — handig, want wifi in
een leslokaal is geen afhankelijkheid die je wil hebben.

## 2. De notebook starten

```bash
.venv/bin/jupyter lab autoencoder-fraudedetectie.ipynb
```

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter` of met `source .venv/bin/activate`.
> Typ je gewoon `jupyter lab`, dan pakt macOS de systeem-Jupyter en vindt Python `torch` niet.

> ⚠️ Houd de notebook **gesloten in JupyterLab** terwijl je hem ergens anders bewerkt. Een open
> tab schrijft bij het opslaan zijn eigen (oude) versie over het bestand heen.

## 3. Controleren of alles werkt

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace autoencoder-fraudedetectie.ipynb
```

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 0–1 | Setup, de dataset, en waarom `Amount` een logaritme nodig heeft |
| **2** | **Wat is een autoencoder** — met een tekening van de flessenhals, en waarom je hem op *normale* data traint |
| 3 | Data klaarmaken: splitsen, schalen, en de fraude uit de trainset gooien |
| 4 | Het netwerk bouwen en trainen (PyTorch, 30 epochs, ~5 seconden) |
| **5** | **De reconstructiefout als score** — de histogramplot, ROC-AUC vs PR-AUC |
| **6** | **Een drempel kiezen** — en zien wat die keuze kost in gemiste fraude en vals alarm |
| **7** | **Was het netwerk nodig?** — vergelijking met PCA en IsolationForest |
| 8 | Waarom werd déze transactie gemarkeerd? (fout per feature) |

### De kern in één zin

We trainen het model **alleen op normale transacties** en gebruiken de labels nergens om te
leren — alleen om achteraf te evalueren en een drempel te kiezen. Daarom kan deze aanpak ook
fraudepatronen oppikken die nog nooit eerder gezien zijn.

### Wat de notebook uitrekent

| | |
|---|---|
| Fraude reconstrueert | **33× slechter** dan normaal (mediaan) |
| ROC-AUC / PR-AUC | 0,942 / **0,626** (willekeurig raden: 0,5 / 0,008) |
| Bij drempel op p99 | 121 van de 148 fraudegevallen gevonden, 191 valse alarmen |

---

## Didactische aanknopingspunten

- **§1** — laat eerst uitrekenen dat "altijd nee zeggen" 99,19% accuracy oplevert. Daarmee is
  meteen duidelijk waarom de rest van de notebook nooit over accuracy gaat.
- **§2** — de tekening van de flessenhals. Vraag de klas waarom een autoencoder *zonder*
  flessenhals nutteloos zou zijn (antwoord: dan kopieert hij gewoon de invoer).
- **§5** — het histogram met de overlappende staarten. Dit is dé plaat: er is geen drempel die
  alles goed doet.
- **§6** — de drempeltabel. Laat de klas kiezen alsof ze de fraudeafdeling runnen: 25 analisten
  of 2? Dat is geen technische vraag.
- **§7** — hier verliest de autoencoder van PCA op ROC-AUC en wint hij op PR-AUC. Uitstekend
  moment om te bespreken waarom je je maat vooraf kiest, en waarom je altijd met een simpele
  baseline begint.
- **§8** — de brug naar topic 1: van een score naar een uitleg.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §4 | de `8` in de flessenhals | smaller = grovere samenvatting; probeer 2, 4, 14 |
| §4 | `range(30)` (epochs) | ondertraind vs uitgetraind |
| §6 | `DREMPEL` | schuif langs de precision-recall-afweging |
| §8 | `gepakt` | een andere transactie ontleden |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `FileNotFoundError: Dataset niet gevonden` | Je bent `haal_data.py` vergeten. Zie stap 1. |
| `ModuleNotFoundError: No module named 'torch'` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Andere getallen dan hierboven | Controleer dat `RANDOM_STATE = 42` bovenaan staat en dat je de cellen van boven naar onder draait. |
| Training duurt lang | Normaal is ~5 seconden voor 30 epochs. Draait er iets anders op je CPU? |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · torch 2.14 · scikit-learn 1.9.1 · pandas 3.0.5.
Exacte versies: `requirements-lock.txt`.
