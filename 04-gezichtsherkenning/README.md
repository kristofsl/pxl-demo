# Topic 4 — Gezichtsherkenning

Van twee foto's bepalen of daar dezelfde persoon op staat, zonder zelf een gezichtsmodel te
trainen — en onderweg **transfer learning** en **contrastive learning** leren.

| | |
|---|---|
| **Notebook** | [`gezichtsherkenning.ipynb`](gezichtsherkenning.ipynb) |
| **Dataset** | LFW: 16 personen × 40 foto's, lokaal gecacht |
| **Geleend** | MTCNN (detectie) · FaceNet (embeddings) |
| **Zelf gebouwd** | 1 Keras-model: een vergelijker die twee embeddings tegen elkaar houdt |
| **Duur** | ~2 uur les · notebook draait volledig in ± 65 seconden |
| **Voorkennis** | Python, numpy, basis machine learning. Keras is een plus, geen vereiste. |

---

## Snel starten

```bash
./start.sh
```

Dat is alles. Het script maakt de virtual environment aan als die er nog niet is, installeert
wat nodig is, haalt de data op, en opent JupyterLab op **http://localhost:8884**.

Elk topic heeft zijn eigen poort, dus je kunt ze allemaal tegelijk open hebben.
Stoppen doe je met Ctrl-C in het venster waar je het script startte.

> Draai je het een tweede keer, dan slaat hij de installatie over en staat JupyterLab er
> binnen enkele seconden.

---

## Met de hand, als je liever ziet wat er gebeurt

### 1. Eenmalige installatie

Vanuit deze map, twee stappen:

```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python haal_data.py
```

De tweede stap haalt LFW op, knipt er 16 personen uit en zet die als JPEG's in
`data/gezichten/`. **Daarna heeft de notebook geen internet meer nodig.**

> De modelgewichten van MTCNN en FaceNet worden bij de eerste `import` gedownload (~90 MB) en
> daarna gecacht. Draai de notebook dus één keer vóór de les.

### 2. De notebook starten

```bash
.venv/bin/jupyter lab gezichtsherkenning.ipynb
```

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter`, anders vindt Python `tensorflow` niet.

> ⚠️ Houd de notebook **gesloten in JupyterLab** terwijl je hem ergens anders bewerkt.

### 3. Je eigen foto toevoegen

Dat is het leukste deel van de les, en het kost niets:

1. Maak een map `data/mijn-gezichten/Jouw_Naam/`
2. Zet er minstens **twee** foto's van jezelf in (JPEG of PNG, gezicht duidelijk in beeld)
3. Draai §8 opnieuw

Er hoeft niets hertraind te worden: §8 vergelijkt gewoon twee foto's. Zet je ook een map van
een huisgenoot erbij, dan kun je meteen testen of het systeem jullie uit elkaar houdt.

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 1–2 | De foto's, en **MTCNN**: bounding boxes, landmarks, en hoeveel gezichten hij mist |
| **3** | **FaceNet**: van gezicht naar 512 getallen — uitgetekend, met de embedding van drie foto's naast elkaar |
| **4** | **Contrastive learning**: waarom die 512 getallen überhaupt iets betekenen. Triplet loss, de siamese opzet, en waarom een gewone classifier het minder goed doet. Alleen uitleg — we trainen hier niets |
| 5 | De embeddingruimte verkennen: afstanden, histogram, 2D-projectie |
| 6 | **De nullijn**: een eerlijke testopstelling, en verificatie met niets anders dan een drempel op de afstand |
| **7** | **Ons eigen netwerk op het verschil** — en de waarschuwing die erbij hoort |
| 8 | Je eigen foto: is dit dezelfde persoon? |

### De rode draad

```
foto A ──┐
         ├──► [MTCNN] ──► [FaceNet] ──► vergelijken ──► zelfde persoon?
foto B ──┘    geleend      geleend      §6 vast / §7 geleerd
```

Detecteren en embedden lenen we. Het enige wat wij trainen is de **vergelijking** — en §6 laat
zien hoe ver je komt zonder zelfs dat te doen.

### Wat de notebook uitrekent

| | |
|---|---|
| MTCNN mist | **0** van de 640 gezichten |
| Gekozen drempel op de cosinusafstand (§6) | **0,53**, bepaald op de trainfoto's |
| Daarmee, op ongeziene personen | 95,5% van de gelijke paren herkend · 99,6% van de ongelijke afgewezen |

Verificatie (zijn dit twee foto's van dezelfde persoon?), AUC:

| aanpak | bekende personen | ongeziene personen | snel doorzoekbaar |
|---|---|---|---|
| FaceNet + cosinus, niets getraind | 0,972 | 0,979 | ja |
| **onze vergelijker** | **~0,992** | **~0,991** | nee |

De vergelijker wordt opnieuw getraind bij elke run, dus die twee getallen schommelen een
paar duizendsten. "Ongeziene personen" zijn vier mensen die in geen enkele trainingsstap
voorkomen. Dat de
vergelijker ook daar wint, laat zien dat hij geen gezichten uit het hoofd heeft geleerd maar
echt beter is gaan vergelijken.

---

## Didactische aanknopingspunten

- **§2** — de teller "geen gezicht gevonden: 0". Een detector die stilletjes rijen laat vallen
  is een klassieke bron van stille fouten. Laat de klas bedenken wanneer MTCNN het wél moeilijk
  krijgt (zonnebril, profiel, slecht licht) en test het met eigen foto's.
- **§3** — de tekening van de embedding: één gezicht, één streepjespatroon van 512 getallen.
  Daaronder drie foto's naast elkaar (twee keer dezelfde persoon, één keer iemand anders) zodat
  studenten zelf zien dat het patroon overeenkomt. Dat maakt "embedding" concreet.
- **§4** — de siamese tekening. Vraag de klas eerst hoeveel netwerken ze zien. Antwoord: één.
  Daarna de vervolgvraag: waarom móéten die gewichten gedeeld zijn? Omdat je anders met drie
  verschillende meetlatten meet en afstand tussen de uitvoeren nergens op slaat.
- **§4** — het berekende verlies komt op **0,000** uit. Dat is geen fout maar de kern: deze
  triplet zit al goed, dus er valt niets te leren. Opstap naar *hard negative mining*.
- **§4 slot** — "had het ook met een gewone classifier gekund?" Goede discussie over waarom je
  model moet afgerekend worden op wat je echt wil weten.
- **§6** — de drempel wordt op de **trainfoto's** gekozen, niet op de testfoto's. Vraag waarom.
  Dit is dezelfde fout die studenten in hun eigen projecten het vaakst maken.
- **§7 slot** — "De waarschuwing". De vergelijker wint op nauwkeurigheid maar kan niet in een
  vectordatabase. Laat de klas kiezen voor twee scenario's: een toegangspoort (één-op-één) en
  zoeken in een miljoen gezichten. En noteer dat LFW uit scherpe persfoto's bestaat: op
  camerabeelden zakken beide getallen fors.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §2 | `marge` in `knip_gezicht()` | strakker of ruimer uitknippen |
| §4 | `marge` in de verliesberekening | hoeveel ruimte zou het model eisen tussen de klassen |
| §6 | `DREMPEL` | strenger of soepeler: minder vals alarm, maar ook minder treffers |
| §7 | de lagen van `maak_vergelijker()` | expressiever of juist kleiner |
| §7 | het aantal paren (40 000) | hoeveel voorbeelden heeft de vergelijker nodig? |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `FileNotFoundError: Foto's niet gevonden` | Je bent `haal_data.py` vergeten. Zie stap 1. |
| `ModuleNotFoundError: No module named 'tensorflow'` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Eerste run is traag | De modelgewichten worden eenmalig gedownload. Daarna gecacht. |
| Jouw foto geeft "geen gezicht gevonden" | Gezicht te klein, te schuin of te donker. Probeer een foto waarop je gezicht minstens 100 pixels breed is. |
| Twee foto's van jezelf worden "andere persoon" | Kijk naar de afstand: zit die net boven de drempel, dan is de foto waarschijnlijk te schuin of te donker. Een drempel is altijd een afweging. |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · tensorflow 2.21 · keras 3.15.1 · mtcnn 1.0.0 ·
keras-facenet 0.3.2. Exacte versies: `requirements-lock.txt`.
