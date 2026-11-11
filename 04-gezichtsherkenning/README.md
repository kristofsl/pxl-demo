# Topic 4 — Gezichtsherkenning

Een werkend gezichtsherkenningssysteem bouwen zonder zelf een gezichtsmodel te trainen —
en onderweg **transfer learning** en **contrastive learning** leren.

| | |
|---|---|
| **Notebook** | [`gezichtsherkenning.ipynb`](gezichtsherkenning.ipynb) |
| **Dataset** | LFW: 16 personen × 40 foto's, lokaal gecacht |
| **Geleend** | MTCNN (detectie) · FaceNet (embeddings) |
| **Zelf gebouwd** | 3 Keras-modellen: classifier, triplet-hoofd, pairwise vergelijker |
| **Duur** | ~3 uur les · notebook draait volledig in ± 90 seconden |
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
2. Sleep er een stuk of vijf foto's in (JPEG of PNG, gezicht duidelijk in beeld)
3. Draai §8 opnieuw

Zonder eigen foto's werkt de notebook ook — dan doet §8 alsof een testfoto je upload is.
Wil je dat het model je écht herkent, zet je map dan in `data/gezichten/` en draai alles
opnieuw: je wordt dan gewoon persoon 17.

---

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 1–2 | De foto's, en **MTCNN**: bounding boxes, landmarks, en hoeveel gezichten hij mist |
| **3** | **FaceNet**: van gezicht naar 512 getallen — uitgetekend, met de embedding van drie foto's naast elkaar |
| 4 | De embeddingruimte verkennen: afstanden, histogram, 2D-projectie |
| **5** | **Ons eigen classifiermodel** in Keras — plus: hoeveel foto's heb je nodig? |
| **6** | **Contrastive learning**: triplet loss, zelf geschreven — mét een tekening van de siamese architectuur |
| **7** | **De pairwise vergelijker**: laat het model de meetlat zélf leren — ook uitgetekend |
| 8 | Je eigen foto door de hele pijplijn |

### De rode draad

```
foto ──► [MTCNN] ──► [FaceNet] ──► [ons Keras-model] ──► naam
         geleend      geleend       zelf geschreven
```

Stap 1 en 2 lenen we. Stap 3 is van ons, en dat is precies het stuk dat het voor *jouw*
gezichten laat werken.

### Wat de notebook uitrekent

| | |
|---|---|
| MTCNN mist | **0** van de 640 gezichten |
| Classifier met 1 foto per persoon | ruim **90%** accuracy · met 2 foto's al 98% |
| Classifier met 70% van de foto's | **97,9%** op 192 testgezichten |

Verificatie (zijn dit twee foto's van dezelfde persoon?), AUC:

| aanpak | bekende personen | ongeziene personen | snel doorzoekbaar |
|---|---|---|---|
| FaceNet + cosinus, niets getraind | 0,972 | 0,979 | ja |
| ons triplet-hoofd (64 dim) | 0,987 | 0,977 | ja |
| **onze pairwise vergelijker** | **0,992** | **0,991** | nee |

---

## Didactische aanknopingspunten

- **§2** — de teller "geen gezicht gevonden: 0". Een detector die stilletjes rijen laat vallen
  is een klassieke bron van stille fouten. Laat de klas bedenken wanneer MTCNN het wél moeilijk
  krijgt (zonnebril, profiel, slecht licht) en test het met eigen foto's.
- **§5** — de few-shot-curve. Eén foto per persoon en je zit boven de 90%. Vraag de klas te
  schatten hoeveel foto's een netwerk vanaf nul nodig zou hebben.
- **§3** — de tekening van de embedding: één gezicht, één streepjespatroon van 512 getallen.
  Daaronder drie foto's naast elkaar (twee keer dezelfde persoon, één keer iemand anders) zodat
  studenten zelf zien dat het patroon overeenkomt. Dat maakt "embedding" concreet.
- **§6** — de architectuurtekening. Grijs is bevroren (FaceNet), oranje leert (ons hoofd), en
  de streepjeslijn eromheen zegt: dit is **één** netwerk, drie keer gebruikt. Vraag de klas
  waarom die gewichten gedeeld moeten zijn. Antwoord: anders meet je met drie meetlatten.
- **§6** — de tekening van triplet loss (trekken en duwen) en de verliesfunctie die we zelf
  schrijven. Merk op dat het verlies naar nul zakt: de meeste triplets zijn te makkelijk.
  Dat is de opstap naar *hard negative mining*.
- **§6 slot** — het triplet-hoofd wint bij bekende personen maar **verliest licht** bij
  ongeziene. Specialiseren kost algemeenheid; dat is geen bug maar de afweging zelf.
- **§7** — de vergelijkingstabel. De pairwise MLP wint op nauwkeurigheid, maar kan niet in een
  vectordatabase. Laat de klas kiezen voor twee scenario's: een toegangspoort (één-op-één) en
  zoeken in een miljoen gezichten.

### Knoppen om live mee te spelen

| Waar | Variabele | Effect |
|---|---|---|
| §2 | `marge` in `knip_gezicht()` | strakker of ruimer uitknippen |
| §5 | de lijst `[1, 2, 3, 5, 10, 20]` | few-shot-curve fijner maken |
| §6 | `marge` in `triplet_verlies()` | hoeveel ruimte eist het model tussen de klassen |
| §6 | `dim_uit` van het projectiehoofd | 32, 64, 128 — hoever kun je comprimeren? |
| §7 | de lagen van `maak_vergelijker()` | expressiever of juist kleiner |
| §8 | `drempel` in `herken_met_twijfel()` | wanneer zegt het systeem "onbekend"? |

---

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| `FileNotFoundError: Foto's niet gevonden` | Je bent `haal_data.py` vergeten. Zie stap 1. |
| `ModuleNotFoundError: No module named 'tensorflow'` | Je draait de systeem-Jupyter. Start via `.venv/bin/jupyter lab`. |
| Eerste run is traag | De modelgewichten worden eenmalig gedownload. Daarna gecacht. |
| Rommelige `lz4`-meldingen bij het afsluiten | Onschuldig; komt uit het inladen van de FaceNet-gewichten. |
| Jouw foto geeft "geen gezicht gevonden" | Gezicht te klein, te schuin of te donker. Probeer een foto waarop je gezicht minstens 100 pixels breed is. |
| Jouw foto wordt als iemand anders herkend | Logisch: de classifier kent maar 16 personen en kiest altijd één daarvan. §8 laat zien hoe je dat oplost met een drempel. |

## Getest met

Python 3.13.5 · macOS (Apple Silicon) · tensorflow 2.21 · keras 3.15.1 · mtcnn 1.0.0 ·
keras-facenet 0.3.2. Exacte versies: `requirements-lock.txt`.
