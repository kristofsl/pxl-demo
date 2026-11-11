# PXL — lesnotebooks

Lesmateriaal in de vorm van Jupyter-notebooks. **Elk topic staat op zichzelf**: eigen map,
eigen virtual environment, eigen documentatie. Zo kan één topic nooit de dependencies van een
ander breken, en kan een student één map downloaden en meteen aan de slag.

## Topics

| # | Map | Onderwerp | Kernbibliotheken |
|---|---|---|---|
| 1 | [`01-eerste-model/`](01-eerste-model/) | **Van data naar model**: verkennen, categorische kolommen omzetten, splitsen, een beslisboom en een random forest — en in deel 2 een neuraal netwerk op dezelfde data | scikit-learn · keras |
| 2 | [`02-explainable-ai/`](02-explainable-ai/) | Een predictie verklaren met **SHAP**: waarom zegt het model dit over déze patiënt? | shap · scikit-learn |
| 3 | [`03-autoencoders/`](03-autoencoders/) | **Anomaliedetectie**: fraude opsporen met een autoencoder, zonder ooit een fraudelabel te leren | torch · scikit-learn |
| 4 | [`04-gezichtsherkenning/`](04-gezichtsherkenning/) | **Transfer learning** en **contrastive learning**: gezichtsherkenning op FaceNet-embeddings, met eigen Keras-modellen | keras · tensorflow · mtcnn |
| 5 | [`05-conformal-predictions/`](05-conformal-predictions/) | **Conformal prediction**: voorspellingen met een garantie die je kunt natellen, en een werkstroom mens-versus-machine | scikit-learn |

## Snel starten

Elk topic heeft een `start.sh` die alles doet: virtual environment aanmaken, pakketten
installeren, data ophalen en JupyterLab openen.

```bash
cd 01-eerste-model && ./start.sh
```

**Elk topic draait op zijn eigen poort**, dus je kunt ze allemaal tegelijk open hebben:

| topic | poort | |
|---|---|---|
| 1 — Van data naar model | 8881 | http://localhost:8881 |
| 2 — Explainable AI | 8882 | http://localhost:8882 |
| 3 — Autoencoders | 8883 | http://localhost:8883 |
| 4 — Gezichtsherkenning | 8884 | http://localhost:8884 |
| 5 — Conformal prediction | 8885 | http://localhost:8885 |

Stoppen doe je met Ctrl-C in het venster waar je het script startte. De eerste keer duurt het
installeren even; daarna staat JupyterLab er binnen enkele seconden.

Liever met de hand? Elke topicmap heeft een eigen `README.md` met de losse stappen.

> ⚠️ Start je Jupyter zelf, doe dat dan **altijd** via `.venv/bin/jupyter`. Typ je gewoon
> `jupyter lab`, dan gebruik je de systeem-Jupyter en vindt Python de pakketten niet.

## De conventie per topic

```
NN-topicnaam/
├── README.md                 # wat het topic behandelt + hoe je de notebook start
├── requirements.txt          # losjes gepind (>=), voor een verse installatie
├── requirements-lock.txt     # exacte versies (pip freeze), voor reproduceerbaarheid
├── <topicnaam>.ipynb         # de notebook zelf, met uitleg in markdown en comments in de code
└── .venv/                    # virtual environment, niet in git
```

Afspraken die het lesmateriaal bruikbaar houden:

- **Eén notebook per topic**, met markdowncellen als cursustekst en comments in de code.
  De notebook moet leesbaar zijn zonder dat de docent erbij staat.
- **Alles moet van boven naar onder draaien.** Controleer met
  `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`.
- **Eén vaste `RANDOM_STATE`** bovenaan, zodat de output van de student gelijk is aan die op het scherm.
- **Geen downloads tijdens de les** — gebruik ingebouwde of meegeleverde datasets.
  Wifi in een leslokaal is geen afhankelijkheid die je wil hebben.
- **Oefeningen achteraan**, gegroepeerd van opwarmer naar uitdaging.
- **Valkuilen expliciet benoemen.** Wat misgaat is didactisch waardevoller dan wat werkt.

## Een nieuw topic beginnen

```bash
./nieuw-topic.sh 02 timeseries-forecasting
```

Dat maakt de map, de venv, een `requirements.txt`, een README-sjabloon en een lege notebook
volgens bovenstaande structuur.
