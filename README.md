# PXL — lesnotebooks

Lesmateriaal in de vorm van Jupyter-notebooks. **Elk topic staat op zichzelf**: eigen map,
eigen virtual environment, eigen documentatie. Zo kan één topic nooit de dependencies van een
ander breken, en kan een student één map downloaden en meteen aan de slag.

## Topics

| # | Map | Onderwerp | Kernbibliotheken |
|---|---|---|---|
| 1 | [`01-explainable-ai/`](01-explainable-ai/) | Een predictie verklaren met **SHAP** en **LIME** op tabulaire data | shap · lime · scikit-learn |
| 2 | [`02-autoencoders/`](02-autoencoders/) | **Anomaliedetectie**: fraude opsporen met een autoencoder, zonder ooit een fraudelabel te leren | torch · scikit-learn |
| 3 | [`03-gezichtsherkenning/`](03-gezichtsherkenning/) | **Transfer learning** en **contrastive learning**: gezichtsherkenning op FaceNet-embeddings, met eigen Keras-modellen | keras · tensorflow · mtcnn |

## Snel starten

```bash
cd 01-explainable-ai
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

Elke topicmap heeft een eigen `README.md` met de precieze instructies.

> ⚠️ Start Jupyter **altijd** via `.venv/bin/jupyter` (of met de venv geactiveerd).
> Typ je gewoon `jupyter lab`, dan gebruik je de systeem-Jupyter en vindt Python de
> geïnstalleerde pakketten niet.

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
