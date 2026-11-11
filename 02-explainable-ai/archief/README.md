# Archief

Niet voor de les — dit is een eerdere versie die hier bewaard staat.

## `explainable-ai-shap-lime-ig.ipynb`

De notebook uit deze map, maar met **LIME** en een derde methode erbij:
**Integrated Gradients** (§5), inclusief een uitgetekende uitleg van het saturatieprobleem,
het pad door de featureruimte, een implementatie van IG in 10 regels `torch.autograd`,
en de brug naar Expected Gradients (`shap.GradientExplainer`).

Omdat IG gradiënten nodig heeft, traint die versie naast het random forest ook een klein
PyTorch-netwerk op dezelfde dataset.

### Draaien

Er zijn twee extra pakketten nodig die **niet** in `../requirements.txt` staan:

```bash
../.venv/bin/pip install lime torch captum
../.venv/bin/jupyter lab explainable-ai-shap-lime-ig.ipynb
```

Getest met torch 2.14.0 en captum 0.9.0. De opgeslagen output in de notebook komt van een
volledige run, dus je kunt hem ook gewoon lezen zonder iets te installeren.

Heb je dit niet meer nodig? Dan mag deze hele map weg.
