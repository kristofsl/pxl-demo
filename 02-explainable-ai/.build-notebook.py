import nbformat as nbf, sys

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ============================================================ INTRO
md(r"""
# Explainable AI: waarom zegt het model dit?

**Topic 2 — PXL · SHAP op tabulaire data**

---

## Waarom explainable AI?

Moderne modellen (random forests, gradient boosting, neurale netwerken) zijn vaak *black boxes*:
je stopt er features in, er komt een predictie uit, maar **waarom** die predictie?

In topic 1 heb je dat zelf gebouwd: een random forest van 300 bomen. Eén boom kon je nog
uittekenen en voorlezen; driehonderd kan geen mens meer overzien. En toch was dat het model
dat het beste presteerde.

Dat "waarom" is om drie redenen belangrijk:

| Reden | Voorbeeld |
|---|---|
| **Vertrouwen** | Een arts gebruikt een model alleen als ze begrijpt waarop het zich baseert. |
| **Debuggen** | Als het model vooral kijkt naar het patiëntnummer, heb je *data leakage*. |
| **Wetgeving** | AVG/GDPR art. 22 en de EU AI Act eisen uitleg bij geautomatiseerde beslissingen. |

""")

# ============================================================ 0 SETUP
md(r"""
## 0. Setup

We zetten de imports, printen de versienummers (handig als iets later niet meer werkt)
en leggen één *random seed* vast zodat de hele notebook reproduceerbaar is.
""")

code(r"""
# --- standaard data-stack ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- scikit-learn: data, modellen, evaluatie ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay,
)

# --- de hoofdrolspeler ---
import shap

# Zorgt dat matplotlib-figuren in de notebook zelf verschijnen.
%matplotlib inline
plt.rcParams["figure.dpi"] = 110          # iets scherpere plots op een beamer
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

# EEN vaste seed voor de hele notebook. Alles wat toeval gebruikt
# (train/test-split, random forest)
# krijgt deze waarde, zodat jouw output identiek is aan die van de docent.
RANDOM_STATE = 42

import sklearn
print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("shap        ", shap.__version__)
""")

# ============================================================ 1 DATA
md(r"""
## 1. De dataset

We gebruiken **Breast Cancer Wisconsin (Diagnostic)**, een klassieke tabulaire dataset die
in scikit-learn zit ingebakken — dus geen download nodig.

- **569 rijen**: elke rij is één weefselstaal (een fine needle aspirate van een borsttumor).
- **30 kolommen**: numerieke metingen van celkernen op een microscoopbeeld.
  Van 10 basiskenmerken (`radius`, `texture`, `perimeter`, `area`, `smoothness`,
  `compactness`, `concavity`, `concave points`, `symmetry`, `fractal dimension`)
  zijn telkens drie varianten berekend: `mean` (gemiddelde), `error` (standaardfout)
  en `worst` (gemiddelde van de drie "ergste" kernen).
- **Target**: `0 = malignant` (kwaadaardig), `1 = benign` (goedaardig).

> ⚠️ **Let op de codering**: in deze dataset is `1` de *goede* uitkomst. Dat is
> contra-intuïtief en een klassieke bron van fouten bij het interpreteren van uitleg.
> We houden daarom overal expliciet bij welke klasse we verklaren.
""")

code(r"""
# as_frame=True geeft ons meteen een pandas DataFrame in plaats van een numpy array.
# Dat is cruciaal voor XAI: SHAP heeft de FEATURENAMEN nodig
# om leesbare uitleg te maken.
data = load_breast_cancer(as_frame=True)

X = data.data           # DataFrame, 569 x 30
y = data.target         # Series met 0/1

class_names = list(data.target_names)   # ['malignant', 'benign'] -> index 0 en 1
feature_names = list(X.columns)

print(f"Vorm van X: {X.shape}  (rijen x features)")
print(f"Klassenamen: {class_names}  ->  0={class_names[0]}, 1={class_names[1]}")
print()
print("Verdeling van de target:")
print(y.value_counts().rename(index={0: class_names[0], 1: class_names[1]}))
""")

code(r"""
# Even kijken naar de eerste rijen en een paar beschrijvende statistieken.
# Merk op dat de features heel verschillende SCHALEN hebben
# (bv. 'mean area' rond de 650, 'mean smoothness' rond de 0.1).
# Voor een random forest maakt dat niet uit, maar het is wel belangrijk bij het
# lezen van uitleg: een grote SHAP-waarde heeft niets te maken met een grote
# featurewaarde.
display(X.head())
display(X[["mean radius", "mean texture", "mean area", "worst concave points"]].describe().T)
""")

md(r"""
### Train/test-split

We splitsen in een trainset (waarop de modellen leren) en een testset
(waarop we evalueren én verklaren). `stratify=y` houdt de klassenverhouding gelijk.
""")

code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # 25% apart houden
    random_state=RANDOM_STATE,
    stratify=y,              # zelfde klassenverhouding in train en test
)

print(f"Train: {X_train.shape[0]} rijen")
print(f"Test : {X_test.shape[0]} rijen")
""")

md(r"""
### Onze patiënt

Beide methodes gaan **dezelfde predictie** verklaren. We kiezen die patiënt hier,
één keer, zodat je hem later maar op één plaats hoeft te veranderen.
""")

code(r"""
# Testset-rij 7. Verander dit getal tijdens de les: alle uitleg hieronder
# past zich automatisch aan.
i = 7

print(f"We verklaren testset-rij i = {i} (originele index {X_test.index[i]}).")
print(f"Werkelijke diagnose: {class_names[y_test.iloc[i]]}")
display(X_test.iloc[[i]][["mean radius", "mean texture", "worst radius",
                          "worst concave points", "worst perimeter", "worst area"]].T
        .rename(columns={X_test.index[i]: "waarde"}))
""")

# ============================================================ 2 MODEL
md(r"""
## 2. Het black-box model: een random forest

Een random forest is een ensemble van honderden beslisbomen. Eén boom kun je nog uittekenen
en lezen; 300 bomen samen kan geen mens meer overzien. Precies daarom is dit een goed
voorbeeld voor XAI: het model presteert uitstekend, maar is niet direct leesbaar.

We schalen de features niet — bomen zijn invariant voor monotone transformaties
van individuele features.
""")

code(r"""
model = RandomForestClassifier(
    n_estimators=300,        # 300 bomen
    max_depth=None,          # laat bomen volledig uitgroeien
    min_samples_leaf=2,      # lichte regularisatie tegen overfitting
    n_jobs=-1,               # gebruik alle CPU-cores
    random_state=RANDOM_STATE,
)

model.fit(X_train, y_train)

y_pred  = model.predict(X_test)              # harde klasse: 0 of 1
y_proba = model.predict_proba(X_test)[:, 1]  # kans op klasse 1 (= benign/goedaardig)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.3f}")
print()
print(classification_report(y_test, y_pred, target_names=class_names))
""")

code(r"""
# Een confusion matrix maakt concreet WELKE fouten het model maakt.
# In een medische context is de rechterbovenhoek (kwaadaardig gemist!)
# veel erger dan de linkeronderhoek.
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names, cmap="Blues", colorbar=False, ax=ax
)
ax.set_title("Confusion matrix (testset)")
plt.tight_layout()
plt.show()
""")

code(r"""
# De predictie voor ONZE patiënt.
pred_label = model.predict(X_test.iloc[[i]])[0]
pred_proba = model.predict_proba(X_test.iloc[[i]])[0]

print(f"--- Patiënt {i} volgens het random forest ---")
print(f"Werkelijke diagnose : {class_names[y_test.iloc[i]]}")
print(f"Modelpredictie      : {class_names[pred_label]}")
print(f"Kans op 'benign'    : {pred_proba[1]:.4f}")
print(f"Kans op 'malignant' : {pred_proba[0]:.4f}")
print()
print("De vraag voor de rest van de notebook: WAAROM?")
""")

md(r"""
### 2.1 Eerste (naïeve) poging: de ingebouwde feature importance

`model.feature_importances_` geeft de **mean decrease in impurity** (MDI): hoeveel elke feature
gemiddeld bijdroeg aan het "zuiverder" maken van de splits in de bomen.

Dat is gratis meegeleverd, maar het heeft drie serieuze problemen:

1. **Bias richting features met veel unieke waarden** — continue features winnen bijna altijd van binaire.
2. **Berekend op de trainingsdata** — een feature die het model laat overfitten lijkt belangrijk.
3. **Alleen globaal en zonder richting** — je weet niet of een hoge waarde de predictie *omhoog* of *omlaag* duwt, en al helemaal niet wat er bij één patiënt gebeurde.

Er bestaat een eenvoudiger en eerlijker alternatief: **permutation importance**.
""")

md(r"""
### 2.2 Permutation importance: kapotmaken en kijken wat je mist

Het idee is verrassend simpel. Wil je weten hoe belangrijk een feature is?
**Maak hem onbruikbaar en kijk hoeveel slechter het model wordt.**

Concreet, voor bijvoorbeeld `mean radius`:

1. Neem de testset.
2. **Schud die ene kolom willekeurig door elkaar.** Elke patiënt krijgt nu de straal van een
   *willekeurige andere* patiënt. Alle andere kolommen blijven kloppen.
3. Laat het model opnieuw voorspellen en kijk hoeveel accuracy je verliest.

```
        origineel                      na het schudden van 'radius'
   ┌────────┬─────────┬───────┐    ┌────────┬─────────┬───────┐
   │ radius │ texture │ echt  │    │ radius │ texture │ echt  │
   ├────────┼─────────┼───────┤    ├────────┼─────────┼───────┤
   │  17.9  │  10.4   │  ✗    │    │  12.4  │  10.4   │  ✗    │  ← straal van
   │  20.6  │  17.8   │  ✗    │    │  17.9  │  17.8   │  ✗    │     iemand anders
   │  12.4  │  15.7   │  ✓    │    │  20.6  │  15.7   │  ✓    │
   └────────┴─────────┴───────┘    └────────┴─────────┴───────┘
     kolom hoort bij de patiënt      kolom is losgekoppeld van de patiënt
```

De uitkomst laat zich lezen als een gemis:

- **Zakt de score?** Dan leunde het model op die feature.
- **Verandert er niets?** Dan deed die feature niets — het model kan perfect zonder.

Herhaal dat kolom per kolom en je hebt een ranglijst. Drie redenen waarom dit beter is dan MDI:

| | Waarom het helpt |
|---|---|
| Het gebeurt op de **testdata** | Je meet wat de feature waard is voor *nieuwe* patiënten, niet hoe vaak het model hem tijdens het trainen toevallig gebruikte. |
| Het model wordt **niet hertraind** | Je meet dit model, zoals het nu is. Snel, en geen toeval van een nieuwe training erbij. |
| Het werkt op **elk** model | Random forest, neuraal netwerk, black box van een leverancier — je hebt alleen voorspellingen nodig. |

> **Waarom schudden en niet gewoon de kolom weglaten?** Omdat het model 30 kolommen verwacht.
> Door te schudden houd je de vorm én de verdeling van de data intact — alleen het *verband*
> met de patiënt is kapot. Dat is precies wat je wil meten.
>
> En omdat het schudden toeval is, herhaal je het een paar keer (`n_repeats`) en neem je het
> gemiddelde.

We meten de schade hieronder met de **AUC** in plaats van de accuracy. Reden: accuracy telt
alleen hoeveel patiënten van kant wisselen, en dat zijn er in een testset van 143 al snel nul
of één. AUC kijkt naar de voorspelde kansen zelf en is dus veel gevoeliger voor kleine schade.
""")

code(r"""
# Laten we het één keer met de hand doen, voor twee kolommen: eentje waarvan we
# vermoeden dat het model erop leunt, en eentje die er waarschijnlijk niet toe doet.
def schud_en_meet(kolom):
    X_geschud = X_test.copy()
    rng = np.random.default_rng(RANDOM_STATE)               # zelfde toeval voor elke kolom
    X_geschud[kolom] = rng.permutation(X_geschud[kolom].values)
    return roc_auc_score(y_test, model.predict_proba(X_geschud)[:, 1])

basis_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC met correcte data: {basis_auc:.4f}")
print()

for kolom in ["worst concave points", "fractal dimension error"]:
    auc = schud_en_meet(kolom)
    print(f"  {kolom:24s} geschud -> AUC {auc:.4f}   verlies {basis_auc - auc:+.4f}")

print()
print("Dat verlies IS de permutation importance van die kolom.")
print("De ene feature wordt gemist, de andere helemaal niet.")
""")

md(r"""
**Kijk goed naar de grootte van dat verlies: het is klein.** Dat is geen fout, en het is
belangrijk om te bespreken.

De reden: `worst radius`, `worst perimeter` en `worst area` meten alledrie in feite hoe *groot*
de tumor is. Schud je er één door elkaar, dan pakt het bos gewoon een van de andere twee als
vervanger. Elke feature lijkt daardoor afzonderlijk misbaar, terwijl de *groep* samen
cruciaal is.

Dit is de bekendste beperking van permutation importance, en je komt hem straks bij SHAP
opnieuw tegen: **sterk gecorreleerde features verdelen hun belang onder elkaar.**
Wil je weten hoe belangrijk "tumorgrootte" is, schud dan die kolommen samen.
""")

code(r"""
# --- (a) MDI importance: gratis, maar op de TRAINdata en zonder richting ---
mdi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

# --- (b) Permutation importance op de TESTdata: eerlijker ---
# We hussellen elke kolom 10x en meten het gemiddelde verlies aan accuracy.
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,               # 10 keer schudden per kolom, om het toeval uit te middelen
    scoring="roc_auc",          # zelfde gevoelige maat als in de handmatige demo hierboven
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
perm_s = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
mdi.head(10)[::-1].plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("MDI importance (train, ingebouwd)")
perm_s.head(10)[::-1].plot.barh(ax=axes[1], color="#DD8452")
axes[1].set_title("Permutation importance (test)")
axes[1].set_xlabel("gemiddelde daling in AUC")
plt.tight_layout()
plt.show()
""")

md(r"""
**Observatie voor de klas:** de twee ranglijsten verschillen. Sterk gecorreleerde features
(`mean radius`, `mean perimeter` en `mean area` meten in feite hetzelfde) delen hun belang
en duiken op wisselende plaatsen op.

Maar het grootste bezwaar blijft: dit zegt **niets over patiënt 7**.
Daarvoor hebben we lokale uitleg nodig.

---
""")

# ============================================================ 3 SHAP
md(r"""
## 3.1 Het idee

SHAP staat voor **SHapley Additive exPlanations** (Lundberg & Lee, 2017). Het leent een idee uit
de coöperatieve speltheorie van Lloyd Shapley.

### Het spel

Stel je voor dat de **features samen een spel spelen** en dat de "winst" de predictie is.
De vraag: *hoe verdeel je die winst eerlijk over de spelers?*

De **SHAP-waarde** van een feature is zijn gemiddelde bijdrage over **alle mogelijke volgordes**
waarin je de features aan het model kunt toevoegen. In gewone taal:

> *"Wat verandert er aan de predictie als het model deze ene feature erbij krijgt — gemiddeld
> over alle groepen features die het al zou kunnen kennen?"*

Het resultaat is per feature één getal, dat zegt **hoeveel die featurewaarde de predictie
wegduwde van het gemiddelde**, voor deze ene rij.

### Waarom die verdeling "eerlijk" heet

De manier waarop SHAP de winst verdeelt, heeft precies de eigenschappen die je intuïtief zou
eisen van een eerlijke verdeling:

- Twee features die altijd hetzelfde doen, krijgen dezelfde waarde.
- Een feature die de predictie nooit verandert, krijgt nul.
- De bijdragen samen brengen je van de gemiddelde voorspelling naar de voorspelling voor
  deze patiënt — dat is precies wat de waterfall plot in §3.4 uittekent.

Shapley bewees dat er maar **één** manier van verdelen bestaat die dat alledrie tegelijk doet.
Dat is de reden dat SHAP als de theoretisch correcte methode geldt.

### De prijs: rekenkracht

Al die groepen features doorlopen zijn er bij 30 features ruim een miljard.
Daarom bestaan er *explainers* die dit benaderen of exact-maar-snel uitrekenen:

| Explainer | Voor welk model | Snelheid |
|---|---|---|
| `TreeExplainer` | bomen, random forest, XGBoost, LightGBM | exact & snel |
| `LinearExplainer` | lineaire modellen | triviaal |
| `KernelExplainer` | **elk** model (model-agnostisch) | traag, benadering |
| `DeepExplainer` / `GradientExplainer` | neurale netwerken | middelmatig |

Wij hebben een random forest → `TreeExplainer`. 🎉

> **Misverstand om weg te nemen:** een SHAP-waarde is **geen** causaal effect en
> **geen** "hoe belangrijk is deze feature". Het is: *hoeveel duwde deze featurewaarde de
> predictie weg van het gemiddelde, voor deze ene rij.*
""")

md(r"""
## 3.2 De explainer opzetten
""")

code(r"""
# De explainer krijgt het GETRAINDE model. Voor bomen heeft hij geen
# achtergronddataset nodig: de boomstructuur bevat al alle informatie
# over hoe vaak welke paden gevolgd worden.
explainer = shap.TreeExplainer(model)

# De explainer aanroepen op data geeft een 'Explanation'-object terug.
# Dat bundelt drie dingen: de SHAP-waarden, de originele featurewaarden
# en de base value. Alle shap.plots.* functies kunnen daar meteen mee overweg.
shap_exp = explainer(X_test)

print("Type          :", type(shap_exp).__name__)
print("values.shape  :", shap_exp.values.shape)
print()
print("Lees dat als (aantal rijen, aantal features, aantal klassen).")
""")

md(r"""
### Even opletten: de derde dimensie

Voor een **binaire classifier** geeft `TreeExplainer` waarden voor *beide* klassen:
shape `(n_rijen, 30 features, 2 klassen)`. De twee lagen zijn elkaars spiegelbeeld
(ze tellen samen op tot 0), want de kansen tellen op tot 1.

We kiezen consequent **klasse 1 = `benign` (goedaardig)**.
Een **positieve** waarde betekent dan: *deze featurewaarde duwt de predictie richting "goedaardig"*.
Een **negatieve** waarde duwt richting "kwaadaardig".
""")

code(r"""
# Slice de laatste as: alle rijen, alle features, klasse-index 1.
shap_benign = shap_exp[:, :, 1]

print("Vorm na slicing :", shap_benign.values.shape)
print(f"Base value      : {shap_benign.base_values[0]:.4f}")
print()
print("De base value is de gemiddelde voorspelling van het model over de trainingsdata:")
print(f"  aandeel 'benign' in de trainset = {y_train.mean():.4f}")
print("Dat is het VERTREKPUNT van elke uitleg: zonder enige feature te kennen,")
print("zou het model deze kans voorspellen.")
""")

md(r"""
## 3.3 Globale uitleg — SHAP-waarden aggregeren

Een SHAP-waarde is lokaal, maar als je per feature het **gemiddelde van de absolute waarden**
neemt over alle testrijen, krijg je een globale importance die wél consistent is. Grote
bijdragen tellen dan mee, of ze nu naar links of naar rechts duwden.
""")

code(r"""
# De bar plot doet precies die aggregatie: gemiddelde |SHAP| per feature.
shap.plots.bar(shap_benign, max_display=12, show=False)
plt.title("Globale importance: gemiddelde |SHAP-waarde|", fontsize=11)
plt.tight_layout()
plt.show()
""")

md(r"""
**Te bespreken in de les:** bovenaan staan `worst concave points`, `worst perimeter` en
`worst area` — allemaal maten voor hoe groot en hoe grillig gevormd de tumor is. Medisch gezien
klopt dat: grote tumoren met veel concave randen zijn verdachter. Het model heeft dus iets
geleerd dat een patholoog herkent.

Dat is precies het soort *sanity check* waarvoor je XAI gebruikt. Had hier een administratief
veld bovenaan gestaan — een patiëntnummer, een labcode — dan wist je meteen dat er iets mis was
met je data.
""")

md(r"""
## 3.4 Lokale uitleg — patiënt 7 ontleed

### De waterfall plot

Dit is *de* plot om aan een niet-technisch publiek te tonen. Je leest hem van onder naar boven:

- Onderaan start je bij de **base value**: de gemiddelde voorspelling van het model, dus wat
  het zou gokken zonder ook maar één feature te kennen.
- Elke balk is één feature die de predictie verschuift (rood = omhoog duwen, blauw = omlaag).
- Bovenaan kom je uit bij de uiteindelijke voorspelde kans voor **deze** patiënt.

De features die het minst bijdragen worden onderaan samengevat als "n other features".
""")

code(r"""
# shap_benign[i] selecteert de uitleg van één rij -> een Explanation van 1 instantie.
shap.plots.waterfall(shap_benign[i], max_display=12, show=False)
plt.title(f"Waarom voorspelt het model '{class_names[pred_label]}' voor patiënt {i}?",
          fontsize=11)
plt.tight_layout()
plt.show()
""")

code(r"""
# Dezelfde informatie als een leesbare tabel: de 10 sterkste bijdragen,
# gesorteerd op absolute grootte. Dit is vaak nuttiger dan een plot
# als je de uitleg in een rapport of een applicatie moet gieten.
shap_tabel = (
    pd.DataFrame({
        "featurewaarde": X_test.iloc[i].values,
        "shap_waarde": shap_benign.values[i],
    }, index=feature_names)
    .assign(richting=lambda d: np.where(d.shap_waarde > 0, "-> benign", "-> malignant"))
    .reindex(pd.Series(np.abs(shap_benign.values[i]), index=feature_names)
             .sort_values(ascending=False).index)
    .head(10)
    .round(4)
)
display(shap_tabel)
""")

md(r"""
### De force plot: dezelfde uitleg, horizontaal

De force plot toont de bijdragen als krachten die tegen elkaar in duwen. Handig als je
meerdere patiënten naast elkaar wil tonen.
""")

code(r"""
# matplotlib=True rendert een statische figuur. (Zonder die parameter krijg je
# een interactieve JavaScript-versie; die vereist shap.initjs() en werkt niet
# in elke notebook-viewer, dus voor lesmateriaal is de statische versie veiliger.)
shap.plots.force(shap_benign[i], matplotlib=True, show=False, figsize=(20, 3))
plt.tight_layout()
plt.show()
""")

md(r"""
---

### De drie dingen die blijven hangen

**Een SHAP-waarde is geen causaal effect.** Hij zegt wat het *model* deed, niet wat de
*werkelijkheid* doet. Leert je model dat het aantal brandweerwagens een brand voorspelt, dan
toont SHAP dat netjes. De uitleg klopt; het model is verkeerd.

**Gecorreleerde features delen hun krediet.** `mean radius`, `mean perimeter` en `mean area`
meten alledrie hetzelfde. Elk krijgt ongeveer een derde van het belang, waardoor ze
afzonderlijk onbelangrijk lijken terwijl de groep cruciaal is. Dat zag je in §2.2 al gebeuren
bij permutation importance, en het geldt hier net zo goed.

**Uitleg van een slecht model is waardeloos.** Controleer eerst of het model überhaupt iets
presteert. Een perfect uitgelegde muntworp blijft een muntworp.

### Waar je dit voor gebruikt

Niet om een mooi plaatje bij je rapport te plakken, maar om drie concrete vragen te
beantwoorden:

- **Klopt dit wel?** Staat er een administratief veld bovenaan je importance-lijst — een
  patiëntnummer, een datum — dan heb je data leakage en geen model.
- **Waarom deze beslissing?** De AVG en de EU AI Act vragen bij geautomatiseerde beslissingen
  om uitleg. Een waterfall plot per geval is een antwoord dat standhoudt.
- **Waar gaat het mis?** Pak een geval waar het model ernaast zat en kijk welke features het de
  verkeerde kant op duwden. Dat is vaak leerzamer dan honderd correcte voorspellingen.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (xai)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nbf.write(nb, sys.argv[1])
print(f"Notebook geschreven: {sys.argv[1]} ({len(cells)} cellen)")
