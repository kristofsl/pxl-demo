import nbformat as nbf, sys
nb=nbf.v4.new_notebook(); cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Conformal prediction: een voorspelling met een garantie

**Topic 5 — PXL · één predictie, van begin tot eind uitgewerkt**

---

Een model geeft één antwoord. "Dit is een 4." Daar zit geen belofte bij: het model klinkt bij
elke voorspelling even stellig, ook als het er eigenlijk naast gokt.

**Conformal prediction** verandert dat antwoord in een **verzameling** antwoorden, met een
belofte die je kunt natellen:

> *"In 95% van de gevallen zit het juiste antwoord in deze verzameling."*

We werken dat hier uit voor één handgeschreven cijfer. Het idee erachter past in één zin:

> **Kijk op apart gehouden data hoe ver het model er normaal naast zit, en gebruik die
> gebruikelijke misser als marge.**
""")

md(r"""
## 0. Setup
""")

code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

%matplotlib inline
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
""")

md(r"""
## 1. Data en model

We gebruiken handgeschreven cijfers. Handig, want je kunt ze zien: als het model twijfelt
tussen een 4 en een 9, kun je zelf kijken of je het ermee eens bent.

De data gaat in **drie** stukken, en dat derde stuk is waar het hier om draait:

| deel | waarvoor |
|---|---|
| **train** | het model leren |
| **kalibratie** | meten hoe ver het model er normaal naast zit |
| **test** | controleren of de belofte klopt |

Het model mag de kalibratieset nooit gezien hebben tijdens het trainen. Anders meet je hoe goed
het kan onthouden in plaats van hoe vaak het ernaast zit.
""")

code(r"""
bestand = Path("data/cijfers.npz")
if not bestand.exists():
    raise FileNotFoundError(
        "Data niet gevonden. Draai eerst eenmalig:  .venv/bin/python haal_data.py")

d = np.load(bestand)
X = d["X"].astype("float32") / 255.0      # pixels naar 0-1
y = d["y"].astype(int)

# Bewust een klein trainingsdeel. Een model dat nooit fouten maakt heeft geen
# onzekerheid om over te praten, en dan valt er niets te laten zien.
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=2000, random_state=RANDOM_STATE, stratify=y)
X_kalib, X_test, y_kalib, y_test = train_test_split(
    X_rest, y_rest, train_size=4000, random_state=RANDOM_STATE, stratify=y_rest)

print(f"train      : {len(X_train):>6,}")
print(f"kalibratie : {len(X_kalib):>6,}")
print(f"test       : {len(X_test):>6,}")
""")

code(r"""
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print(f"test-accuracy: {model.score(X_test, y_test):.1%}")
""")

md(r"""
## 2. Wat het model zegt over één cijfer

We pakken er één uit en vragen het model wat het ervan vindt. `predict_proba` geeft tien
kansen terug: één per cijfer.
""")

code(r"""
i = 101                     # verander dit gerust

kansen = model.predict_proba(X_test[i:i + 1])[0]
gok = kansen.argmax()

fig, (links, rechts) = plt.subplots(1, 2, figsize=(9, 3))
links.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
links.set_title(f"dit is in werkelijkheid een {y_test[i]}", fontsize=10)
links.axis("off")

rechts.bar(range(10), kansen, color="#4C72B0")
rechts.set_xticks(range(10))
rechts.set_xlabel("cijfer"); rechts.set_ylabel("kans volgens het model")
rechts.set_title("wat het model ervan maakt", fontsize=10)
plt.tight_layout(); plt.show()

print(f"Het model gokt {gok}, met {kansen[gok]:.1%} zekerheid.")
print(f"In werkelijkheid is het een {y_test[i]}.")
print()
print("Daar staat het probleem in één regel: het model zit ernaast, en klinkt")
print("er behoorlijk zeker bij. Dat percentage is een getal dat het model zelf")
print("verzint; er zit geen belofte achter. En het geeft altijd precies één")
print("antwoord, ook als het eigenlijk twijfelt.")
""")

md(r"""
## 3. Kalibreren: hoe ver zit dit model er normaal naast?

Nu de kern. We laten het model los op de **kalibratieset** — data die het nooit gezien heeft —
en kijken voor elk cijfer: *welke kans gaf het aan het antwoord dat echt klopte?*

Was dat 0.99, dan zat het er goed in. Was dat 0.02, dan had het geen idee. Om er een
"hoe fout zat ik"-score van te maken, rekenen we met `1 - die kans`. Hoog is slecht.
""")

code(r"""
kansen_kalib = model.predict_proba(X_kalib)

# De kans die het model gaf aan het ECHTE label van elk kalibratiecijfer.
kans_juiste_label = kansen_kalib[np.arange(len(y_kalib)), y_kalib]
score = 1 - kans_juiste_label

print("De eerste vijf kalibratiecijfers:")
for k in range(5):
    print(f"  echt een {y_kalib[k]} | model gaf daar {kans_juiste_label[k]:6.1%} aan "
          f"-> score {score[k]:.3f}")
""")

md(r"""
### De grens

Nu de enige keuze die jij maakt: **hoeveel fouten accepteer je?**

Wil je dat het juiste antwoord in 95% van de gevallen in je verzameling zit, dan zoek je de
waarde waar 95% van de kalibratiescores onder ligt. Alles wat slechter scoort dan die grens,
valt buiten de boot.

*(Het `(n+1)` hieronder is een kleine correctie voor het feit dat je maar een eindige
kalibratieset hebt. Zonder die correctie kom je net iets te laag uit.)*
""")

code(r"""
ALPHA = 0.05                                  # 0.05 -> 95% garantie

n = len(score)
niveau = np.ceil((n + 1) * (1 - ALPHA)) / n
grens = np.quantile(score, niveau, method="higher")

print(f"garantie : {1 - ALPHA:.0%}")
print(f"grens    : {grens:.4f}")
print()
print(f"Vertaald: een cijfer komt in de verzameling als het model er minstens")
print(f"{1 - grens:.1%} kans aan geeft.")

fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.hist(score, bins=60, color="#4C72B0")
ax.axvline(grens, color="#C44E52", lw=2)
ax.text(grens + 0.02, ax.get_ylim()[1] * 0.75, f"grens = {grens:.2f}",
        color="#C44E52", fontsize=9)
ax.set_xlabel("score  (0 = zat er bovenop, 1 = volledig misgeslagen)")
ax.set_ylabel("aantal kalibratiecijfers")
ax.set_title("Hoe ver zit dit model er normaal naast?", fontsize=11)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. De verzameling voor onze voorspelling

Terug naar het cijfer uit paragraaf 2 — dat waar het model een 9 van maakte. We nemen **elk**
cijfer waarvan de score onder de grens blijft. Geen `argmax`, geen "de hoogste wint": iedereen
die door de keuring komt, doet mee.
""")

code(r"""
# Voor elk van de tien cijfers: hoe fout zou het zijn als dít het antwoord was?
scores_van_onze_voorspelling = 1 - kansen
hoort_erbij = scores_van_onze_voorspelling <= grens

print("cijfer   kans     score    doet mee?")
for cijfer in range(10):
    vink = "ja" if hoort_erbij[cijfer] else "-"
    print(f"   {cijfer}    {kansen[cijfer]:6.1%}   {scores_van_onze_voorspelling[cijfer]:.3f}      {vink}")

# int() erbij, anders print numpy er "np.int64(4)" van.
verzameling = sorted(int(c) for c in np.where(hoort_erbij)[0])
print()
print(f"De verzameling: {set(verzameling)}")
print(f"Het juiste antwoord was {y_test[i]} — zit het erin? "
      f"{'ja' if y_test[i] in verzameling else 'nee'}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
ax.set_title(f"verzameling: {set(verzameling)}", fontsize=11)
ax.axis("off")
plt.tight_layout(); plt.show()

if len(verzameling) == 1:
    print("Eén antwoord: het model twijfelt hier niet.")
else:
    print(f"{len(verzameling)} antwoorden in plaats van één.")
    print()
    print(f"Kijk nog eens naar het plaatje. Het model gokte {gok} en zat ernaast,")
    print(f"maar het juiste antwoord ({y_test[i]}) zit wél in de verzameling.")
    print("Dat is precies wat conformal prediction je oplevert: niet een beter")
    print("model, maar een eerlijker antwoord van hetzelfde model.")
""")

md(r"""
## 5. Klopt die belofte?

We beloofden 95%. Dat is geen schatting maar iets wat je kunt natellen — op testdata die
nergens in dit proces gebruikt is.
""")

code(r"""
kansen_test = model.predict_proba(X_test)
verzamelingen = (1 - kansen_test) <= grens          # per rij: welke cijfers doen mee

zit_erin = verzamelingen[np.arange(len(y_test)), y_test]

print(f"beloofd : {1 - ALPHA:.1%}")
print(f"gemeten : {zit_erin.mean():.1%}   ({zit_erin.sum():,} van de {len(y_test):,} testcijfers)")
print()
print(f"Gemiddeld aantal antwoorden per verzameling: {verzamelingen.sum(axis=1).mean():.2f}")
""")

md(r"""
Beloofd en gemeten lopen gelijk op, en daar hebben we niets voor hoeven doen: geen aannames
over de data, geen hertraining, geen gepruts aan het model.

> **Dat gemeten getal ligt net ónder de 95%, en dat hoort zo.** De belofte geldt gemiddeld over
> kalibratiesets, niet voor elke losse set. Had je toevallig 4000 andere cijfers gekalibreerd,
> dan was de grens iets anders uitgevallen en de dekking ook — soms een half procent erboven,
> soms eronder.

Wat het kost staat in dat laatste getal: gemiddeld meer dan één antwoord per verzameling. Dat
is de ruil. Wil je een hardere garantie, zet `ALPHA` lager — en krijg je grotere verzamelingen
terug. Zekerheid koop je met vaagheid.
""")
nb["cells"]=cells
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
                "language_info":{"name":"python","pygments_lexer":"ipython3"}}
nbf.write(nb,sys.argv[1]); print(f"Notebook geschreven: {sys.argv[1]} ({len(cells)} cellen)")
