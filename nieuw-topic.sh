#!/usr/bin/env bash
# Zet een nieuw lestopic op volgens de conventie uit README.md.
#
#   ./nieuw-topic.sh 02 timeseries-forecasting
#
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Gebruik: $0 <nummer> <topicnaam-met-streepjes>"
    echo "Voorbeeld: $0 02 timeseries-forecasting"
    exit 1
fi

NUMMER="$1"
NAAM="$2"
MAP="${NUMMER}-${NAAM}"
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOEL="${HIER}/${MAP}"

if [ -e "$DOEL" ]; then
    echo "Bestaat al: $DOEL" >&2
    exit 1
fi

echo "==> Map aanmaken: $MAP"
mkdir -p "$DOEL"

cat > "$DOEL/requirements.txt" <<REQ
# Topic ${NUMMER} — ${NAAM}
# Installeren:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

numpy>=1.26
pandas>=2.1
matplotlib>=3.8
scikit-learn>=1.4

jupyterlab>=4.1
ipywidgets>=8.1

# Voeg hier de bibliotheken van dit topic toe.
REQ

cat > "$DOEL/README.md" <<DOC
# Topic ${NUMMER} — ${NAAM}

Eén zin over wat dit topic behandelt.

| | |
|---|---|
| **Notebook** | [\`${NAAM}.ipynb\`](${NAAM}.ipynb) |
| **Dataset** | … |
| **Duur** | … |
| **Voorkennis** | … |

## 1. Eenmalige installatie

Vanuit deze map:

\`\`\`bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
\`\`\`

## 2. De notebook starten

\`\`\`bash
.venv/bin/jupyter lab ${NAAM}.ipynb
\`\`\`

> ⚠️ Start Jupyter altijd via \`.venv/bin/jupyter\` (of met \`source .venv/bin/activate\`).
> Anders gebruik je de systeem-Jupyter en vindt Python de pakketten niet.

## 3. Controleren of alles werkt

\`\`\`bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace ${NAAM}.ipynb
\`\`\`

## Inhoud van de notebook

| § | Onderwerp |
|---|---|
| 0 | Setup |
| 1 | … |

## Problemen oplossen

| Symptoom | Oorzaak en oplossing |
|---|---|
| \`ModuleNotFoundError\` | Je draait de systeem-Jupyter. Start via \`.venv/bin/jupyter lab\`. |
DOC

echo "==> Virtual environment bouwen (dit duurt even)"
python3 -m venv "$DOEL/.venv"
"$DOEL/.venv/bin/pip" install --upgrade pip --quiet
"$DOEL/.venv/bin/pip" install -r "$DOEL/requirements.txt" --quiet

echo "==> Lege notebook aanmaken"
"$DOEL/.venv/bin/python" - "$DOEL/${NAAM}.ipynb" "$NUMMER" "$NAAM" <<'PYEOF'
import sys, nbformat as nbf
pad, nummer, naam = sys.argv[1], sys.argv[2], sys.argv[3]
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        f"# Topic {nummer} — {naam}\n\n"
        "**PXL · lesnotebook**\n\n---\n\n"
        "## Waarover gaat dit topic?\n\n"
        "…\n\n## Inhoud\n\n0. Setup\n1. …"
    ),
    nbf.v4.new_markdown_cell("## 0. Setup"),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "%matplotlib inline\n"
        "plt.rcParams[\"figure.dpi\"] = 110\n\n"
        "# Eén vaste seed voor de hele notebook, zodat de output van de student\n"
        "# gelijk is aan die op het scherm.\n"
        "RANDOM_STATE = 42\n\n"
        "print(\"numpy \", np.__version__)\n"
        "print(\"pandas\", pd.__version__)"
    ),
    nbf.v4.new_markdown_cell("## Oefeningen\n\n1. …"),
    nbf.v4.new_markdown_cell("## Verder lezen\n\n- …"),
]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nbf.write(nb, pad)
PYEOF

echo
echo "✅ Klaar. Volgende stappen:"
echo "   cd ${MAP}"
echo "   .venv/bin/jupyter lab ${NAAM}.ipynb"
echo
echo "   Vergeet ${MAP} niet toe te voegen aan de topictabel in ../README.md"
