#!/usr/bin/env bash
#
# Start dit topic. Doet alles wat nodig is en opent JupyterLab op een eigen poort,
# zodat je alle topics tegelijk kunt laten draaien.
#
#   ./start.sh
#
# Stoppen: Ctrl-C in dit venster.
#
set -euo pipefail

# Altijd werken vanuit de map waar dit script staat, ook als je het van elders aanroept.
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HIER"

POORT=8882
NOTEBOOK="explainable-ai-shap.ipynb"
TOPIC="Topic 2 — Explainable AI met SHAP"

echo "=============================================================="
echo " $TOPIC"
echo "=============================================================="

# --- 0. is er een bruikbare python? ---
if ! command -v python3 >/dev/null 2>&1; then
    echo "FOUT: python3 niet gevonden. Installeer Python 3.10 of nieuwer." >&2
    exit 1
fi

# --- 1. een kapotte venv opruimen -----------------------------------------
# Een virtual environment bevat absolute paden. Is de map hernoemd of verplaatst,
# dan start jupyter niet meer. Dan maken we hem gewoon opnieuw.
if [ -d .venv ] && ! .venv/bin/jupyter --version >/dev/null 2>&1; then
    echo "--> de virtual environment werkt niet meer (map verplaatst?), opnieuw aanmaken"
    rm -rf .venv
fi

# --- 2. virtual environment ------------------------------------------------
if [ ! -d .venv ]; then
    echo "--> virtual environment aanmaken (eenmalig)"
    python3 -m venv .venv
    .venv/bin/pip install --quiet --disable-pip-version-check --upgrade pip
fi

# --- 3. pakketten ----------------------------------------------------------
# We onthouden een vingerafdruk van requirements.txt. Ongewijzigd? Dan slaan we
# de installatie over en start je meteen.
STEMPEL=".venv/.requirements-vingerafdruk"
HUIDIG="$(shasum requirements.txt | cut -d' ' -f1)"
if [ ! -f "$STEMPEL" ] || [ "$(cat "$STEMPEL")" != "$HUIDIG" ]; then
    echo "--> pakketten installeren (de eerste keer duurt dit even)"
    .venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt
    echo "$HUIDIG" > "$STEMPEL"
fi

# --- 4. data ---------------------------------------------------------------
if [ -f haal_data.py ]; then
    .venv/bin/python haal_data.py
fi

# --- 5. starten ------------------------------------------------------------
echo
echo "--> JupyterLab opent op http://localhost:$POORT"
echo "    Stoppen doe je met Ctrl-C in dit venster."
echo

# --port-retries=0: als de poort bezet is, stoppen met een foutmelding in plaats
# van stilletjes een andere poort te kiezen. Anders klopt het overzicht niet meer.
exec .venv/bin/jupyter lab --port="$POORT" --port-retries=0 "$NOTEBOOK"
