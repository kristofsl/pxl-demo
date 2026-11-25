#!/usr/bin/env bash
#
# Start alle topics tegelijk, elk op zijn eigen poort.
#
#   ./start-alles.sh          alles starten en het overzicht tonen
#   ./start-alles.sh stop     alles stoppen
#
# Stoppen kan ook met Ctrl-C in dit venster: dan gaan ze allemaal netjes uit.
#
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HIER"

LOGMAP=".logs"
WACHT_MAX=600          # seconden; de allereerste keer worden pakketten geinstalleerd

# Alle mappen die een start.sh hebben, op volgorde.
# (geen mapfile: dat bestaat niet in de bash die standaard op macOS staat)
TOPICS=()
while IFS= read -r regel; do
    TOPICS+=("$regel")
done < <(find . -mindepth 2 -maxdepth 2 -name start.sh | sed 's|^\./||; s|/start\.sh$||' | sort)

poort_van() { grep -m1 '^POORT=' "$1/start.sh" | cut -d= -f2; }
titel_van() { grep -m1 '^TOPIC=' "$1/start.sh" | cut -d'"' -f2; }

stop_alles() {
    local gestopt=0
    for t in "${TOPICS[@]}"; do
        local pids
        pids="$(lsof -ti ":$(poort_van "$t")" 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
            gestopt=$((gestopt + 1))
        fi
    done
    echo "$gestopt notebook(s) gestopt."
}

if [ "${1:-}" = "stop" ] || [ "${1:-}" = "--stop" ]; then
    stop_alles
    exit 0
fi

trap 'echo; echo "Stoppen..."; stop_alles; exit 0' INT TERM

mkdir -p "$LOGMAP"

# Geen vijf browsertabbladen tegelijk; het overzicht hieronder is handiger.
export JUPYTER_EXTRA="--no-browser"

echo "=============================================================="
echo " Alle topics starten"
echo "=============================================================="
echo

gestart=0
for t in "${TOPICS[@]}"; do
    poort="$(poort_van "$t")"
    if lsof -ti ":$poort" >/dev/null 2>&1; then
        printf "  %-28s draaide al op poort %s\n" "$t" "$poort"
        continue
    fi
    "$t/start.sh" > "$LOGMAP/$t.log" 2>&1 &
    gestart=$((gestart + 1))
    printf "  %-28s gestart\n" "$t"
done

if [ "$gestart" -gt 0 ]; then
    echo
    echo "Wachten tot ze allemaal luisteren. De eerste keer duurt dit een paar"
    echo "minuten omdat de pakketten nog geinstalleerd moeten worden."
fi

# Per topic wachten tot de URL met token in het log verschijnt.
# URLS loopt gelijk op met TOPICS; geen associatieve array, want die kent
# de standaard-bash van macOS niet.
URLS=()
for t in "${TOPICS[@]}"; do
    poort="$(poort_van "$t")"
    gevonden=""
    for ((s = 0; s < WACHT_MAX; s++)); do
        if [ -f "$LOGMAP/$t.log" ]; then
            gevonden="$(grep -o "http://localhost:$poort/lab?token=[a-f0-9]*" "$LOGMAP/$t.log" | head -1 || true)"
        fi
        # Draaide hij al, dan staat er niets in ons log maar luistert de poort wel.
        if [ -z "$gevonden" ] && lsof -ti ":$poort" >/dev/null 2>&1 && [ "$s" -gt 2 ]; then
            gevonden="http://localhost:$poort/lab"
        fi
        [ -n "$gevonden" ] && break
        sleep 1
    done
    URLS+=("${gevonden:-NIET GESTART - kijk in $LOGMAP/$t.log}")
done

echo
echo "=============================================================="
echo " Klaar. Open deze in je browser:"
echo "=============================================================="
echo
for i in "${!TOPICS[@]}"; do
    printf "  %-32s %s\n" "$(titel_van "${TOPICS[$i]}")" "${URLS[$i]}"
done
echo
echo "Stoppen: Ctrl-C hier, of './start-alles.sh stop' in een ander venster."
echo

# Open blijven zodat Ctrl-C alles kan afsluiten.
while true; do sleep 3600 & wait $!; done
