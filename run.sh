#!/usr/bin/env bash
# serato-cue-writer — analyze (librosa) then write (mutagen).
#
#   ./run.sh /path/to/tracks            # analyze + DRY-RUN (nothing written)
#   ./run.sh /path/to/tracks --write    # analyze + WRITE (backs up first)
#
# QUIT SERATO before --write. After writing, in Serato select the tracks ->
# right-click -> "Read Tags" / "Re-scan File Info" if cues don't appear.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$HERE/venv/bin/python}"
[ -x "$PY" ] || PY="python3"
INPUT="${1:?usage: run.sh <file-or-dir> [--write]}"
WRITE="${2:-}"

INPUT_ABS="$(python3 -c "import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))" "$INPUT")"
if [ -d "$INPUT_ABS" ]; then PLAN="$INPUT_ABS/_cue_plan.json"; else PLAN="${INPUT_ABS%.mp3}.cue_plan.json"; fi

echo ">> analyze"
"$PY" "$HERE/analyze.py" --input "$INPUT_ABS"

echo
echo ">> write"
if [ "$WRITE" = "--write" ]; then
  "$PY" "$HERE/write_cues.py" --plan "$PLAN" --write
else
  "$PY" "$HERE/write_cues.py" --plan "$PLAN"
fi
