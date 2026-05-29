"""
ANALYZE step (run in the venv: needs librosa + mutagen).

For each MP3: read the Serato BeatGrid (exact BPM + first downbeat) and any
existing hot cues, detect the first/last drop, derive the 32-beat build cues
and the mixout, and emit a plan JSON consumed by write_cues.py.

  Begin       RED     (skipped later if a cue already sits near 0:00)
  First drop  ORANGE
  -32 beats   BLUE    (build into first drop)
  Last drop   ORANGE
  -32 beats   BLUE    (build into last drop)
  Mixout      PURPLE  (outro thins out -> room to mix out)

Usage:
  ./venv/bin/python analyze.py --input /path/to/tracks [--out plan.json]
                               [--pre-beats 32] [--min-gap 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mutagen.id3 import ID3

# sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
import serato_format as sf          # noqa: E402
from detect_drops import detect, fmt, render_png, Drop  # noqa: E402


ROLE_COLOR = {
    "BEGIN": "RED",
    "DROP1": "ORANGE",
    "PRE_DROP1": "BLUE",
    "DROP_LAST": "ORANGE",
    "PRE_DROP_LAST": "BLUE",
    "MIXOUT": "PURPLE",
}


def read_geob(tags: ID3, desc: str) -> bytes | None:
    for g in tags.getall("GEOB"):
        if g.desc == desc:
            return g.data
    return None


def get_grid(tags: ID3, tempo_fallback: float | None, duration: float):
    """Return (anchor_sec, bpm, source). Prefer the Serato BeatGrid."""
    data = read_geob(tags, "Serato BeatGrid")
    if data:
        try:
            bg = sf.parse_beatgrid(data)
            if bg.bpm > 0:
                return bg.anchor_sec, bg.bpm, "beatgrid"
        except Exception as e:
            print(f"  [grid] beatgrid parse failed ({e}); using librosa", file=sys.stderr)
    bpm = tempo_fallback or 124.0
    return 0.0, float(bpm), "librosa"


def existing_cues(tags: ID3):
    data = read_geob(tags, "Serato Markers2")
    if not data:
        return []
    try:
        m = sf.parse_markers2(data)
    except Exception:
        return []
    return [{"index": c.index, "pos_ms": c.position_ms, "rgb": c.color.hex(), "name": c.name}
            for c in m.cues]


def subgate_drops(times, bass_sm, beat, anchor):
    """Detect drops as the *return of the kick/sub-bass after a trough*.

    A house/tech-house drop is defined by the low end (kick + sub, the
    40-120 Hz `bass_sm` band) slamming back in and staying, after a breakdown
    / build where it was reduced or absent. This beats scoring raw energy
    jumps, which gets fooled by the build's rising mids/risers.

    Returns a sorted list of drop times (rising edges, bar-aligned), one per
    sustained low->high sub transition.
    """
    bar = 4 * beat
    dur = float(times[-1])
    nbars = int((dur - anchor) / bar)
    if nbars < 8 or len(times) < 2:
        return []
    edges = anchor + np.arange(nbars + 1) * bar
    lvl = np.array([
        bass_sm[(times >= edges[k]) & (times < edges[k + 1])].mean()
        if np.any((times >= edges[k]) & (times < edges[k + 1])) else 0.0
        for k in range(nbars)])
    full = float(np.percentile(lvl, 75))
    if full <= 0:
        return []
    hi, lo = 0.5 * full, 0.32 * full           # hysteresis kills flicker
    on = np.zeros(nbars, bool)
    state = lvl[0] > hi
    for k in range(nbars):
        if state and lvl[k] < lo:
            state = False
        elif (not state) and lvl[k] > hi:
            state = True
        on[k] = state
    drops = []
    k = 1
    while k < nbars - 6:
        if on[k] and not on[k - 1]:
            off = 0
            j = k - 1
            while j >= 0 and not on[j]:
                off += 1
                j -= 1
            if off >= 4 and on[k:k + 6].mean() > 0.7:   # real trough + sustained
                drops.append(anchor + k * bar)
                k += 8
                continue
        k += 1
    return drops


def find_mixout(times, rms_sm, first_drop_t, last_drop_t, beat, duration, lead_bars=8):
    """Start of the final outro = the last moment the track is at (near) full
    energy; everything after that is the thinned-out outro you mix over.

    Why not "first point below a threshold after the last drop": at the drop's
    *onset* the track is still building (low energy), so a reference taken there
    is far too low and nothing ever falls under it until the final fade. Instead
    we take the loud-section energy as the reference (90th percentile over the
    post-first-drop region) and find the last time RMS reaches ~that level.

    lead_bars shifts the cue earlier (in bars) so you can start mixing out
    before the energy actually drops — a stylistic runway.
    """
    bar = 4 * beat
    if len(times) < 2:
        return max(0.0, duration - 16 * beat)

    def idx(t):
        return int(min(len(times) - 1, max(0, np.searchsorted(times, t))))

    loud = rms_sm[idx(first_drop_t):idx(duration - 4 * bar)]
    if len(loud) == 0:
        return min(last_drop_t + 32 * beat, max(0.0, duration - 16 * beat))
    peak = float(np.percentile(loud, 90))
    hi = 0.85 * peak

    # last index (before the final couple of bars) still at near-full energy
    region = rms_sm[:idx(duration - 2 * bar)]
    above = np.where(region >= hi)[0]
    if len(above) == 0:
        return min(last_drop_t + 32 * beat, max(0.0, duration - 16 * beat))
    outro_start = float(times[above[-1]]) + bar      # next bar = outro begins
    mix = outro_start - lead_bars * bar
    return min(max(mix, last_drop_t + 8 * bar), max(0.0, duration - 8 * beat))


def plan_for(path: Path, pre_beats: int, min_gap: float, png: bool = True, mixout_lead_bars: int = 8):
    tags = ID3(str(path))
    # snap_to_beat=False: skips librosa's expensive beat_track. We re-snap every
    # cue to the Serato BeatGrid anyway, so its beat snapping is redundant.
    drops, times, rms_sm, bass_sm, scores, tempo = detect(
        path, min_gap=min_gap, top_n=None, snap_to_beat=False)
    duration = float(times[-1]) if len(times) else 0.0
    anchor, bpm, gsrc = get_grid(tags, tempo, duration)
    beat = 60.0 / bpm

    bar = 4 * beat

    def snap_beat(t):
        return max(0.0, anchor + round((t - anchor) / beat) * beat)

    def snap_bar(t):
        # Drops land on a downbeat (the "1" of a bar). The Serato grid anchor
        # is the first downbeat, so snap to the nearest bar boundary — snapping
        # to the nearest *beat* can land a drop 1 beat early (on beat 4 of the
        # previous bar).
        return max(0.0, anchor + round((t - anchor) / bar) * bar)

    planned = []

    def add(role, t, on_bar=True):
        snapped = snap_bar(t) if on_bar else snap_beat(t)
        ms = int(round(snapped * 1000))
        planned.append({"role": role, "color": ROLE_COLOR[role],
                        "rgb": sf.COLORS[ROLE_COLOR[role]].hex(),
                        "pos_ms": ms, "time_str": fmt(ms / 1000.0)})

    add("BEGIN", 0.0, on_bar=False)

    # PRIMARY: drops = sustained return of the kick/sub after a trough
    # (robust against builds). FALLBACK: the old energy-jump detector when the
    # sub-gate finds nothing (e.g. tracks with no clean breakdown).
    gate = subgate_drops(times, bass_sm, beat, anchor)
    if gate:
        drop_times = gate
        drop_source = "subgate"
    else:
        floor = max(1.5, 0.3 * max((d.score for d in drops), default=0))
        drop_times = sorted(d.time_sec for d in drops
                            if d.score >= floor and 12.0 <= d.time_sec <= duration - 8.0)
        drop_source = "energy-fallback"

    if drop_times:
        first_t = drop_times[0]
        last_t = drop_times[-1]
        add("DROP1", first_t)
        if first_t - pre_beats * beat > 0:
            add("PRE_DROP1", first_t - pre_beats * beat)
        if last_t > first_t + beat:           # only if a distinct second drop
            add("DROP_LAST", last_t)
            if last_t - pre_beats * beat > first_t:
                add("PRE_DROP_LAST", last_t - pre_beats * beat)
        mixout_t = find_mixout(times, rms_sm, first_t, last_t, beat, duration,
                               lead_bars=mixout_lead_bars)
        add("MIXOUT", mixout_t)

    # Validation PNG: the drop-detector waveform with the *planned cue times*
    # overlaid (green lines), so the plan can be eyeballed before writing.
    if png:
        marks = [Drop(time_sec=c["pos_ms"] / 1000.0, time_str=c["time_str"],
                      score=0.0, snapped_to_beat=True) for c in planned]
        try:
            render_png(path.with_suffix(".cue_plan.png"), path.name,
                       times, rms_sm, bass_sm, scores, marks, duration)
        except Exception as e:
            print(f"  [png] skipped: {e}", file=sys.stderr)

    return {
        "file": str(path),
        "duration_sec": round(duration, 3),
        "grid": {"anchor_sec": round(anchor, 4), "bpm": round(bpm, 3), "source": gsrc},
        "drop_source": drop_source,
        "drops_detected": [fmt(t) for t in drop_times],
        "existing_cues": existing_cues(tags),
        "planned": planned,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, type=Path, help="MP3 file or directory")
    p.add_argument("--out", "-o", type=Path, default=None, help="plan JSON (default: <input>/_cue_plan.json)")
    p.add_argument("--pre-beats", type=int, default=32, help="beats before each drop for the BLUE build cue (default 32 = 8 bars)")
    p.add_argument("--min-gap", type=float, default=16.0, help="min seconds between detected drops")
    p.add_argument("--no-png", action="store_true", help="skip the validation PNG")
    p.add_argument("--mixout-lead-bars", type=int, default=8,
                   help="bars before the detected outro start to place the OUT cue (default 8 = 1 phrase of runway; 0 = exactly at the outro)")
    p.add_argument("--limit", type=int, default=None, help="process at most N files")
    args = p.parse_args()

    inp = args.input.expanduser()
    if inp.is_dir():
        files = sorted(inp.glob("*.mp3"))
        out = args.out or (inp / "_cue_plan.json")
    else:
        files = [inp]
        out = args.out or inp.with_suffix(".cue_plan.json")
    if args.limit:
        files = files[:args.limit]

    plans = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        try:
            plans.append(plan_for(f, args.pre_beats, args.min_gap, png=not args.no_png,
                                  mixout_lead_bars=args.mixout_lead_bars))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            plans.append({"file": str(f), "error": str(e), "planned": []})

    out.write_text(json.dumps(plans, indent=2))
    print(f"\nwrote plan for {len(plans)} track(s) -> {out}")
    for pl in plans:
        if pl.get("error"):
            print(f"  ! {Path(pl['file']).name}: {pl['error']}")
            continue
        g = pl["grid"]
        print(f"\n  {Path(pl['file']).name}  [{g['bpm']} BPM, grid={g['source']}]")
        print(f"    drops detected ({pl.get('drop_source')}): {pl.get('drops_detected')}")
        ex = {c['index'] for c in pl['existing_cues']}
        print(f"    existing cues: slots {sorted(ex) if ex else '(none)'}")
        for c in pl["planned"]:
            print(f"    {c['role']:14} {c['time_str']:>9}  {c['color']}")


if __name__ == "__main__":
    main()
