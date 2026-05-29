"""
WRITE step (run in python3: needs mutagen).

Consume the plan JSON from analyze.py and write the hot cues into each MP3:
  - Markers2 is authoritative -> add the planned cues there.
  - Markers_ (if present) mirrors the first <=5 hot-cue slots, exactly like
    Serato itself does.

Safety:
  - --dry-run (DEFAULT) just prints the merge table; pass --write to save.
  - every file is copied to backups/<timestamp>/ before its first write.
  - MERGE, never wipe: existing cues are preserved; a BEGIN cue is skipped if
    one already sits near 0:00; a planned cue is skipped if a cue already
    exists near that position (idempotent re-runs add nothing).
  - all other GEOB frames (BeatGrid/Overview/Autotags/Analysis/FLIP/LOOP) are
    left untouched, and re-verified byte-identical after saving.

Usage:
  python3 write_cues.py --plan /path/to/_cue_plan.json            # dry-run
  python3 write_cues.py --plan /path/to/_cue_plan.json --write     # for real
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from mutagen.id3 import ID3, GEOB

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serato_format as sf          # noqa: E402

NEAR_MS = 1500          # treat cues within 1.5 s as "the same" (dedup / begin)
N_SLOTS = 8             # Serato hot-cue banks

# Label written into the Serato cue (distinct from the internal role).
ROLE_NAME = {
    "BEGIN": "IN",
    "DROP1": "DROP",
    "PRE_DROP1": "BUILD",
    "DROP_LAST": "DROP",
    "PRE_DROP_LAST": "BUILD",
    "MIXOUT": "OUT",
}


def get_frame(tags: ID3, desc: str):
    for g in tags.getall("GEOB"):
        if g.desc == desc:
            return g
    return None


def other_geob_snapshot(tags: ID3) -> dict[str, bytes]:
    return {g.desc: bytes(g.data) for g in tags.getall("GEOB")
            if g.desc not in ("Serato Markers2", "Serato Markers_")}


def merge_track(plan: dict, replace: bool = False):
    """Return (markers2, markers_legacy_or_None, actions[]) without saving.

    actions: list of dicts describing each planned cue's disposition.
    replace=True wipes existing hot cues first (keeping track colour, bpm-lock,
    loops and flips) and writes our scheme fresh from slot 0.
    """
    path = Path(plan["file"])
    tags = ID3(str(path))

    m2_frame = get_frame(tags, "Serato Markers2")
    if m2_frame is not None:
        m2 = sf.parse_markers2(m2_frame.data)
    else:
        m2 = sf.Markers2(cues=[], entries=[])

    mu_frame = get_frame(tags, "Serato Markers_")
    mu = sf.parse_markers_(mu_frame.data) if mu_frame is not None else None

    if replace:
        m2.cues = []                       # drop existing hot cues (Markers2)
        if mu is not None:                 # and unset Markers_ cue slots
            for e in mu.entries:
                if e.is_cue_slot:
                    e.start_flag = 0x7F
                    e.start_ms = 0xFFFFFF
                    e.color = (0, 0, 0)
                    e.type = 0x00
                    e.locked = 0x00

    occupied = {c.index for c in m2.cues}
    positions = [c.position_ms for c in m2.cues]

    def pos_taken(ms: int) -> bool:
        return any(abs(ms - p) <= NEAR_MS for p in positions)

    def free_slot() -> int | None:
        for k in range(N_SLOTS):
            if k not in occupied:
                return k
        return None

    actions = []
    for c in plan.get("planned", []):
        role, ms, rgb_hex = c["role"], c["pos_ms"], c["rgb"]
        rgb = bytes.fromhex(rgb_hex)

        if role == "BEGIN" and any(p <= NEAR_MS for p in positions):
            actions.append({**c, "status": "skip", "reason": "begin already present"})
            continue
        if pos_taken(ms):
            actions.append({**c, "status": "skip", "reason": "cue already near this position"})
            continue
        slot = free_slot()
        if slot is None:
            actions.append({**c, "status": "skip", "reason": "no free cue slot"})
            continue

        m2.cues.append(sf.Cue(index=slot, position_ms=ms, color=rgb,
                              name=ROLE_NAME.get(role, role.replace("_", " "))))
        occupied.add(slot)
        positions.append(ms)
        mirrored = False
        if mu is not None:
            mirrored = sf.set_legacy_cue(mu, slot, ms, rgb)
        actions.append({**c, "status": "add", "slot": slot, "mirrored_markers_": mirrored})

    return tags, m2_frame, m2, mu_frame, mu, actions


def save_track(path: Path, tags, m2_frame, m2, mu_frame, mu, backup_dir: Path):
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    if not dest.exists():
        shutil.copy2(path, dest)

    before = other_geob_snapshot(tags)

    pad = len(m2_frame.data) if m2_frame is not None else None
    blob = sf.build_markers2(m2, pad_to=pad)
    if m2_frame is not None:
        m2_frame.data = blob
    else:
        tags.add(GEOB(encoding=0, mime="application/octet-stream",
                      desc="Serato Markers2", data=blob))
    if mu_frame is not None and mu is not None:
        mu_frame.data = sf.build_markers_(mu)

    # mutagen only writes ID3 v2.3 or v2.4; bump anything older (v2.2) to v2.3.
    v = tags.version[1] if (tags.version and tags.version[1] in (3, 4)) else 3
    tags.save(str(path), v2_version=v)

    # verify other frames untouched
    after = other_geob_snapshot(ID3(str(path)))
    changed = [k for k in before if before[k] != after.get(k)]
    if changed:
        raise RuntimeError(f"other GEOB frames changed during save: {changed}")


def fmt_ms(ms: int) -> str:
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s - 60 * int(s // 60):05.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, type=Path, help="plan JSON from analyze.py")
    p.add_argument("--write", action="store_true", help="actually write (default is dry-run)")
    p.add_argument("--replace", action="store_true",
                   help="wipe existing hot cues first, then write our scheme fresh (loops/flips/track-colour kept; backed up first)")
    p.add_argument("--backup-dir", type=Path, default=None, help="backup dir (default: ./backups/<ts>)")
    args = p.parse_args()

    import json
    plans = json.loads(args.plan.read_text())
    if isinstance(plans, dict):
        plans = [plans]

    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = args.backup_dir or (Path(__file__).resolve().parent / "backups" / ts)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"=== {mode} === ({len(plans)} track(s))  backups -> {backup_dir}\n")

    total_add = total_skip = 0
    for plan in plans:
        name = Path(plan["file"]).name
        if plan.get("error"):
            print(f"! {name}: {plan['error']}\n")
            continue
        tags, m2_frame, m2, mu_frame, mu, actions = merge_track(plan, replace=args.replace)
        has_mu = mu_frame is not None
        tag = " (Markers_=%s%s)" % ("yes" if has_mu else "no", ", REPLACE" if args.replace else "")
        print(f"{name}{tag}")
        for a in actions:
            if a["status"] == "add":
                tag = f"slot {a['slot']}" + ("" if a["mirrored_markers_"] else " [M2-only]")
                print(f"   + {a['role']:14} {fmt_ms(a['pos_ms']):>9}  {a['color']:7} {tag}")
                total_add += 1
            else:
                print(f"   - {a['role']:14} {fmt_ms(a['pos_ms']):>9}  {a['color']:7} skip: {a['reason']}")
                total_skip += 1
        if args.write and any(a["status"] == "add" for a in actions):
            try:
                save_track(Path(plan["file"]), tags, m2_frame, m2, mu_frame, mu, backup_dir)
                print("   => saved")
            except Exception as e:
                print(f"   !! SAVE FAILED, not modified: {e}")
        print()

    print(f"total: {total_add} cue(s) to add, {total_skip} skipped")
    if not args.write:
        print("dry-run only — re-run with --write to apply.")


if __name__ == "__main__":
    main()
