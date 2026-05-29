"""
Serato ID3 GEOB codecs — read/write hot cues + colors in MP3 files.

Stdlib only (no numpy/mutagen here) so it imports cleanly everywhere.

Formats reverse-engineered AND verified against a real Serato DJ Pro library
of ~1600 MP3s (see README.md "How the Serato format works"):

  Serato Markers2  (authoritative for hot cues in modern Serato DJ Pro)
    data  = b'\\x01\\x01' + base64(payload, wrapped at 72 cols) + b'\\x00' [+ null pad]
    base64 quirks seen in the wild: standard '=' padding OR stripped;
      a length of (mod 4 == 1) decodes by appending b'A==' (Holzhaus rule).
    payload = b'\\x01\\x01' + entries + b'\\x00'
    entry   = name(utf-8) + b'\\x00' + uint32_be(len(body)) + body
    CUE body (>=13 bytes):
      [0]   0x00 field
      [1]   index   (hot-cue slot 0..7)
      [2:6] position ms, uint32 big-endian
      [6]   0x00
      [7:10] RGB (3 bytes, e.g. cc 00 00)
      [10:12] 0x00 0x00
      [12:] name utf-8, null-terminated (empty -> single 0x00)
    Other entries (COLOR, BPMLOCK, LOOP, FLIP) are preserved verbatim.

  Serato Markers_ (legacy ScratchLive; modern Serato mirrors only the
    first <=5 hot cues here, rest live only in Markers2). RAW binary in
    the ID3 frames of THIS library (not base64).
    data  = b'\\x02\\x05' + uint32_be(count) + count*22-byte entries
            + 4-byte serato32 track-color footer
    entry (22 bytes):
      [0]    start flag  0x00=set, 0x7f=unset
      [1:5]  start pos   serato32  (24-bit ms)
      [5]    end flag    0x7f for cues (no loop end)
      [6:10] end pos     serato32  (0x7f7f7f7f when unset)
      [10:16] reserved   (copied verbatim; observed 00 7f 7f 7f 7f 7f)
      [16:20] color      serato32  (RGB)
      [20]   type        0x00/0x01 = cue, 0x03 = loop
      [21]   locked      0x00/0x01

  Serato BeatGrid (raw binary, NOT base64):
    data = b'\\x01\\x00' + uint32_be(count)
           + (count-1) * (float32_be pos_sec, uint32_be beats_to_next)
           + 1 * (float32_be pos_sec, float32_be bpm)        # terminal
           + optional 1 footer byte
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Standard Serato hot-cue colours (the bytes STORED in the tag; Serato DJ Pro
# renders them as its palette). Verified present in the real library.
# ---------------------------------------------------------------------------
COLORS = {
    "RED":     b"\xcc\x00\x00",
    "ORANGE":  b"\xcc\x88\x00",
    "BLUE":    b"\x00\x00\xcc",
    "PURPLE":  b"\x88\x00\xcc",
    "YELLOW":  b"\xcc\xcc\x00",
    "GREEN":   b"\x00\xcc\x00",
    "CYAN":    b"\x00\xcc\xcc",
    "MAGENTA": b"\xcc\x00\xcc",
}


# ---------------------------------------------------------------------------
# serato32 — 7-bit packed codec used by Markers_ for positions and colours.
# ---------------------------------------------------------------------------
def serato32_decode(w: int, x: int, y: int, z: int) -> tuple[int, int, int]:
    c = (z & 0x7F) | ((y & 0x01) << 7)
    b = ((y & 0x7F) >> 1) | ((x & 0x03) << 6)
    a = ((x & 0x7F) >> 2) | ((w & 0x07) << 5)
    return a, b, c


def serato32_encode(a: int, b: int, c: int) -> bytes:
    z = c & 0x7F
    y = ((c >> 7) | (b << 1)) & 0x7F
    x = ((b >> 6) | (a << 2)) & 0x7F
    w = (a >> 5) & 0x7F
    return bytes((w, x, y, z))


def _s32_ms_decode(four: bytes) -> int:
    a, b, c = serato32_decode(*four)
    return (a << 16) | (b << 8) | c


def _s32_ms_encode(ms: int) -> bytes:
    return serato32_encode((ms >> 16) & 0xFF, (ms >> 8) & 0xFF, ms & 0xFF)


# ===========================================================================
# Markers2
# ===========================================================================
@dataclass
class Cue:
    index: int          # hot-cue slot 0..7
    position_ms: int
    color: bytes        # 3 bytes RGB
    name: str = ""


@dataclass
class Markers2:
    """Round-trippable view of a Serato Markers2 frame.

    `entries` keeps every non-CUE entry (COLOR/BPMLOCK/LOOP/FLIP) verbatim
    as (name_bytes, body_bytes) so we never lose track colour, bpm-lock,
    flips, etc. Cues are parsed out separately for editing.
    """
    cues: list[Cue]
    entries: list[tuple[bytes, bytes]]   # non-CUE entries, original order


def _b64decode_loose(b64: bytes) -> bytes:
    b64 = b64.replace(b"\n", b"").replace(b"\r", b"")
    rem = len(b64) % 4
    if rem == 1:
        b64 += b"A=="           # documented Serato quirk
    elif rem:
        b64 += b"=" * (4 - rem)
    return base64.b64decode(b64)


def parse_markers2(data: bytes) -> Markers2:
    if data[:2] != b"\x01\x01":
        raise ValueError("Markers2: bad prefix %r" % data[:2])
    end = data.find(b"\x00", 2)
    if end == -1:
        end = len(data)
    payload = _b64decode_loose(data[2:end])
    if payload[:2] != b"\x01\x01":
        raise ValueError("Markers2 payload: bad prefix %r" % payload[:2])

    cues: list[Cue] = []
    entries: list[tuple[bytes, bytes]] = []
    p = 2
    while p < len(payload):
        nul = payload.find(b"\x00", p)
        if nul == -1:
            break
        name = payload[p:nul]
        if not name:               # terminator
            break
        (length,) = struct.unpack(">I", payload[nul + 1:nul + 5])
        body = payload[nul + 5:nul + 5 + length]
        if name == b"CUE":
            idx = body[1]
            (pos,) = struct.unpack(">I", body[2:6])
            color = body[7:10]
            nm = body[12:].split(b"\x00", 1)[0].decode("utf-8", "replace")
            cues.append(Cue(idx, pos, color, nm))
        else:
            entries.append((name, body))
        p = nul + 5 + length
    return Markers2(cues=cues, entries=entries)


def _cue_body(cue: Cue) -> bytes:
    name = cue.name.encode("utf-8")
    return (b"\x00" + bytes((cue.index,)) + struct.pack(">I", cue.position_ms)
            + b"\x00" + bytes(cue.color[:3]) + b"\x00\x00" + name + b"\x00")


def build_markers2(m: Markers2, pad_to: int | None = None) -> bytes:
    """Serialize back to a Serato-accepted GEOB blob.

    Entry order mirrors what Serato writes: leading COLOR, then CUE entries
    sorted by index, then the remaining preserved entries (BPMLOCK/LOOP/FLIP).
    """
    color_entries = [(n, b) for n, b in m.entries if n == b"COLOR"]
    other_entries = [(n, b) for n, b in m.entries if n != b"COLOR"]

    payload = bytearray(b"\x01\x01")
    for name, body in color_entries:
        payload += name + b"\x00" + struct.pack(">I", len(body)) + body
    for cue in sorted(m.cues, key=lambda c: c.index):
        body = _cue_body(cue)
        payload += b"CUE\x00" + struct.pack(">I", len(body)) + body
    for name, body in other_entries:
        payload += name + b"\x00" + struct.pack(">I", len(body)) + body
    payload += b"\x00"           # terminator

    b64 = base64.b64encode(bytes(payload))
    wrapped = b"\n".join(b64[i:i + 72] for i in range(0, len(b64), 72))
    blob = b"\x01\x01" + wrapped + b"\x00"
    if pad_to and len(blob) < pad_to:
        blob = blob + b"\x00" * (pad_to - len(blob))
    return blob


# ===========================================================================
# Markers_
# ===========================================================================
@dataclass
class MarkersEntry:
    start_flag: int
    start_ms: int
    end_flag: int
    end_raw: bytes      # 4 bytes, kept verbatim (loops only)
    reserved: bytes     # 6 bytes, verbatim
    color: tuple[int, int, int]
    type: int           # 0/1 cue, 3 loop
    locked: int

    @property
    def is_cue_slot(self) -> bool:
        return self.type in (0, 1)

    @property
    def is_set(self) -> bool:
        return self.start_flag == 0x00


@dataclass
class MarkersLegacy:
    entries: list[MarkersEntry]
    footer: bytes       # 4-byte serato32 track colour (verbatim)


def parse_markers_(data: bytes) -> MarkersLegacy:
    if data[:2] != b"\x02\x05":
        raise ValueError("Markers_: bad prefix %r" % data[:2])
    (count,) = struct.unpack(">I", data[2:6])
    off = 6
    entries: list[MarkersEntry] = []
    for _ in range(count):
        e = data[off:off + 22]
        off += 22
        entries.append(MarkersEntry(
            start_flag=e[0],
            start_ms=_s32_ms_decode(e[1:5]),
            end_flag=e[5],
            end_raw=e[6:10],
            reserved=e[10:16],
            color=serato32_decode(*e[16:20]),
            type=e[20],
            locked=e[21],
        ))
    return MarkersLegacy(entries=entries, footer=data[off:])


def build_markers_(m: MarkersLegacy) -> bytes:
    out = bytearray(b"\x02\x05" + struct.pack(">I", len(m.entries)))
    for e in m.entries:
        out += bytes((e.start_flag,))
        out += _s32_ms_encode(e.start_ms) if e.is_set else b"\x7f\x7f\x7f\x7f"
        out += bytes((e.end_flag,))
        out += e.end_raw
        out += e.reserved
        out += serato32_encode(*e.color)
        out += bytes((e.type, e.locked))
    out += m.footer
    return bytes(out)


def set_legacy_cue(m: MarkersLegacy, index: int, ms: int, rgb: bytes) -> bool:
    """Mirror a hot cue into the index-th *cue slot* of Markers_.

    Returns True if written, False if there is no such cue slot (Markers_
    only has ~5 cue slots; cues beyond that live only in Markers2).
    """
    cue_slots = [i for i, e in enumerate(m.entries) if e.is_cue_slot]
    if index >= len(cue_slots):
        return False
    e = m.entries[cue_slots[index]]
    e.start_flag = 0x00
    e.start_ms = ms
    e.end_flag = 0x7F
    e.end_raw = b"\x7f\x7f\x7f\x7f"
    e.color = (rgb[0], rgb[1], rgb[2])
    e.type = 0x01
    e.locked = 0x00
    return True


# ===========================================================================
# BeatGrid (read-only)
# ===========================================================================
@dataclass
class BeatGrid:
    anchor_sec: float        # position of the first beat marker
    bpm: float


def parse_beatgrid(data: bytes) -> BeatGrid:
    if data[:2] != b"\x01\x00":
        raise ValueError("BeatGrid: bad prefix %r" % data[:2])
    (count,) = struct.unpack(">I", data[2:6])
    if count < 1:
        raise ValueError("BeatGrid: no markers")
    off = 6
    first_pos = struct.unpack(">f", data[off:off + 4])[0]
    # walk to the terminal marker (last one carries the BPM as float32)
    for i in range(count):
        if i < count - 1:
            off += 8           # non-terminal: pos(f32) + beats(u32)
        else:
            term_pos = struct.unpack(">f", data[off:off + 4])[0]
            bpm = struct.unpack(">f", data[off + 4:off + 8])[0]
            off += 8
    if count == 1:
        first_pos = term_pos
    return BeatGrid(anchor_sec=float(first_pos), bpm=float(bpm))


def snap_to_grid(t_sec: float, grid: BeatGrid) -> float:
    beat = 60.0 / grid.bpm
    n = round((t_sec - grid.anchor_sec) / beat)
    return grid.anchor_sec + n * beat
