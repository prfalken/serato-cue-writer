"""
Detect drops in a DJ mix (house / tech house / techno).

Heuristic: a drop is characterized by a sudden, sustained jump in low-band
energy (kick + sub-bass returning after a buildup). We score each frame on
(bass_jump * 0.6 + rms_jump * 0.4) gated by post-drop sustainability, then
non-max-suppress peaks at least --min-gap seconds apart, and snap each peak
to the nearest beat for clean cuts.

Outputs (next to the input file, or in --out-dir):
  <mix>_drops.json   machine-readable timecodes + scores + tempo
  <mix>_drops.txt    one timecode per line, mm:ss format
  <mix>_drops.png    waveform + bass energy + score curve + detected drops
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


HOP = 512
N_FFT = 2048
TARGET_SR = 22050
BASS_LOW_HZ = 40
BASS_HIGH_HZ = 120


@dataclass
class Drop:
    time_sec: float
    time_str: str
    score: float
    snapped_to_beat: bool


def fmt(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


def smooth(x: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames < 2:
        return x
    kernel = np.ones(window_frames) / window_frames
    return np.convolve(x, kernel, mode="same")


def detect(audio_path: Path, min_gap: float, top_n: int | None, snap_to_beat: bool, drop_offset: float = -1.0):
    print(f"[load] {audio_path.name}", file=sys.stderr)
    y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    duration = len(y) / sr
    print(f"[load] duration={fmt(duration)} sr={sr}", file=sys.stderr)

    fps = sr / HOP
    times = librosa.frames_to_time(np.arange(int(len(y) / HOP) + 1), sr=sr, hop_length=HOP)

    print("[features] computing STFT, RMS, onset strength", file=sys.stderr)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    bass_mask = (freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ)
    bass_energy = S[bass_mask, :].mean(axis=0)

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    n = min(len(rms), len(bass_energy), len(times))
    rms = rms[:n]
    bass_energy = bass_energy[:n]
    times = times[:n]

    smooth_w = max(2, int(1.5 * fps))
    rms_sm = smooth(rms, smooth_w)
    bass_sm = smooth(bass_energy, smooth_w)

    print("[scoring] drop signature (bass jump + RMS jump + sustainability)", file=sys.stderr)
    pre_w = int(4 * fps)
    post_w = int(8 * fps)
    sustain_w = int(8 * fps)

    scores = np.zeros(n, dtype=np.float64)
    eps = 1e-6
    for i in range(pre_w, n - post_w):
        bass_pre = bass_sm[i - pre_w:i].mean()
        bass_post = bass_sm[i:i + post_w].mean()
        rms_pre = rms_sm[i - pre_w:i].mean()
        rms_post = rms_sm[i:i + post_w].mean()

        bass_jump = bass_post / (bass_pre + eps)
        rms_jump = rms_post / (rms_pre + eps)

        sustain_end = min(i + sustain_w, n)
        sustain_min = rms_sm[i:sustain_end].min()
        sustain_ratio = sustain_min / (rms_post + eps)
        sustain_gate = sustain_ratio if sustain_ratio > 0.6 else 0.3

        raw = (bass_jump * 0.6 + rms_jump * 0.4) * sustain_gate
        scores[i] = max(0.0, raw - 1.0)

    print("[peaks] non-max-suppression", file=sys.stderr)
    min_dist_frames = max(1, int(min_gap * fps))
    peak_idx, props = find_peaks(scores, distance=min_dist_frames, prominence=0.15)

    if len(peak_idx) == 0:
        print("[peaks] none found — try lowering thresholds or check the mix", file=sys.stderr)
        return [], times, rms_sm, bass_sm, scores, None

    print(f"[peaks] {len(peak_idx)} candidate(s)", file=sys.stderr)

    tempo = None
    beats_t = np.array([])
    if snap_to_beat:
        print("[beats] beat-tracking for snap-to-beat", file=sys.stderr)
        tempo_arr, beats_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
        tempo = float(np.atleast_1d(tempo_arr)[0])
        beats_t = librosa.frames_to_time(beats_frames, sr=sr, hop_length=HOP)

    drops: list[Drop] = []
    for idx in peak_idx:
        # The scoring peak lags the real drop attack by ~1s (energy needs
        # time to settle in the post-window). drop_offset shifts back.
        t_raw = float(times[idx]) + drop_offset
        if t_raw < 0:
            continue
        snapped = False
        t = t_raw
        if snap_to_beat and len(beats_t) > 0:
            j = int(np.argmin(np.abs(beats_t - t_raw)))
            if abs(beats_t[j] - t_raw) < 0.5:
                t = float(beats_t[j])
                snapped = True
        drops.append(Drop(
            time_sec=round(t, 3),
            time_str=fmt(t),
            score=round(float(scores[idx]), 4),
            snapped_to_beat=snapped,
        ))

    drops.sort(key=lambda d: -d.score)
    if top_n:
        drops = drops[:top_n]
    drops.sort(key=lambda d: d.time_sec)

    return drops, times, rms_sm, bass_sm, scores, tempo


def render_png(out_path: Path, audio_name: str, times, rms_sm, bass_sm, scores, drops, duration):
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f"Drop detection — {audio_name}", fontsize=13)

    def norm(x):
        m = x.max()
        return x / m if m > 0 else x

    axes[0].plot(times, norm(rms_sm), color="#444", linewidth=0.7)
    axes[0].set_ylabel("RMS (norm)")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(times, norm(bass_sm), color="#c0392b", linewidth=0.7)
    axes[1].set_ylabel("Bass 40-120Hz (norm)")
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(times, scores, color="#2c3e50", linewidth=0.7)
    axes[2].set_ylabel("Drop score (clipped)")
    axes[2].set_xlabel("Time (s)")
    # Clip score panel so small drops stay visible despite 1-2 outlier peaks.
    if len(drops):
        sorted_scores = sorted((d.score for d in drops), reverse=True)
        cap = sorted_scores[min(2, len(sorted_scores) - 1)] * 1.1
        cap = max(cap, 2.0)
    else:
        cap = max(scores.max(), 1.0)
    axes[2].set_ylim(0, cap)

    for d in drops:
        for ax in axes:
            ax.axvline(d.time_sec, color="#27ae60", linewidth=1.1, alpha=0.85)
        label_y = min(d.score, cap) * 0.95
        axes[2].annotate(
            f"{d.time_str}\n{d.score:.1f}",
            xy=(d.time_sec, label_y),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=7, color="#27ae60",
        )

    for ax in axes:
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, type=Path, help="path to mix file (wav/mp3/flac/m4a)")
    p.add_argument("--out-dir", type=Path, default=None, help="output dir (default: alongside input)")
    p.add_argument("--min-gap", type=float, default=20.0, help="minimum seconds between drops (default 20)")
    p.add_argument("--top-n", type=int, default=None, help="keep only top-N highest scoring drops")
    p.add_argument("--drop-offset", type=float, default=-1.0,
                   help="seconds to add to each detected time (default -1.0; the score peak lags the real drop attack)")
    p.add_argument("--no-snap", action="store_true", help="disable snap-to-beat")
    p.add_argument("--no-png", action="store_true", help="skip waveform PNG export")
    args = p.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    drops, times, rms_sm, bass_sm, scores, tempo = detect(
        args.input, args.min_gap, args.top_n,
        snap_to_beat=not args.no_snap,
        drop_offset=args.drop_offset,
    )

    duration = float(times[-1]) if len(times) else 0.0

    payload = {
        "input": str(args.input),
        "duration_sec": round(duration, 3),
        "tempo_bpm": round(tempo, 2) if tempo else None,
        "min_gap_sec": args.min_gap,
        "drops": [asdict(d) for d in drops],
    }
    (out_dir / f"{stem}_drops.json").write_text(json.dumps(payload, indent=2))
    (out_dir / f"{stem}_drops.txt").write_text(
        "\n".join(f"{d.time_str}  score={d.score:.2f}" for d in drops) + "\n"
    )

    if not args.no_png:
        render_png(out_dir / f"{stem}_drops.png", args.input.name, times, rms_sm, bass_sm, scores, drops, duration)

    print()
    print(f"=== {len(drops)} drop(s) detected in {args.input.name} ===")
    if tempo:
        print(f"tempo: {tempo:.1f} BPM")
    print(f"{'#':>3}  {'time':>8}  {'score':>6}  beat-snap")
    for i, d in enumerate(drops, 1):
        print(f"{i:>3}  {d.time_str:>8}  {d.score:>6.2f}  {'yes' if d.snapped_to_beat else 'no'}")
    print()
    print(f"outputs in: {out_dir}")
    print(f"  - {stem}_drops.json")
    print(f"  - {stem}_drops.txt")
    if not args.no_png:
        print(f"  - {stem}_drops.png  ← open this to visually validate")


if __name__ == "__main__":
    main()
