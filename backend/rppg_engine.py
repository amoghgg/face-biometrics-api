"""
rPPG (Remote Photoplethysmography) Engine
==========================================
Detects a real heartbeat signal from facial video using the CHROM algorithm.

How it works:
  1. Every frame, we sample mean R/G/B from the forehead region.
  2. After collecting enough frames (~5 seconds), we run CHROM:
       - Normalize channels (removes lighting DC offset)
       - Build a combined pulse signal S = Xs - alpha*Ys
       - Bandpass filter S at 0.7–4.0 Hz (42–240 BPM)
       - FFT → dominant frequency → BPM
  3. If BPM is physiologically plausible (50–150), it's a real face.

A deepfake/photo/screen replay cannot reproduce this signal because:
  - The subtle per-pixel color oscillation isn't modeled in generative AI output
  - Screen replays have fixed pixel values per frame — no blood flow variance
"""

import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt


# How many seconds of data we maintain in the sliding window
WINDOW_SEC = 10

# Assumed camera FPS (used before we have enough timestamps to measure it)
DEFAULT_FPS = 15.0

# Minimum seconds of data before we attempt BPM estimation
MIN_SEC = 5

# Minimum *good* samples (stable-face frames only) before computing BPM.
# Using sample count rather than elapsed time because we skip motion-corrupted
# frames — elapsed time would be misleading if we were paused for 4 of 5 seconds.
MIN_SAMPLES = int(MIN_SEC * DEFAULT_FPS)   # 75 samples

# Physiological BPM bounds for "real human" decision
LIVE_BPM_LOW  = 45
LIVE_BPM_HIGH = 160

# Frequency band for bandpass filter (Hz)
FREQ_LOW  = 0.7   # 42 BPM
FREQ_HIGH = 4.0   # 240 BPM

# Minimum FFT peak confidence to trust the BPM reading
# Confidence = peak_power / total_power_in_band
MIN_CONFIDENCE = 0.15


class RPPGEngine:
    """
    Per-session rPPG processor. One instance per liveness session.

    Usage:
        engine = RPPGEngine()
        engine.add_sample(r_mean, g_mean, b_mean)   # called every frame
        bpm, confidence = engine.compute_bpm()
        live = engine.is_live()                      # None until enough data
    """

    def __init__(self):
        max_frames = int(WINDOW_SEC * DEFAULT_FPS * 2)  # generous buffer
        self.r_buf = deque(maxlen=max_frames)
        self.g_buf = deque(maxlen=max_frames)
        self.b_buf = deque(maxlen=max_frames)
        self.ts_buf = deque(maxlen=max_frames)  # timestamps for real FPS estimation

    def add_sample(self, r_mean: float, g_mean: float, b_mean: float, ts: float | None = None) -> None:
        """
        Add one frame's forehead RGB means to the buffer.
        ts: optional monotonic timestamp (seconds). Defaults to time.monotonic().
            Pass explicit timestamps in tests or when frame timing is known.
        """
        self.r_buf.append(r_mean)
        self.g_buf.append(g_mean)
        self.b_buf.append(b_mean)
        self.ts_buf.append(ts if ts is not None else time.monotonic())

    @property
    def n_samples(self) -> int:
        return len(self.g_buf)

    @property
    def ready(self) -> bool:
        """True once we have enough *good* samples (stable-face frames only)."""
        return len(self.g_buf) >= MIN_SAMPLES

    def _estimate_fps(self) -> float:
        """Compute actual FPS from timestamps — more accurate than assuming 15."""
        if len(self.ts_buf) < 2:
            return DEFAULT_FPS
        elapsed = self.ts_buf[-1] - self.ts_buf[0]
        if elapsed <= 0:
            return DEFAULT_FPS
        return (len(self.ts_buf) - 1) / elapsed

    def compute_bpm(self) -> tuple[float | None, float | None]:
        """
        Run CHROM algorithm on the current buffer.

        Returns:
            (bpm, confidence) — both None if not enough data yet.
            confidence is the fraction of power at the peak frequency
            relative to total power in the physiological band (0–1).
        """
        if not self.ready:
            return None, None

        R = np.array(self.r_buf, dtype=np.float64)
        G = np.array(self.g_buf, dtype=np.float64)
        B = np.array(self.b_buf, dtype=np.float64)
        fps = self._estimate_fps()

        # ── Step 1: Normalize each channel by its temporal mean ──────────────
        # This removes the effect of ambient lighting brightness.
        # After this, R/G/B ≈ 1.0 on average; fluctuations encode pulse.
        eps = 1e-6
        Rn = R / (R.mean() + eps)
        Gn = G / (G.mean() + eps)
        Bn = B / (B.mean() + eps)

        # ── Step 2: CHROM projection ──────────────────────────────────────────
        # Xs and Ys are two orthogonal color difference signals.
        # The heartbeat signal lies along a specific direction in (Xs, Ys) space.
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        # alpha scales Ys to match Xs variance, then we subtract to isolate pulse
        alpha = np.std(Xs) / (np.std(Ys) + eps)
        S = Xs - alpha * Ys

        # ── Step 3: Bandpass filter (0.7–4.0 Hz) ─────────────────────────────
        # Removes slow drift (lighting changes) and high-freq noise.
        # Only heartbeat frequencies (42–240 BPM) pass through.
        nyq = fps / 2.0
        low  = FREQ_LOW  / nyq
        high = min(FREQ_HIGH / nyq, 0.99)  # must be < 1 (Nyquist)

        if low >= high:
            return None, None  # fps too low to resolve heartbeat band

        b_coef, a_coef = butter(3, [low, high], btype="band")
        try:
            S_filtered = filtfilt(b_coef, a_coef, S)
        except ValueError:
            return None, None  # filtfilt needs padlen > signal length

        # ── Step 4: FFT → find dominant frequency ─────────────────────────────
        fft_vals = np.abs(np.fft.rfft(S_filtered))
        freqs    = np.fft.rfftfreq(len(S_filtered), d=1.0 / fps)

        # Only consider physiological frequency range
        mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
        if not mask.any():
            return None, None

        band_fft   = fft_vals[mask]
        band_freqs = freqs[mask]

        peak_idx    = np.argmax(band_fft)
        peak_freq   = band_freqs[peak_idx]
        bpm         = float(peak_freq * 60.0)

        # Confidence: how dominant is the peak vs everything else in band?
        peak_power  = float(band_fft[peak_idx])
        total_power = float(band_fft.sum()) + eps
        confidence  = peak_power / total_power

        return round(bpm, 1), round(confidence, 3)

    def is_live(self) -> bool | None:
        """
        Returns:
            True  — physiological pulse signal detected → real face
            False — no pulse / flat signal / out-of-range → likely spoof/photo/screen
            None  — not enough data yet

        Two independent checks must both pass:
          1. Signal variance check: a photo or screen replay has near-zero
             temporal variance in its skin-patch pixel values — there's no
             blood pulsing. Real faces always have measurable oscillation
             (even small). If the green channel std < 0.3 over the window,
             it's almost certainly a flat image.
          2. CHROM + FFT: the dominant frequency must fall in the
             physiological heartbeat band (45–160 BPM) with enough
             peak-to-band-power ratio to be trusted.
        """
        if not self.ready:
            return None

        # ── Check 1: flat-signal detector ────────────────────────────────────
        G = np.array(self.g_buf, dtype=np.float64)
        g_std = float(np.std(G))
        # Normalised std: divide by mean so it's lighting-independent
        g_mean = float(np.mean(G))
        g_cv = g_std / (g_mean + 1e-6)   # coefficient of variation
        # A real face typically has CV > 0.002 (0.2%). Photos are < 0.0005.
        if g_cv < 0.001:
            return False   # flat signal → spoof/still image

        # ── Check 2: physiological frequency check ───────────────────────────
        bpm, confidence = self.compute_bpm()
        if bpm is None or confidence is None:
            return None

        in_range    = LIVE_BPM_LOW <= bpm <= LIVE_BPM_HIGH
        trustworthy = confidence >= MIN_CONFIDENCE
        return in_range and trustworthy
