"""Extract numerical acoustic descriptors FROM audio waveforms, not tabular source rows.
All files are decoded and assessed as 16 kHz mono WAV, max 10 seconds.
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
MAX_SECONDS = 10
N_MFCC = 20
DESCRIPTORS = ["rms", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "zero_crossing_rate"]
FEATURE_NAMES = [f"mfcc_{i:02d}_{s}" for i in range(N_MFCC) for s in ("mean", "std")]
FEATURE_NAMES += [f"{name}_{s}" for name in DESCRIPTORS for s in ("mean", "std")]
FEATURE_VERSION = "acoustics-v1-16khz-mfcc20"


def decode_audio(file_or_bytes):
    if isinstance(file_or_bytes, (bytes, bytearray)):
        source = io.BytesIO(file_or_bytes)
    else:
        source = str(file_or_bytes)
    y, sr = sf.read(source, dtype="float32", always_2d=True)
    if y.shape[0] < 1024 or sr < 2000 or sr > 192000:
        raise ValueError("Invalid or insufficient audio; provide a genuine WAV recording")
    if y.shape[0] / sr > 120:
        raise ValueError("Recording exceeds 120 second safety limit")
    if not np.isfinite(y).all():
        raise ValueError("Audio contains invalid sample values")
    y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = np.ascontiguousarray(y[: SAMPLE_RATE * MAX_SECONDS], dtype="float32")
    if len(y) < 512:
        raise ValueError("Audio too short for acoustic features")
    return y


def compute_features(source):
    """Return one 52-dimensional vector (20 MFCC means/stds + 6 acoustic statistics)."""
    y = decode_audio(source)
    n_fft = 1024
    hop = 256
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                 n_fft=n_fft, hop_length=hop)
    vals = []
    for row in mfcc:
        vals.extend([float(np.mean(row)), float(np.std(row))])
    a = {
        "rms": librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop),
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop),
        "spectral_flatness": librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop),
    }
    for name in DESCRIPTORS:
        vals.extend([float(np.mean(a[name])), float(np.std(a[name]))])
    out = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(out).all() or len(out) != len(FEATURE_NAMES):
        raise ValueError("Feature extraction failed")
    return out


def list_labeled_audio(root: str | Path, splits=("train", "val", "test")):
    root = Path(root)
    records = []
    for split in splits:
        for label in ("normal", "abnormal"):
            for p in sorted((root / split / label).rglob("*.wav")):
                records.append({"path": str(p.resolve()), "label": label, "split": split})
    return records


def features_for(records):
    features, labels, kept = [], [], []
    for r in records:
        try:
            features.append(compute_features(r["path"]))
            labels.append(1 if r["label"] == "abnormal" else 0)
            kept.append(r)
        except Exception as ex:
            raise RuntimeError(f"Audio processing failed for {r['path']}: {ex}") from ex
    if not features:
        raise ValueError("No labeled .wav recordings found")
    return np.stack(features), np.asarray(labels, dtype=int), kept
