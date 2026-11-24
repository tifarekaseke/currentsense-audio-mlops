"""Evaluate the SAVED audio-v2 model on the untouched 90-file MIMII test split.

Does not fit, retrain or modify the model. Produces a version-specific JSON report,
confusion matrix plot, and sample-by-sample predictions for auditing/video selection.
Usage (from repository root): python scripts/evaluate_audio_v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Support: python scripts/evaluate_audio_v2.py from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model, metrics_for
from src.preprocessing import features_for, list_labeled_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root holding train/val/test")
    parser.add_argument("--out-dir", default="outputs", help="Directory for versioned results")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    bundle = load_model()
    if bundle.get("version") != "audio-v2":
        raise SystemExit(
            f"STOP: expected audio-v2, found {bundle.get('version')!r}. "
            "Use the saved retrained model; never retrain on test data."
        )
    if bundle.get("synthetic_fixture", False):
        raise SystemExit("STOP: synthetic demonstration model; real-data evaluation required")

    records = list_labeled_audio(root, splits=("test",))
    n_normal = sum(r["label"] == "normal" for r in records)
    n_abnormal = sum(r["label"] == "abnormal" for r in records)
    if (len(records), n_normal, n_abnormal) != (90, 45, 45):
        raise SystemExit(
            f"STOP: expected original held-out 90 WAV files (45/45); "
            f"found total={len(records)}, normal={n_normal}, abnormal={n_abnormal}. "
            "Confirm you're using the original data/test folder."
        )

    X, y, kept = features_for(records)
    X_scaled = bundle["scaler"].transform(X)
    p_abnormal = bundle["classifier"].predict_proba(X_scaled)[:, 1]
    predicted = (p_abnormal >= 0.5).astype(int)
    metrics = metrics_for(y, p_abnormal)

    details = []
    for record, actual, pred, proba in zip(kept, y, predicted, p_abnormal):
        src = Path(record["path"])
        details.append({
            "file": src.name,
            "test_path": str(src.relative_to(root)),
            "actual_label": "abnormal" if actual else "normal",
            "predicted_label": "abnormal" if pred else "normal",
            "confidence": round(float(max(proba, 1 - proba)), 5),
            "correct": bool(actual == pred),
        })

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": bundle["version"],
        "dataset": "Real MIMII -6_dB_fan id_00",
        "split": "original held-out test (not used for fitting or selection)",
        "records": len(kept),
        "normal_records": n_normal,
        "abnormal_records": n_abnormal,
        "test_metrics": metrics,
        "note": "Evaluation only; existing model weights remain unchanged.",
    }
    report_path = output / "evaluation_audio_v2_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output / "audio_v2_test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    fig, ax = plt.subplots(figsize=(5.7, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=["normal", "abnormal"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title("CurrentSense audio-v2: held-out test")
    fig.tight_layout()
    plot_path = output / "confusion_matrix_audio_v2_test.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("Model:", bundle["version"], "| Test WAVs:", len(kept))
    for metric in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"):
        print(f"{metric:>17}: {metrics.get(metric):.4f}")
    print("Confusion matrix [normal, abnormal]:", metrics["confusion_matrix"])
    print("Results:", report_path, csv_path, plot_path, sep="\n  ")
    for label in ("normal", "abnormal"):
        matches = [r for r in details if r["actual_label"] == label and r["correct"]]
        if matches:
            example = matches[0]
            print(f"Correct {label} demo sample: {example['test_path']} "
                  f"(confidence {example['confidence']:.1%})")
        else:
            print(f"No correctly classified {label} sample in this test split.")


if __name__ == "__main__":
    main()
