# CurrentSense Audio MLOps — Final Validation Report

**Author:** Tifare Kaseke  
**Dataset:** Real MIMII industrial fan WAV recordings, `-6_dB_fan`, machine `id_00`  
**Active deployed model:** `audio-v2`  
**Repository:** https://github.com/tifarekaseke/currentsense-audio-mlops

## 1. Data authenticity and evaluation protocol

The project prepared **600 real MIMII WAV files**: 300 normal and 300 abnormal. The split was stratified at recording level and the prepared recordings were checked against the downloaded source WAV files using SHA-256 hashes.

| Split | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Training | 210 | 210 | 420 |
| Validation | 45 | 45 | 90 |
| Held-out test | 45 | 45 | 90 |
| **Total** | **300** | **300** | **600** |

The notebook executed feature extraction on the real split: **52 numerical audio features** per recording, with no missing or infinite feature values reported. Visual evidence includes normal/abnormal waveforms and comparisons of RMS energy, spectral centroid and zero-crossing rate.

The repository also contains a separate **64-recording synthetic smoke-test fixture** for software checks. These synthetic WAV files were **not** used for the real-data training or reported evaluation scores. An additional **10 real MIMII recordings** (five per class), separate from the 600-recording split, were uploaded for the retraining demonstration.

**Methodological limitation:** A split by recording does not independently establish performance on previously unseen physical machines or recording sessions.

## 2. Model development

The pipeline converts WAV audio to mono, resamples to 16 kHz, uses up to ten seconds, and extracts MFCC and acoustic descriptors. It fits a training-only `StandardScaler` and uses a scikit-learn MLP classifier with hidden layers `(64, 32)`, Adam optimization, L2 regularization and validation-based early stopping.

The executed notebook includes a majority-class dummy baseline with **50% test accuracy** and a real-data training run that saved initial model `audio-v1`.

### Initial `audio-v1` held-out test results

| Metric | Result |
|---|---:|
| Accuracy | 68.89% |
| Precision (abnormal) | 70.73% |
| Recall (abnormal) | 64.44% |
| F1 (abnormal) | 67.44% |
| ROC AUC | 0.7457 |

**Evidence:** `notebook/currentsense_audio_mlops.ipynb`, `outputs/evaluation.json`, `src/model.py`.

## 3. Upload and incremental retraining

The local application accepted **10 additional verified-label WAV uploads** (5 normal and 5 abnormal), recorded their metadata in SQLite, and ran a retraining job against the existing saved model. The job completed and produced **`audio-v2`**.

### `audio-v2` validation results *during retraining*

| Metric | Result |
|---|---:|
| Accuracy | 71.11% |
| Precision (abnormal) | 73.17% |
| Recall (abnormal) | 66.67% |
| F1 (abnormal) | 69.77% |
| ROC AUC | 0.7472 |
| New labeled recordings incorporated | 10 |

**Evidence:** `outputs/last_retraining.json`, local dashboard / upload and job-status functionality. These are **validation** results, not held-out test results.

## 4. Final `audio-v2` held-out test evaluation

The retrained saved model was subsequently tested on the **90 original held-out WAV recordings**, which were not used for training or the retraining update.

| Metric | Result |
|---|---:|
| Accuracy | **68.89%** |
| Precision (abnormal) | **69.77%** |
| Recall (abnormal) | **66.67%** |
| F1 (abnormal) | **68.18%** |
| Balanced accuracy | **68.89%** |
| ROC AUC | **0.7353** |

**Confusion matrix** (rows = actual class; columns = predicted class, in order normal then abnormal):

| Actual / Predicted | Normal | Abnormal |
|---|---:|---:|
| Normal | 32 | 13 |
| Abnormal | 15 | 30 |

The final model classified **62 of 90** test WAVs correctly, including **30 of 45 abnormal** recordings. Compared with `audio-v1`, `audio-v2` maintained test accuracy, improved abnormal-class F1 slightly (67.44% to 68.18%) and had a lower ROC AUC (0.7457 to 0.7353). No universal performance improvement is claimed.

**Evidence:** `outputs/evaluation_audio_v2_test.json`, `outputs/confusion_matrix_audio_v2_test.png`, `outputs/audio_v2_test_predictions.csv`, `scripts/evaluate_audio_v2.py`.

## 5. FastAPI, Streamlit and cloud verification

- Local FastAPI and Streamlit interfaces were exercised with real WAV files.
- The cloud FastAPI service was deployed to **Render** and reported `status: online`, `model_loaded: true`, `model_version: audio-v2` at `GET /health`.
- The public **Streamlit Community Cloud** dashboard successfully submitted `id_00_00000008.wav` and received an **ABNORMAL** prediction from `audio-v2` with **95.5% confidence**. This is one correct example, not an overall accuracy estimate.
- Administrative upload and retraining functionality was demonstrated in the local environment. Public deployment is configured for inference, not open retraining.

**Public links:**

- Dashboard: https://currentsense-audio-mlops.streamlit.app/
- API documentation: https://currentsense-audio-mlops.onrender.com/docs
- Health: https://currentsense-audio-mlops.onrender.com/health

Cloud-free-tier cold starts, storage persistence and limited computational resources remain operational constraints. No sustained cloud load test or verified production ground-truth accuracy is claimed.

## 6. Docker and Locust performance testing

Docker Compose and Nginx were used locally with **one, two and three FastAPI replicas**. Nginx logs showed requests handled by multiple replicas. Locust sent real WAV file submissions to `POST /predict` with **10 simulated users**, **2 users/second spawn rate**, and **60-second runs**.

### Successful recorded runs

| API replicas | Requests | Failed | Average latency | P95 latency | Requests/s |
|---|---:|---:|---:|---:|---:|
| 1 | 65 | 0 | 7,595 ms | 12,000 ms | 1.12 |
| 2 | 263 | 0 | 1,516 ms | 2,700 ms | 4.45 |
| 3 | 263 | 0 | 1,469 ms | 2,700 ms | 4.64 |

There was also an **earlier failed two-replica run**: 19 requests, 10 failures (**52.63%**), average latency **22.57 s**, throughput **0.32 requests/s**. A subsequent successful run indicates variability; the exact reason for the earlier failures was not established. Original failure and successful-run CSVs are retained rather than reporting only the best outcome.

**Evidence:** `locustfile.py`, `outputs/scaling_comparison.csv`, individual `outputs/locust_*_stats.csv` and failure CSV files.

## 7. Automated tests and reproducibility

The repository includes automated preprocessing and API tests in `tests/`. The test suite is designed to check that WAV feature extraction yields 52 finite features, the health endpoint responds, and unauthorized upload attempts are rejected. Run in the configured Python environment to verify the current code:

```powershell
conda activate currentsense-mlops
python -m pytest -q
```

The executed Jupyter notebook records the real dataset counts, feature exploration, baseline classifier, initial model training and metrics. Raw MIMII files and private `.env` secrets are excluded from GitHub. Dataset preparation and retraining steps are documented in `README.md`.

## 8. Conclusion and limitations

CurrentSense demonstrates an end-to-end educational MLOps pipeline: genuine WAV ingestion, acoustic feature engineering, offline model training, held-out evaluation, user predictions, SQLite-backed labeled uploads, update of the existing model, Docker/Nginx scaling experiments, monitoring and public cloud inference.

The **68.89% held-out test accuracy** is a prototype benchmark on a particular MIMII fan/machine subset. False negatives and limited cross-machine validation mean the model is **not suitable as a certified diagnostic system**. Cloud persistence, retraining-worker isolation and broader evaluation are future improvements.

**Video demonstration:** https://youtu.be/sDzIfruq-Fk (verify visibility/access before submission).
