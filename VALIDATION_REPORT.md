# Local Validation Report (Nov 15, 2026)

**Scope:** Only code-flow testing on synthetic diagnostic WAV fixtures. This is not an empirical result for the MIMII industrial sound dataset and not a cloud benchmark.

- `pytest -q`: **3 tests passed**. One feature-extraction test and two API behavior tests.
- Generated PCM WAV smoke fixtures with `scripts/make_synthetic_smoke_data.py`.
- Trained an initial custom MLP on those synthetic fixtures; model file created and reloaded.
- FastAPI `GET /health` returned HTTP 200 with live uptime and model version `audio-v1`.
- `POST /predict` accepted a valid WAV, returned a label, confidence, model version, feature highlights and measured request latency.
- `POST /upload` accepted an authenticated, correctly labeled WAV and stored its metadata in SQLite.
- `POST /retrain` returned HTTP 202, background job reported `complete`, and the incrementally trained model advanced to `audio-v2` in `/health`.
- Model and test database were reset after the integration test. Current model package is marked **`synthetic_fixture=True`**, and the dashboard shows an explicit warning. Synthetic data and resulting 100% fixture scores must NOT be used as coursework evaluation findings.

## Not yet verified

- Training and held-out evaluation using REAL MIMII source recordings.
- Docker Compose deployment and replica scaling.
- Live/cloud API uptime and connectivity.
- Locust traffic simulations and measured request latency at each replica count.
- Actual ground-truth-labeled predictions in production.
- Camera-on demo and final GitHub/Canvas submissions.

**Interpretation:** A complete local smoke-test path has been exercised, but the submission is NOT academically complete until the real-data and real-deployment evidence is generated.

## Final audio-v2 held-out test evaluation

The final model was evaluated on 90 previously held-out MIMII
industrial fan recordings, comprising 45 normal and 45 abnormal
samples.

| Metric | Result |
|---|---:|
| Accuracy | 68.89% |
| Precision | 69.77% |
| Recall | 66.67% |
| F1 score | 68.18% |
| Balanced accuracy | 68.89% |
| ROC AUC | 0.7353 |

Confusion matrix (rows: actual normal, abnormal;
columns: predicted normal, abnormal):

[[32, 13], [15, 30]]

Compared with audio-v1, audio-v2 maintained test accuracy
at 68.89%, increased F1 from 67.44% to 68.18%, and
decreased ROC AUC from 0.7457 to 0.7353.

The evaluation uses real MIMII recordings. These results
should not be interpreted as validated performance on
other machine types or operating conditions.

