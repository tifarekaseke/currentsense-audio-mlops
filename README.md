# CurrentSense MLOps: Acoustic Motor Condition Classification

**Status:** runnable project implementation; **not yet an evidence-complete assignment submission**. Real MIMII training, public cloud deployment, measured Docker scaling, and the camera-on video require execution by the student. Never present synthetic smoke-test measurements as real industrial findings.

## 1. Project and relationship to the previous summative

The previous **CurrentSense** ML summative classified 11 anonymized motor operating conditions from 48 already-extracted numerical current-signal features. The original report and code are at [CurrentSense-Motor-Fault-Classification](https://github.com/tifarekaseke/CurrentSense-Motor-Fault-Classification). The strongest former Extra Trees model reached approximately 99.97% test accuracy *on the UCI tabular dataset*. That result is **not a result for the audio model in this project**.

This **new** summative preserves the machine-condition-monitoring use case while changing the input and classification task: raw non-tabular `.wav` industrial fan/motor sound recordings, classified as **normal / abnormal**, plus deployment, monitoring, a prediction API, retraining, Docker, and load tests. The old numerical current-data model is not reused as an audio model because the input domains are different.

## 2. Real dataset and scientific limitations

**MIMII public 1.0**, Hitachi/Purohit et al., https://zenodo.org/records/3384388, DOI [10.5281/zenodo.3384388](https://doi.org/10.5281/zenodo.3384388). This has eight-channel, 16 kHz, ten-second industrial audio recordings of normal and anomalous operating conditions for fans, pumps, valves, and slide rails. This project is configured for a **single fan machine ID, `id_00`**, with binary labels; other source IDs or types would require thoughtful distribution-shift validation. The source dataset is **CC BY-SA 4.0**. Attribution is required when using or redistributing derived content; refer to the original dataset license.

Do not download all 100 GB of MIMII. Choose a **fan** ZIP (e.g. `6_dB_fan.zip`) on Zenodo and extract it on your own machine or cloud VM. The single archive can still exceed 7 GB. This ZIP submission intentionally does not redistribute the raw dataset.

> **Limitations:** The MIMII recording-level split is not machine/session-independent. A high test score does not establish performance on previously unseen motors or operating conditions. Classes are normal vs. abnormal, **not specific mechanical fault diagnoses**. Performance during deployment should be evaluated from independently labeled feedback, not inferred from confidence scores.

## 3. Repository layout

```text
CurrentSense_MLOps/
├── README.md
├── SUBMISSION_STATUS.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── nginx/default.conf
├── app.py                    # Streamlit operator dashboard
├── api.py                    # FastAPI inference & admin endpoints
├── locustfile.py             # Locust load generator
├── notebook/
│   └── currentsense_audio_mlops.ipynb
├── src/
│   ├── preprocessing.py     # WAV decode, mono conversion, MFCC/spectral features
│   ├── model.py             # Saved model, training, metrics and warm-start retraining
│   ├── prediction.py
│   └── database.py          # SQLite audit, uploads, jobs, prediction feedback
├── scripts/
│   ├── prepare_mimii.py
│   ├── make_synthetic_smoke_data.py
│   ├── run_load_tests.sh
│   └── benchmark_summary.py
├── tests/
├── data/
│   ├── train/{normal,abnormal}/
│   ├── val/{normal,abnormal}/
│   ├── test/{normal,abnormal}/
│   └── uploads/
├── models/
│   └── current_sense_audio.pkl  # Created by a genuine training run
└── outputs/
    └── evaluation.json / confusion_matrix.png / three_audio_features.png
```

## 4. Local setup

Use Python 3.11, tested dependency versions specified in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Download real data** from the original Zenodo record and extract a fan ZIP. Prepare labeled, stratified train/validation/test recordings (recordings are partitioned **before** audio feature extraction; a file appears in only one split):

```bash
python scripts/prepare_mimii.py --source /PATH/TO/extracted/6_dB_fan --machine-id id_00 --max-per-class 120
```

The argument should point at a folder that contains paths like `id_00/normal/00000000.wav` and `id_00/abnormal/00000000.wav` beneath it. The script searches recursively. It samples up to 120 real recordings per class for a reasonable first experiment. For a stronger experiment, increase the sample size; always use the original normal/abnormal labels and document your subsampling. It writes `data/dataset_manifest.json` with SHA256 identifiers and the exact split.

**Train the custom model offline, preserve the model file and evaluate on held-out test recordings:**

```bash
python -m src.model --data-root data
```

This creates `models/current_sense_audio.pkl`, `outputs/evaluation.json`, confusion matrix, learning curve, and three feature-distribution plots. All classification metrics are computed from predictions; values are never prefilled. Open and run `notebook/currentsense_audio_mlops.ipynb` from the repository root to display acquisition, preprocessing, training and evaluation evidence.

### Optional synthetic smoke test (not scientific evidence)

If you cannot immediately download the real dataset, you can verify the **software wiring only**:

```bash
python scripts/make_synthetic_smoke_data.py
python -m src.model --data-root data/fixture --synthetic-fixture
```

The API and dashboard will visibly warn that this is a synthetic fixture model. **These fixture metrics must never be submitted as empirical MIMII/model results.** Training with a real dataset overwrites the smoke-test model and removes the warning.

### Run the web UI and API

```bash
# Terminal 1
export ADMIN_TOKEN='CHANGE_TO_LONG_RANDOM_SECRET'  # Windows PowerShell: $env:ADMIN_TOKEN='...'
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2
export ADMIN_TOKEN='CHANGE_TO_LONG_RANDOM_SECRET'
export API_URL='http://localhost:8000'
streamlit run app.py
```

Open **http://localhost:8501** for prediction, visualization, bulk WAV upload, retraining and monitoring. API Swagger UI is at **http://localhost:8000/docs**.

**API endpoints:**

| Verb | Endpoint | Function |
|---|---|---|
| GET | `/health` | API uptime, online state, model readiness/version |
| POST | `/predict` | Upload one WAV and classify normal/abnormal |
| POST | `/upload` | Upload multiple WAV files + accurate label, save on disk and in SQLite |
| POST | `/retrain` | Queue incremental training from *existing saved weights*, using original and newly uploaded labeled data |
| GET | `/jobs/{id}` | Retraining job status, success or failure |
| GET | `/stats` | Test metrics, three feature interpretations, request latency and labeled production feedback |
| POST | `/feedback` | Record independently confirmed actual class for a known prediction; compute production accuracy/F1/etc. |

Use `X-Admin-Token` for `/upload`, `/retrain` and `/feedback` (set `ADMIN_TOKEN`). This is a coursework prototype; for public deployment add HTTPS, stronger access controls, rate limits and centralized database storage.

**Example prediction with a REAL held-out WAV**:

```bash
curl -F "file=@data/test/normal/YOUR_WAV.wav" http://localhost:8000/predict
```

**Upload labeled files and trigger retraining**:

```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" -F "label=normal" \
  -F "files=@path/NEW_NORMAL.wav" http://localhost:8000/upload
curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" http://localhost:8000/retrain
# Poll GET /jobs/JOB_ID and then GET /health for the new model version
```

**What counts as retraining?** The saved MLP has learned weights; its `.partial_fit` method is called for further training on the original training split plus labeled new uploads. The initial fitted scaler stays unchanged so the old weights retain the same coordinate system. Validation monitors early stopping and prevents a >0.05 absolute F1 drop; **held-out test is not used to approve a retrain**. Uploaded files are marked `trained` only after a successful atomic model save. The web UI is able to initiate this workflow on demand.

## 5. The required three feature interpretations

The dashboard calculates **training-set group statistics separately by class**, rather than displaying generic unsupported claims.

1. **Average RMS energy**: how powerful/loud the signal is over time. Compare normal vs abnormal average distributions. A larger RMS is not by itself proof of a defect.
2. **Mean spectral centroid**: a broad measure of where spectral energy is concentrated. A higher value generally suggests proportionally stronger high-frequency sound.
3. **Mean zero-crossing rate**: how often the signal changes sign. This can reflect high-frequency content or noise; interpret only in the context of observed class distributions.

Inspect the actual class means and plots after the real MIMII training run. For coursework, add two short numerical observations per feature based on those measured values.

## 6. Training notebook, optimization, and evaluation

- Non-tabular WAV ingestion, channel averaging, resampling to 16 kHz and at most 10 seconds.
- 20 MFCC mean + standard deviation pairs, and mean/std for six acoustic descriptors (52 numeric features **computed from each waveform**).
- StandardScaler fitted on **training recordings only**, then applied without refitting to validation and test.
- Custom sklearn MLP (`64,32` hidden units), Adam optimizer, L2 regularization `alpha=0.01`, validation-loss early stopping and best weight restoration.
- Accuracy, precision, recall, F1, balanced accuracy, ROC AUC when applicable, log loss, classification report and test confusion matrix.
- Classifier is saved in `models/current_sense_audio.pkl`; its fitted scaler, feature version and metrics are saved in the same file.

## 7. Dockerized deployment

Create a secret `.env`:

```bash
cp .env.example .env
# Edit ADMIN_TOKEN to a long random secret
# Train on real data FIRST, so models/current_sense_audio.pkl exists

docker compose up -d --build
```

- **Web dashboard:** http://localhost:8501
- **Prediction API (through Nginx):** http://localhost:8000/docs
- API containers are designed to scale via Docker Compose and Nginx Docker DNS.

**Cloud platform choice: AWS EC2 (Ubuntu Docker host)**. Create an EC2 VM with adequate CPU, RAM and disk for the model/data. Install Docker Engine and the Compose plugin, clone/upload the GitHub repository, create `.env`, provide real WAV training data and train once offline, then run `docker compose up -d --build`. Expose only HTTPS through a proxy or TLS load balancer in a real public deployment; port mappings above are for the classroom demonstration. Set `RETRAIN_ENABLED=false` while load-testing replicas; perform retraining with a single designated writer/worker. SQLite and a shared disk are suitable for small coursework prototypes, not fully distributed retraining.

**No claim is made here that an EC2 deployment has been executed.** Record the actual public URL and deployment evidence after provisioning your instance.

## 8. Locust flood simulation and Docker scaling

Start from a **real trained model**. Use a known WAV from `data/test`, and execute the provided script on a Linux/macOS Docker host:

```bash
export LOCUST_SAMPLE_WAV=data/test/normal/YOUR_REAL_TEST_WAV.wav
bash scripts/run_load_tests.sh
```

It measures **1, 2, and 3 inference API replicas**, each at the same **10, 25, and 50 concurrent users**, for **60 seconds** per workload. The exported actual Locust CSVs record throughput (requests/s), total requests, failures, average, median, 95th and 99th percentile response times. `python scripts/benchmark_summary.py` generates `outputs/locust_comparison.csv`. For fair comparison, use consistent machine resources, WAV payload, Locust spawn rate and test duration; no model training should take place during the load test.

**Actual measured scaling results:** PENDING USER EXECUTION. **Do not invent numbers** in the README or the video. Once measured, paste the real table and discuss whether latency decreased as replica count increased, and whether rate limiting, CPU contention, model loading or disk IO limited scaling.

## 9. Production evaluation, monitoring and retraining

`/health` reports live availability, process uptime and loaded model version; `/stats` reports number of requests, measured API latency, model version, held-out test metrics, recent inference logs and feedback-based production metrics when verified labels exist. To test model quality *in production*, obtain actual ground-truth labels for deployed predictions and submit `/feedback`; aggregate accuracy, precision, recall and F1 are then computed from these verified examples. Confidence alone is not real production accuracy.

## 10. Camera-on demo (required, 5 marks)

Record a video **with your face/camera and clear audio**, ideally 4–7 minutes:

1. Introduce CurrentSense, its original numeric-current summative and the **new non-tabular audio** extension.
2. Show real training WAV files organized into `normal` and `abnormal`, and the executed notebook + at least four evaluation metrics.
3. Show `/health`, model uptime, dashboard charts and interpretations of **three real acoustic features**.
4. Upload a known test WAV and verify its prediction against the source label.
5. Upload multiple *new, correctly labeled* WAVs, confirm the saved count in SQLite, click **Trigger model retraining**, poll to `complete` and show the model version update.
6. Show the public cloud deployment address, Docker containers and real Locust comparisons for 1/2/3 replicas.
7. End with limitations and an honest discussion of deployment metrics.

Add the actual YouTube link here after recording: **PENDING**.

## 11. Submission order

- **Attempt 1:** upload the entire source repository as a `.zip` file (minus secret credentials and extremely large raw audio). Ensure the trained model file and actual evaluation artefacts are included when you package for Canvas.
- **Attempt 2:** submit your **own** GitHub repository URL, with `README.md` updated to include the actual deployment URL, actual Locust results and the recorded YouTube demo.

## 12. Sources and attribution

- Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., & Kawaguchi, Y. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*. Zenodo. https://doi.org/10.5281/zenodo.3384388
- Purohit et al. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*. https://arxiv.org/abs/1909.09347
- Tifare Kaseke (2026). *CurrentSense: Comparative Classification of Electric Motor Conditions Using Traditional Machine Learning and Deep Learning*. Previous ML Summative, original GitHub repository: https://github.com/tifarekaseke/CurrentSense-Motor-Fault-Classification
- Locust documentation: https://docs.locust.io/

**Academic integrity:** Confirm that all explanations match your actual executed experiments. Cite your original project and any dataset/code sources; do not report a synthetic smoke test as an industrial experiment, and do not report deployment or load test results that were not measured.

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

