# CurrentSense Audio MLOps

### Industrial Machine Condition Classification Using Audio and Machine Learning

**Author:** Tifare Kaseke

**Project:** Machine Learning Operations (MLOps) Summative Assignment

**Technologies:** Python · scikit-learn · FastAPI · Streamlit · SQLite · Docker · Nginx · Locust · Render

---

## Video Demonstration

**[▶ Watch the Full Project Demonstration on YouTube](https://youtu.be/sDzIfruq-Fk)**

The video demonstrates the dataset, preprocessing, model training, evaluation, live predictions, additional data uploads, model retraining, Docker scaling, Locust performance testing and cloud deployment.

## Live Application

| Resource | Link |
|---|---|
| **Streamlit Dashboard** | [Launch CurrentSense](https://currentsense-audio-mlops.streamlit.app/) |
| **FastAPI Documentation** | [Explore API Endpoints](https://currentsense-audio-mlops.onrender.com/docs) |
| **API Health Check** | [View Model Status](https://currentsense-audio-mlops.onrender.com/health) |
| **GitHub Repository** | [Source Code](https://github.com/tifarekaseke/currentsense-audio-mlops) |

The application uses a Streamlit frontend connected to a FastAPI inference service deployed on Render.

**Active deployed model:** `audio-v2`

> **Note:** The public application supports audio predictions. Administrative data uploads and retraining are demonstrated in the local environment. Free cloud services may experience startup delays after inactivity.

---

## 1. Project Overview

CurrentSense is an end-to-end Machine Learning Operations application designed to classify industrial fan audio recordings into two operating conditions:

- **Normal:** Audio associated with normal machine operation.
- **Abnormal:** Audio associated with an abnormal machine operating condition.

Industrial machines can produce different acoustic patterns when their operating conditions change. CurrentSense uses machine learning to analyze these patterns and support experimental condition monitoring.

This project extends an earlier motor-condition classification study that used numerical sensor features. The MLOps implementation introduces **non-tabular WAV audio data** and incorporates model deployment, monitoring, additional training data collection, retraining and performance testing.

### Project Objectives

1. Build a reproducible audio preprocessing and classification pipeline.
2. Train, evaluate and save a machine learning model.
3. Provide a user-friendly interface for individual WAV predictions.
4. Support uploading multiple new labeled audio recordings into a database.
5. Enable user-triggered retraining of an existing model.
6. Deploy the application using FastAPI, Streamlit and Docker.
7. Evaluate performance under concurrent requests using Locust.
8. Deploy the inference application to the cloud.

---

## 2. Dataset

**Dataset:** MIMII — Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection.

**Source:** [MIMII Dataset on Zenodo](https://zenodo.org/records/3384388)

The experiment uses real industrial fan recordings from the `-6_dB_fan` subset, machine ID `id_00`.

### Dataset Distribution

| Dataset Split | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Training | 210 | 210 | 420 |
| Validation | 45 | 45 | 90 |
| Testing | 45 | 45 | 90 |
| **Total** | **300** | **300** | **600** |

The dataset was divided at recording level, with balanced labels across the three splits.

All 600 prepared training, validation and test recordings were verified against the original MIMII WAV files.

Ten additional real recordings, consisting of five normal and five abnormal samples, were used separately to demonstrate the retraining workflow.

**Important:** Raw WAV recordings are excluded from GitHub because of their size. The repository includes the dataset preparation script and instructions for reproducing the experiment.

---

## 3. Audio Preprocessing and Feature Engineering

Raw audio is converted into a consistent numerical representation before classification.

The preprocessing workflow includes:

1. WAV decoding and conversion to mono audio.
2. Resampling to 16 kHz.
3. Processing recordings of up to ten seconds.
4. Acoustic feature extraction.
5. Feature standardization using training data statistics.

### Extracted Features

The pipeline extracts **52 acoustic features** from each recording.

| Feature | Description |
|---|---|
| MFCCs | Capture spectral characteristics of audio |
| RMS Energy | Measures audio signal intensity |
| Spectral Centroid | Estimates the center of spectral energy |
| Spectral Bandwidth | Measures spectral spread |
| Spectral Rolloff | Describes the frequency distribution |
| Spectral Flatness | Measures noise-like spectral characteristics |
| Zero-Crossing Rate | Measures how frequently the waveform changes sign |

The notebook includes audio waveform visualizations and feature distribution comparisons for normal and abnormal recordings.

All three feature matrices contain 52 columns, and no missing or infinite feature values were identified.

---

## 4. Machine Learning Model

The classifier is implemented using **scikit-learn's Multilayer Perceptron (MLP)**.

### Model Configuration

| Component | Configuration |
|---|---|
| Algorithm | Multilayer Perceptron |
| Hidden layers | 64 and 32 neurons |
| Optimizer | Adam |
| Input dimensions | 52 features |
| Target classes | Normal and abnormal |
| Feature scaling | StandardScaler |
| Regularization | L2 |
| Training control | Validation-based early stopping |
| Model format | Pickle (`.pkl`) |

A dummy baseline classifier was also evaluated and achieved 50% test accuracy.

The trained model is saved at:

`models/current_sense_audio.pkl`

### Final Model Evaluation

The final `audio-v2` model was evaluated on 90 held-out MIMII recordings.

| Evaluation Metric | audio-v1 | audio-v2 |
|---|---:|---:|
| Test Accuracy | 68.89% | 68.89% |
| Test Precision | 70.73% | 69.77% |
| Test Recall | 64.44% | 66.67% |
| Test F1 Score | 67.44% | 68.18% |
| Test ROC AUC | 0.7457 | 0.7353 |

**Final audio-v2 confusion matrix:**

| Actual / Predicted | Normal | Abnormal |
|---|---:|---:|
| Normal | 32 | 13 |
| Abnormal | 15 | 30 |

The final model correctly classified **62 out of 90 test recordings**.

Retraining maintained the original test accuracy while slightly improving the F1 score. ROC AUC decreased slightly.

These results are experimental and do not establish reliability for unseen machines or real-world fault diagnosis.

Supporting files:

- `outputs/evaluation_audio_v2_test.json`
- `outputs/confusion_matrix_audio_v2_test.png`
- `outputs/audio_v2_test_predictions.csv`

---

## 5. Application Architecture

The application implements the following workflow:

```text
                   WAV Audio Recording
                           |
                           v
                   Streamlit Dashboard
                           |
                           v
                     FastAPI Service
                           |
                           v
                 Audio Feature Extraction
                           |
                           v
                 Saved MLP Model (audio-v2)
                           |
                           v
                 Prediction + Confidence
                           |
                           v
                 Monitoring / SQLite Logs


Administrative Retraining Workflow:

New Labeled WAV Recordings
            |
            v
     Bulk Upload API
            |
            v
      SQLite Storage
            |
            v
    Trigger Retraining
            |
            v
  Update Existing MLP Model
            |
            v
    Validation Evaluation
            |
            v
    Updated Model Version
```

### Main Components

**FastAPI:** Prediction endpoints, health checks, data uploads, retraining, job status and monitoring.

**Streamlit:** Interactive dashboard for WAV predictions, visualizations, model information and local administrative operations.

**SQLite:** Stores prediction records, additional labeled recordings, feedback and retraining job information.

**Docker:** Packages application services into reproducible containers.

**Nginx:** Routes requests and distributes traffic across API replicas.

**Locust:** Simulates concurrent users making audio prediction requests.

---

## 6. Additional Data Upload and Model Retraining

CurrentSense supports updating the existing trained classifier when additional labeled recordings become available.

During the implementation:

- Ten additional real MIMII recordings were uploaded.
- Five recordings were labeled normal.
- Five recordings were labeled abnormal.
- Upload information was stored in SQLite.
- A retraining job was triggered from the dashboard.
- The trained model was updated from `audio-v1` to `audio-v2`.

### Retraining Results

| Metric | Result |
|---|---:|
| New labeled recordings | 10 |
| Updated model version | `audio-v2` |
| Validation accuracy | 71.11% |
| Validation F1 score | 69.77% |
| Job outcome | Completed successfully |

The retraining process continues fitting the existing classifier using the new labeled data and evaluates the resulting model before accepting the update.

**Note:** The retraining validation results are separate from the final held-out test evaluation.

Administrative retraining is disabled on the public cloud service and is demonstrated locally.

---

## 7. Running the Project Locally

### Prerequisites

- Python 3.11
- Git
- Conda or another Python virtual environment
- Docker Desktop for containerized deployment

### Clone the Repository

```bash
git clone https://github.com/tifarekaseke/currentsense-audio-mlops.git
cd currentsense-audio-mlops
```

### Install Dependencies

```bash
conda create -n currentsense-mlops python=3.11 -y
conda activate currentsense-mlops
python -m pip install -r requirements.txt
```

### Prepare the MIMII Dataset

Download and extract the fan recordings from the MIMII dataset.

From the repository root:

```bash
python scripts/prepare_mimii.py --source "PATH_TO_EXTRACTED_-6_dB_fan" --machine-id id_00 --max-per-class 300
```

Train the initial model:

```bash
python -m src.model --data-root data
```

**Note:** Training the initial model writes the saved model file. Back up the existing `audio-v2` model before deliberately retraining it.

### Start FastAPI

In the first terminal:

```bash
python -m uvicorn api:app --port 8000
```

### Start Streamlit

In a second terminal:

```bash
python -m streamlit run app.py
```

Local interfaces:

- Streamlit: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

For local administrative functions, configure a private `ADMIN_TOKEN` consistently for the API and dashboard.

Never commit administrator credentials or the `.env` file.

---

## 8. Docker Deployment

The project includes a `Dockerfile`, Docker Compose configuration and an Nginx gateway.

Create a private `.env` file from the provided example and configure the administrator token.

Start the services:

```bash
docker compose up -d --build
```

Check the running containers:

```bash
docker compose ps
```

Scale the FastAPI service to two replicas:

```bash
docker compose up -d --scale api=2
docker compose restart gateway
```

Scale to three replicas:

```bash
docker compose up -d --scale api=3
docker compose restart gateway
```

The Nginx gateway distributes prediction traffic between available API containers.

**Important:** Retraining should not be initiated during multi-replica inference benchmarking because the prototype shares local model files and SQLite storage.

---

## 9. Locust Load Testing

Locust was used to evaluate the system under concurrent WAV prediction requests.

### Test Configuration

| Parameter | Value |
|---|---|
| Concurrent users | 10 |
| Spawn rate | 2 users/second |
| Test duration | 60 seconds |
| Endpoint | `POST /predict` |
| Payload | WAV audio recording |
| Configurations | 1, 2 and 3 API replicas |

### Recorded Performance

| Metric | 1 Replica | 2 Replicas | 3 Replicas |
|---|---:|---:|---:|
| Requests | 65 | 263 | 263 |
| Failed requests | 0 | 0 | 0 |
| Average response time | 7.60 s | 1.52 s | 1.47 s |
| P95 response time | 12.0 s | 2.7 s | 2.7 s |
| Throughput | 1.12 req/s | 4.45 req/s | 4.64 req/s |

### Interpretation

In the successful experiments, increasing from one to two API replicas improved prediction throughput substantially and reduced average latency.

Scaling from two to three replicas produced a smaller improvement, indicating diminishing returns under this workload.

An earlier two-replica run recorded a 52.63% failure rate due to request timeouts. A subsequent warmed-up run completed without failures. The exact cause of the earlier failures was not conclusively established.

These results reflect local workstation testing and should not be interpreted as guaranteed cloud performance.

### Reproduce a Locust Test

```bash
python -m locust -f locustfile.py --headless --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s --csv outputs/locust_results --only-summary
```

The original experimental results are available in the `outputs/` directory.

---

## 10. Cloud Deployment and Monitoring

CurrentSense has two publicly deployed components.

**Backend:** FastAPI hosted on Render.

https://currentsense-audio-mlops.onrender.com/docs

**Frontend:** Streamlit hosted on Streamlit Community Cloud.

https://currentsense-audio-mlops.streamlit.app/

The deployed backend exposes a health endpoint that reports:

- API availability
- Server uptime
- Model loading status
- Active model version

The public deployment was verified to load `audio-v2` and produce real WAV predictions.

The system also records prediction metadata and latency for basic monitoring.

**Cloud limitations:** Free hosting resources can introduce cold-start delays, temporary storage limitations and lower performance under heavy traffic. Public administrative retraining remains disabled.

---

## 11. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API status, uptime and model version |
| POST | `/predict` | Predict a single WAV recording |
| POST | `/upload` | Upload labeled recordings |
| POST | `/retrain` | Trigger model retraining |
| GET | `/jobs/{job_id}` | Check retraining job status |
| GET | `/stats` | Retrieve monitoring statistics |
| POST | `/feedback` | Submit verified prediction feedback |

Administrative endpoints require an administrator token.

Interactive API documentation is available at:

https://currentsense-audio-mlops.onrender.com/docs

---

## 12. Repository Structure

```text
currentsense-audio-mlops/
│
├── app.py
├── api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── README.md
│
├── notebook/
│   └── currentsense_audio_mlops.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── database.py
│
├── scripts/
│   ├── prepare_mimii.py
│   ├── evaluate_audio_v2.py
│   └── benchmark_summary.py
│
├── models/
│   └── current_sense_audio.pkl
│
├── nginx/
│   └── default.conf
│
├── tests/
│
└── outputs/
    ├── evaluation_audio_v2_test.json
    ├── confusion_matrix_audio_v2_test.png
    └── scaling_comparison.csv
```

The raw dataset, uploaded audio, local databases, private credentials and model backup files are intentionally excluded from version control.

---

## 13. Limitations and Future Improvements

CurrentSense demonstrates a functioning MLOps pipeline, but several improvements remain possible:

- Improve classification accuracy using larger and more diverse audio datasets.
- Evaluate models on unseen machine IDs and recording sessions.
- Investigate convolutional or other deep learning architectures for audio classification.
- Improve the detection of abnormal recordings to reduce false negatives.
- Use durable cloud storage and a dedicated retraining worker.
- Extend monitoring to include data drift and production model performance.
- Perform repeated load tests under controlled cloud infrastructure.

The current classifier is intended for research and educational demonstration, not certified industrial fault diagnosis.

---

## 14. References

**MIMII Dataset:** Purohit, H., et al. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection.* [Zenodo](https://doi.org/10.5281/zenodo.3384388).

**Locust Documentation:** https://docs.locust.io/

**Docker Documentation:** https://docs.docker.com/

**FastAPI Documentation:** https://fastapi.tiangolo.com/

**Streamlit Documentation:** https://docs.streamlit.io/

---

### Author

**Tifare Kaseke**

*CurrentSense Audio MLOps — Machine Learning Operations Summative Project*

**[Live Application](https://currentsense-audio-mlops.streamlit.app/)** · **[Video Demonstration](https://youtu.be/sDzIfruq-Fk)** · **[GitHub Repository](https://github.com/tifarekaseke/currentsense-audio-mlops)**
