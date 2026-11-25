CurrentSense Audio MLOps
Industrial fan condition screening from WAV recordings
FastAPI · Streamlit · scikit-learn · SQLite · Docker · Nginx · Locust

Author: Tifare Kaseke
Repository: https://github.com/tifarekaseke/currentsense-audio-mlops
Live application and video
Resource	Link
Public Streamlit dashboard	https://currentsense-audio-mlops.streamlit.app/
FastAPI interactive documentation	https://currentsense-audio-mlops.onrender.com/docs
API health and active model version	https://currentsense-audio-mlops.onrender.com/health
Camera-on YouTube demonstration	To be added after recording and uploading


Deployment status (25 November 2026): The FastAPI inference API is deployed on Render and the Streamlit dashboard is deployed on Streamlit Community Cloud. The public API has been verified to load audio-v2, and a real abnormal MIMII WAV recording was successfully classified through the public dashboard. The cloud deployment is intended for inference. Administrative labeled-data uploads and retraining are demonstrated locally; no public retraining access is intended.
Overview
CurrentSense is a machine-condition monitoring research prototype that classifies recordings of an industrial fan as normal or abnormal. A user uploads a WAV file to Streamlit, which sends it to FastAPI and displays the predicted class, confidence, model version, response latency, and selected acoustic features.

The project also implements labeled bulk audio upload, SQLite records for prediction and training jobs, and an administrative retraining workflow that updates an existing saved model. It extends an earlier CurrentSense project that used 48 numerical current-signal features and 11 anonymized classes. The audio experiment is a different classification task, so accuracy figures from the earlier work are not directly comparable.
Implemented components
- WAV decoding, resampling, acoustic feature extraction and real-data train/validation/test preparation.
- A saved scikit-learn multilayer perceptron (MLP) binary classifier.
- FastAPI inference, health checks, prediction statistics, labeled upload, retraining jobs and verified-label feedback endpoints.
- Streamlit prediction interface, dataset insights, acoustic feature visualizations and monitoring; administrative workflows are used locally.
- SQLite prediction and upload metadata; Docker Compose with an Nginx gateway and scalable API replicas.
- Locust load tests with real WAV payloads, covering one, two and three API containers.
- Public cloud inference API on Render and dashboard on Streamlit Community Cloud.
Dataset and preparation
Source: MIMII dataset, Purohit et al. (2019). The experiment used the extracted -6_dB_fan recordings from machine ID id_00, with the dataset's normal/abnormal labels.
Split	Normal	Abnormal	Total	Purpose
Training	210	210	420	Model fitting
Validation	45	45	90	Model selection and retraining acceptance
Held-out test	45	45	90	Final performance measurement
Total	300	300	600	Real MIMII recordings


An additional 10 previously unused real WAV files (five normal and five abnormal) were uploaded through the application for the retraining demonstration. They were distinct from the 600-file train/validation/test split. The preparation process was checked against the original extracted MIMII recordings, and the data split contains exact copies of the source WAV files. The separate data/fixture/ folder contains synthetic software-smoke-test recordings and is not used for real-model performance reporting.

Raw WAV files, uploaded recordings and local SQLite data are not committed to the repository. To reproduce the data preparation after downloading and extracting the MIMII fan archive, run from the project root:
python scripts/prepare_mimii.py --source "C:\PATH\TO\-6_dB_fan" --machine-id id_00 --max-per-class 300
python -m src.model --data-root data
Only use the preparation script's --overwrite flag when you intentionally want to replace an existing prepared dataset. The script produces a local dataset manifest with split information and source checksums.

Scientific limitations: The splitting procedure separates recordings, not independent machines or recording sessions. The results therefore do not establish generalization to unseen machines or types of mechanical faults. A normal/abnormal prediction is not a certified diagnosis.
Machine learning workflow
Real MIMII WAV audio (normal / abnormal)
    → Decode, convert to mono, resample to 16 kHz, use up to 10 seconds
    → Extract 52 acoustic features
    → Fit StandardScaler on training data only
    → Train MLP classifier with hidden layers (64, 32)
    → Validation-based early stopping and evaluation
    → Save classifier + scaler
    → FastAPI inference → Streamlit UI and SQLite monitoring
    → Optional local administrative upload + continuing model training
The 52 acoustic features comprise 20 MFCCs summarized by mean and standard deviation (40 numbers), and six additional audio descriptors summarized by mean and standard deviation (12 numbers): RMS energy, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness and zero-crossing rate. The MLP uses Adam optimization, L2 regularization and validation-based early stopping. The notebook includes waveform exploration, three acoustic-feature comparisons, baseline prediction, model evaluation, a confusion matrix and training-history visualizations.
Measured model results
Held-out test results: audio-v1 and audio-v2
Both versions were evaluated against the same 90 held-out real MIMII WAV files.
Test metric	Initial audio-v1	Retrained audio-v2
Accuracy	68.89%	68.89%
Precision, abnormal class	70.73%	69.77%
Recall, abnormal class	64.44%	66.67%
F1, abnormal class	67.44%	68.18%
Balanced accuracy	68.89%	68.89%
ROC AUC	0.7457	0.7353


The audio-v2 held-out confusion matrix (rows = actual normal, abnormal; columns = predicted normal, abnormal) is:
[[32, 13],
 [15, 30]]
This corresponds to 62 correct predictions out of 90. Retraining maintained accuracy and slightly improved the abnormal-class F1 score, but ROC AUC decreased. It would be inaccurate to claim that the retrained model improved every evaluation metric.
Retraining job: validation metrics
Ten previously unused and correctly labeled recordings were added through the local application. The retraining job completed successfully, incorporated the new samples, and updated the serving version to audio-v2. Its separate validation scores were:
Validation metric, audio-v2	Value
Accuracy	71.11%
Precision, abnormal class	73.17%
Recall, abnormal class	66.67%
F1, abnormal class	69.77%
ROC AUC	0.7472


These validation scores are not the final test scores. The independent audio-v2 test evaluation is recorded in outputs/evaluation_audio_v2_test.json, outputs/audio_v2_test_predictions.csv and outputs/confusion_matrix_audio_v2_test.png. The original outputs/evaluation.json may contain the audio-v1 training run; verify model version before citing it.

One held-out abnormal recording, id_00_00000008.wav, was correctly classified as abnormal with 95.5% confidence by the deployed audio-v2 model. This confidence is for that individual prediction, not overall classification accuracy.
Run locally
Requirements: Python 3.11, dependencies from requirements.txt, and a saved real-data model at models/current_sense_audio.pkl. For retraining or complete data-reproduction workflows, prepare the source MIMII recordings as described above.
conda create -n currentsense-mlops python=3.11 -y
conda activate currentsense-mlops
python -m pip install -r requirements.txt
Run the API in terminal A:
conda activate currentsense-mlops
$env:ADMIN_TOKEN = "REPLACE_WITH_YOUR_PRIVATE_LOCAL_TOKEN"
python -m uvicorn api:app --port 8000
Run the Streamlit dashboard in terminal B, using the same private token only for your trusted local administration demo:
conda activate currentsense-mlops
$env:ADMIN_TOKEN = "REPLACE_WITH_YOUR_PRIVATE_LOCAL_TOKEN"
$env:API_URL = "http://localhost:8000"
python -m streamlit run app.py
- Local dashboard: http://localhost:8501
- Local FastAPI docs: http://localhost:8000/docs
- Local health check: http://localhost:8000/health
Verify the active model:
Invoke-RestMethod "http://localhost:8000/health" | Format-List
Expect status: online, model_loaded: True and the saved model's version. Do not expose the administrator token in a public Streamlit app.
API reference
Method	Endpoint	Purpose
GET	/health	Service status, uptime, loaded model and version
POST	/predict	Upload a single WAV and obtain a prediction
POST	/upload	Save multiple labeled WAV recordings
POST	/retrain	Trigger existing-model retraining
GET	/jobs/{job_id}	Check retraining job state and evaluation summary
GET	/stats	Request statistics and inference latency information
POST	/feedback	Attach a verified label to a stored prediction


Administrative endpoints require X-Admin-Token, configured using the private ADMIN_TOKEN environment variable. The public inference deployment should not expose administrative controls or secrets.
Docker and horizontal scaling
Create a private .env from .env.example, configure a secret ADMIN_TOKEN, and ensure the trained model exists. The Docker Compose configuration mounts local models, data and outputs, rather than adding private WAV files or databases to the image.
Copy-Item .env.example .env
# Edit .env and provide a private ADMIN_TOKEN

docker compose up -d --build
docker compose ps
Invoke-RestMethod "http://localhost:8000/health" | Format-List
The Compose services are api (FastAPI), gateway (Nginx) and ui (Streamlit). The gateway publishes port 8000; the UI publishes port 8501. To change replica count:
docker compose up -d --scale api=2
docker compose restart gateway
# Or increase to three replicas:
docker compose up -d --scale api=3
docker compose restart gateway
docker compose ps
Nginx logs showed successful /predict responses from api-1, api-2 and api-3 during the three-replica run. Replicas share local storage and a model file in this prototype; avoid initiating retraining while multiple inference replicas serve traffic. A production system would need coordinated model rollout, durable storage, dedicated retraining workers and stronger access controls.
Locust flood-request simulation
Load tests submitted real WAV files through the Nginx gateway using 10 concurrent Locust users, 2 users/second spawn rate, and a 60-second run time. The original results are retained in outputs/.
python -m locust -f locustfile.py --headless --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s --csv outputs/locust_1container_10users_threaded --only-summary
After scaling to two or three API replicas, repeat using a different CSV output prefix. Warm up the replicas and use consistent sample WAV, user count, run duration and timeout for comparable tests.

Successful 10-user runs
API replicas	Completed requests	Failure rate	Mean latency	P95 latency	Throughput
1	65	0%	7,595 ms	12,000 ms	1.12 req/s
2	263	0%	1,516 ms	2,700 ms	4.45 req/s
3	263	0%	1,469 ms	2,700 ms	4.64 req/s


In these local runs, increasing replicas improved throughput and reduced mean response time, with diminishing gains from two to three. The three-replica maximum observed latency (9.46 seconds) exceeded the successful two-replica maximum (3.20 seconds), despite similar P95 performance. These are workstation measurements, not guarantees of cloud capacity.

Failures and test variability: An earlier two-container test recorded 19 requests, 10 failures (52.63%), 22.57 seconds mean latency and 0.32 req/s. A three-user test and a later ten-user repeat succeeded without failures. The exact reason for the earlier failure was not proven; warm-up and resource contention are possible contributing factors. An initial one-container test also timed out before a worker-thread concurrency change. Reporting both passing and failing runs is necessary for an honest performance assessment.
Monitoring and model limitations
The dashboard displays service uptime, request and prediction information, inference latency and audio-feature insights. RMS describes signal energy, spectral centroid summarizes frequency-weighted spectral energy and zero-crossing rate describes sign-change activity. These values alone do not prove a mechanical fault.

The /feedback endpoint can record independently verified labels. HTTP 200 and model confidence indicate successful inference and the classifier's output, not ground-truth correctness. The observed held-out test accuracy is 68.89%; the system is a research prototype and has not been validated across independent machines, sites or operating conditions.

Cloud hosting limitation: The free hosting services have resource constraints and may start slowly after inactivity. Their local files or databases are not suitable for durable public retraining records. The validated public workflow is inference, with local administrative retraining used for the assessment demonstration.
Repository structure
.
├── api.py
├── app.py
├── locustfile.py
├── Dockerfile
├── docker-compose.yml
├── nginx/default.conf
├── requirements.txt
├── .env.example
├── notebook/currentsense_audio_mlops.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── database.py
├── scripts/
│   ├── prepare_mimii.py
│   ├── evaluate_audio_v2.py
│   ├── make_synthetic_smoke_data.py
│   ├── benchmark_summary.py
│   └── run_load_tests.sh
├── tests/
├── models/current_sense_audio.pkl
└── outputs/
The raw WAV dataset, uploaded training audio, SQLite files, private .env and model backups are excluded from version control. The synthetic data script is intended for software smoke testing, not for reporting model accuracy.


References
- Purohit, H., et al. (2019). MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. Zenodo, DOI 10.5281/zenodo.3384388.
- Purohit, H., et al. (2019). MIMII Dataset. arXiv:1909.09347.
- Locust documentation.
- Docker Compose documentation.