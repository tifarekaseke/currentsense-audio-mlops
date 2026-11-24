# CurrentSense MLOps — execution and submission status

This is a **code package**, not a falsely completed lab report. The tasks below distinguish implemented source from witnessed execution.

| Requirement | Source implemented? | Executed against real MIMII / live cloud? | Evidence location |
|---|---|---|---|
| Non-tabular WAV classification | Yes | Pending download / real training | src/preprocessing.py, src/model.py |
| Offline custom model and persistence | Yes | Pending real audio experiment | notebook/, models/ after training |
| Four or more metrics | Yes | Pending real training | outputs/evaluation.json after training |
| Upload + database | Yes | Local tests possible | api.py, src/database.py |
| Retrain existing trained model | Yes | Pending real-audio end-to-end demonstration | src/model.py, UI |
| Streamlit UI and uptime | Yes | Local tests possible | app.py, /health |
| Cloud deployment | Configuration supplied | NOT EXECUTED | Dockerfile, docker-compose.yml |
| Docker replicas / load tests | Locust and script supplied | NOT EXECUTED | locustfile.py, scripts/run_load_tests.sh |
| Production evaluation | API feedback mechanism implemented | No verified production labels | /feedback and /stats |
| Camera-on video | Storyboard supplied | NOT RECORDED | README section 10 |
| GitHub URL | Repository files supplied | NOT UPLOADED by this package | Create your own repo |

## Blockers requiring user accounts and real evidence

1. Obtain and train on real labeled MIMII recordings and capture evaluation output.
2. Run the app locally, confirm prediction and retraining on real new labeled WAVs.
3. Deploy to your cloud account (e.g., AWS EC2). Configure access controls and HTTPS.
4. Conduct Locust 1/2/3-container benchmark and save actual result files.
5. Record a camera-on YouTube demonstration.
6. Upload the repository to the user's GitHub account and provide Canvas attempt 1 ZIP then attempt 2 URL.

**Do not submit without completing those steps.**
