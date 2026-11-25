# CurrentSense Audio MLOps — Final Submission Status

**Author:** Tifare Kaseke  
**Repository:** https://github.com/tifarekaseke/currentsense-audio-mlops  
**Video:** https://youtu.be/sDzIfruq-Fk  
**Streamlit dashboard:** https://currentsense-audio-mlops.streamlit.app/  
**FastAPI documentation:** https://currentsense-audio-mlops.onrender.com/docs  
**API health check:** https://currentsense-audio-mlops.onrender.com/health

## Completion checklist

| Assignment component | Status | Supporting evidence |
|---|---|---|
| Genuine non-tabular audio dataset | Completed | 600 MIMII `-6_dB_fan`, `id_00` WAV recordings; 420 train, 90 validation, 90 test |
| Audio preprocessing and visualizations | Completed | Executed `notebook/currentsense_audio_mlops.ipynb`; 52 features, waveforms and feature distributions |
| Offline model training and testing | Completed | `src/model.py`, `models/current_sense_audio.pkl`, `outputs/evaluation.json` |
| Final retrained-model evaluation | Completed | `outputs/evaluation_audio_v2_test.json` and `outputs/confusion_matrix_audio_v2_test.png` |
| Single WAV prediction | Demonstrated | Streamlit dashboard and FastAPI `POST /predict`; cloud prediction with `audio-v2` |
| Bulk labeled WAV upload and database | Demonstrated locally | Five normal and five abnormal recordings; SQLite-backed upload workflow |
| Existing-model retraining | Completed | `outputs/last_retraining.json`: 10 recordings incorporated, model updated to `audio-v2` |
| API health and basic monitoring | Demonstrated | `GET /health`, `GET /stats`; cloud `/health` confirmed model loaded and `audio-v2` active |
| Docker Compose and Nginx | Completed locally | `Dockerfile`, `docker-compose.yml`, `nginx/default.conf` |
| Locust flood-request simulations | Completed locally | Real WAV request CSVs in `outputs/` for 1, 2 and 3 API replicas |
| Public cloud inference deployment | Demonstrated | Render FastAPI backend and Streamlit Community Cloud frontend |
| Automated testing | Test suite included | `tests/test_preprocessing.py`, `tests/test_api.py`; run locally to verify current pass status |
| Camera-on demonstration video | Link supplied | YouTube URL above; verify that it is accessible and shows prediction and retraining |
| GitHub repository | Uploaded | Repository URL above; confirm latest documentation commits are pushed |

## Key results

- **Dataset:** 600 real MIMII recordings, balanced across normal and abnormal classes; 10 separate additional labeled recordings used for retraining.
- **Original `audio-v1` held-out test:** accuracy **68.89%**, abnormal-class F1 **67.44%**.
- **Retraining job:** completed; model advanced to `audio-v2` using **10** new recordings. Validation accuracy **71.11%**, validation F1 **69.77%**.
- **Final `audio-v2` held-out test:** accuracy **68.89%**, abnormal-class precision **69.77%**, recall **66.67%**, F1 **68.18%**, ROC AUC **0.7353**. Confusion matrix: `[[32, 13], [15, 30]]` (rows and columns ordered normal, abnormal).
- **Locust, successful 10-user, 60-second runs:**

| API replicas | Requests | Failures | Mean latency | Throughput |
|---|---:|---:|---:|---:|
| 1 | 65 | 0 | 7,595 ms | 1.12 requests/s |
| 2 | 263 | 0 | 1,516 ms | 4.45 requests/s |
| 3 | 263 | 0 | 1,469 ms | 4.64 requests/s |

An earlier two-replica test recorded **10 failures out of 19 requests (52.63%)**. A later warmed-up run succeeded. The cause was not conclusively identified; both results are retained as evidence.

## Scope and limitations

The public cloud version is intended for **inference**. Administrative labeled-file uploads and retraining were demonstrated locally; they are not offered as public unauthenticated actions. Local SQLite storage and shared model files are prototype components, not durable multi-instance production infrastructure. The dataset split is recording-based and does not establish generalization to unseen machine IDs. The classifier is a research prototype, not a certified mechanical diagnostic system.

## Final submission actions

1. Confirm the links at the top are accessible to the lecturer, including the YouTube video's visibility.
2. Ensure the executed notebook, trained `audio-v2` model, test evaluation files, Locust CSVs and this documentation are present on GitHub.
3. Verify `python -m pytest -q` passes in the `currentsense-mlops` environment, if not already done since the final code changes.
4. Submit the **GitHub repository URL** and any ZIP requested by the assignment portal; follow the portal's instructions for each submission attempt.

**Evidence integrity:** Validation metrics and held-out test metrics are reported separately. A successful API response or a high-confidence individual prediction is not the same as overall model accuracy.
