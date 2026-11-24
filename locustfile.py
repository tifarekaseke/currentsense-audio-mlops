"""Locust HTTP flood test. Start only after training a real model.
Sample WAV: LOCUST_SAMPLE_WAV=data/test/normal/my_recording.wav
locust -f locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000 --csv outputs/locust_users10
"""
import os
from pathlib import Path
from locust import HttpUser, task, between

sample=os.getenv('LOCUST_SAMPLE_WAV','')
if sample:
    path=Path(sample)
else:
    clips=list(Path('data/test').rglob('*.wav'))
    path=clips[0] if clips else None
FILE_BYTES=path.read_bytes() if path and path.exists() else None

class PredictionUser(HttpUser):
    wait_time=between(0.2,1)
    @task
    def predict_audio(self):
        if FILE_BYTES is None:
            raise RuntimeError('No labeled test WAV available; set LOCUST_SAMPLE_WAV')
        with self.client.post('/predict',
                files={'file':('known_test.wav',FILE_BYTES,'audio/wav')},
                name='/predict',catch_response=True,timeout=30) as response:
            if response.status_code!=200:
                response.failure(f'HTTP {response.status_code}: {response.text[:120]}')
            else:
                answer=response.json()
                if answer.get('label') not in ('normal','abnormal'):
                    response.failure('Missing/invalid classification label')
