from fastapi.testclient import TestClient
from api import app

def test_health_without_training_is_safe():
    with TestClient(app) as client:
        r=client.get('/health')
        assert r.status_code==200
        assert r.json()['status']=='online'
        assert 'uptime_seconds' in r.json()

def test_upload_requires_token():
    with TestClient(app) as client:
        r=client.post('/upload',data={'label':'normal'},files={'files':('sample.wav',b'bad','audio/wav')})
        assert r.status_code==403
