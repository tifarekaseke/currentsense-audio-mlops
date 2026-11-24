import io
import numpy as np
import soundfile as sf
from src.preprocessing import compute_features, FEATURE_NAMES, decode_audio

def test_wav_features_are_finite():
    sr=16000
    t=np.arange(sr)/sr
    y=(0.3*np.sin(2*np.pi*220*t)).astype('float32')
    buf=io.BytesIO()
    sf.write(buf,y,sr,format='WAV')
    raw=buf.getvalue()
    assert len(decode_audio(raw))==sr
    features=compute_features(raw)
    assert features.shape==(52,)
    assert len(FEATURE_NAMES)==52
    assert np.isfinite(features).all()
