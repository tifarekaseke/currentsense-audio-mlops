"""Synthetic WAV fixture for testing code paths ONLY. NOT a scientific dataset.
Run: python scripts/make_synthetic_smoke_data.py
Train with: python -m src.model --data-root data/fixture --synthetic-fixture
NEVER report fixture scores as MIMII results.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
sr=16000
root=Path('data/fixture')
rng=np.random.default_rng(2026)
for split,count in [('train',18),('val',7),('test',7)]:
 for label in ('normal','abnormal'):
  out=root/split/label
  out.mkdir(parents=True,exist_ok=True)
  for i in range(count):
   duration=1.4
   t=np.arange(int(duration*sr),dtype=float)/sr
   f=85+rng.uniform(-9,9)
   signal=np.sin(2*np.pi*f*t)+0.25*np.sin(2*np.pi*2*f*t)
   signal += rng.normal(0,0.02,len(t))
   if label=='abnormal':
    signal += 0.38*np.sin(2*np.pi*(700+rng.uniform(-100,100))*t)+rng.normal(0,0.06,len(t))
   signal/=max(1.1,np.max(np.abs(signal))*1.1)
   sf.write(out/f'{label}_{i:03d}.wav',signal,sr,subtype='PCM_16')
(root/'SYNTHETIC_FIXTURE_ONLY.txt').write_text('NOT REAL MACHINE RECORDINGS. For testing application wiring only. Never cite the resulting metrics as MIMII results.\n')
print('Created synthetic test fixture at',root)
