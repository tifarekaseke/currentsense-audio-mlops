"""FastAPI inference, administration, health and production feedback endpoints."""
from __future__ import annotations
import asyncio
import hashlib
import os
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from pydantic import BaseModel
from src.database import (init_db, add_upload, pending_uploads, add_prediction, create_job,
                          job_status, get_job, summary, add_feedback)
from src.model import load_model, predict_audio, retrain_pending
from src.preprocessing import decode_audio

app=FastAPI(title='CurrentSense MLOps API',version='1.0.0')
STARTED=time.monotonic()
MAX_UPLOAD=24*1024*1024
UPLOAD_DIR=Path(os.getenv('UPLOAD_DIR','data/uploads'))
_pool=ThreadPoolExecutor(max_workers=1)
_inference_pool=ThreadPoolExecutor(max_workers=2)
_job_lock=threading.Lock()
_model_cache={'mtime':None,'model':None}
_cache_lock=threading.RLock()

@app.on_event('startup')
def startup():
    init_db()
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)

def get_bundle():
    from src.model import MODEL_PATH
    if not MODEL_PATH.exists():
        raise HTTPException(503,'Model unavailable. Train on labeled audio first.')
    with _cache_lock:
        mtime=MODEL_PATH.stat().st_mtime_ns
        if _model_cache['mtime']!=mtime:
            _model_cache['model']=load_model()
            _model_cache['mtime']=mtime
        return _model_cache['model']

def require_admin(token):
    expected=os.getenv('ADMIN_TOKEN','')
    if not expected or token!=expected:
        raise HTTPException(403,'Valid X-Admin-Token required; set ADMIN_TOKEN')

def verify_wav(name,data):
    if not name.lower().endswith('.wav'):
        raise HTTPException(400,'Only .wav files are allowed')
    if len(data)>MAX_UPLOAD or not data:
        raise HTTPException(413,'WAV file is empty or exceeds 24 MB')
    if not (data[:4]==b'RIFF' and data[8:12]==b'WAVE'):
        raise HTTPException(400,'Invalid WAV RIFF header')
    try:
        decode_audio(data)
    except Exception as exc:
        raise HTTPException(400,f'Audio validation error: {exc}') from exc

def _retrain_worker(job_id):
    try:
        job_status(job_id,'running')
        result=retrain_pending()
        job_status(job_id,'complete',detail=str(result))
    except Exception as exc:
        job_status(job_id,'failed',detail=str(exc))
    finally:
        _job_lock.release()

@app.get('/health')
def health():
    from src.model import MODEL_PATH
    exists=MODEL_PATH.exists()
    return {'status':'online','uptime_seconds':round(time.monotonic()-STARTED,1),
            'model_loaded':exists,'model_version':get_bundle()['version'] if exists else None}

def _predict_sync(filename, raw):
    verify_wav(filename, raw)
    return predict_audio(raw, bundle=get_bundle())

@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    start=time.perf_counter()
    raw=await file.read(MAX_UPLOAD+1)
    result=await asyncio.get_running_loop().run_in_executor(
        _inference_pool,_predict_sync,file.filename or '',raw)
    elapsed=(time.perf_counter()-start)*1000
    pred_id=uuid.uuid4().hex
    add_prediction(pred_id,result['label'],result['probability_abnormal'],elapsed,result['model_version'])
    return {'prediction_id':pred_id,'label':result['label'],
            'confidence':round(result['confidence'],5),
            'probability_abnormal':round(result['probability_abnormal'],5),
            'model_version':result['model_version'],
            'latency_ms':round(elapsed,2),
            'feature_highlights':{k:round(result['audio_features'][k],5) for k in
             ('rms_mean','spectral_centroid_mean','zero_crossing_rate_mean')}}

@app.post('/upload')
async def upload(files:list[UploadFile]=File(...),label:str=Form(...),
                 x_admin_token:str|None=Header(default=None)):
    require_admin(x_admin_token)
    if label not in ('normal','abnormal'):
        raise HTTPException(400,'label must be normal or abnormal')
    if len(files)>40:
        raise HTTPException(413,'Maximum 40 recordings per request')
    saved=[]
    for f in files:
        raw=await f.read(MAX_UPLOAD+1)
        verify_wav(f.filename or '',raw)
        identifier=uuid.uuid4().hex
        path=UPLOAD_DIR/f'{identifier}.wav'
        path.write_bytes(raw)
        add_upload(identifier,str(path.resolve()),label,hashlib.sha256(raw).hexdigest())
        saved.append({'id':identifier,'original_name':f.filename,'label':label})
    return {'stored_count':len(saved),'saved':saved,'retraining_eligible':True}

@app.post('/retrain',status_code=202)
def retrain(x_admin_token:str|None=Header(default=None)):
    require_admin(x_admin_token)
    if os.getenv('RETRAIN_ENABLED','true').lower()!='true':
        raise HTTPException(403,'Retraining disabled on this inference replica')
    if not pending_uploads():
        raise HTTPException(400,'Upload new labeled files first')
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409,'A retraining job is already running on this instance')
    job_id=uuid.uuid4().hex
    create_job(job_id)
    _pool.submit(_retrain_worker,job_id)
    return {'job_id':job_id,'status':'queued','message':'Existing trained model will be updated'}

@app.get('/jobs/{job_id}')
def job(job_id:str):
    result=get_job(job_id)
    if not result: raise HTTPException(404,'Job does not exist')
    return result

@app.get('/stats')
def stats():
    s=summary()
    from src.model import OUTPUT_DIR
    import json
    p=OUTPUT_DIR/'feature_insights.json'
    s['feature_insights']=json.loads(p.read_text()) if p.exists() else None
    try:
        b=get_bundle()
        s['model']={'version':b['version'],'created_at':b['created_at'],
                   'synthetic_fixture':b.get('synthetic_fixture',False),
                   'validation_metrics':b['validation_metrics'],
                   'test_metrics':b['test_metrics'],
                   'training_records':b['train_file_count']}
    except HTTPException:
        s['model']=None
    return s

class Feedback(BaseModel):
    prediction_id:str
    actual_label:str

@app.post('/feedback')
def feedback(item:Feedback,x_admin_token:str|None=Header(default=None)):
    require_admin(x_admin_token)
    if item.actual_label not in ('normal','abnormal'):
        raise HTTPException(400,'actual_label must be normal or abnormal')
    try:
        add_feedback(item.prediction_id,item.actual_label)
    except ValueError as exc:
        raise HTTPException(404,str(exc)) from exc
    return {'status':'recorded'}
