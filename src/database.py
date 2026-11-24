"""SQLite records for uploads, prediction logs and training jobs."""
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
DB_PATH = Path(os.getenv("DB_PATH", "data/current_sense.sqlite3"))

def utcnow():
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        yield db
        db.commit()
    finally:
        db.close()

def init_db():
    with connection() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, path TEXT NOT NULL, label TEXT NOT NULL, sha256 TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, trained_at TEXT)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_upload_status ON uploads(status)")
        db.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, prediction TEXT NOT NULL, probability REAL NOT NULL, latency_ms REAL NOT NULL, model_version TEXT NOT NULL, created_at TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS feedback (prediction_id TEXT PRIMARY KEY, actual_label TEXT NOT NULL, created_at TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS training_jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT, detail TEXT)")

def add_upload(record_id, path, label, digest):
    with connection() as db:
        db.execute("INSERT INTO uploads (id,path,label,sha256,created_at) VALUES (?,?,?,?,?)", (record_id, path, label, digest, utcnow()))

def pending_uploads():
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY created_at")]

def mark_trained(ids):
    if not ids:
        return
    with connection() as db:
        db.executemany("UPDATE uploads SET status='trained',trained_at=? WHERE id=?", [(utcnow(), x) for x in ids])

def add_prediction(pred_id, label, probability, latency, version):
    with connection() as db:
        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)", (pred_id, label, float(probability), float(latency), version, utcnow()))

def create_job(job_id):
    with connection() as db:
        db.execute("INSERT INTO training_jobs (id,status,started_at) VALUES (?, 'queued', ?)", (job_id, utcnow()))

def job_status(job_id, status, detail=None):
    with connection() as db:
        db.execute("UPDATE training_jobs SET status=?, detail=?, completed_at=? WHERE id=?", (status, detail, utcnow() if status in ('complete','failed') else None, job_id))

def get_job(job_id):
    with connection() as db:
        r=db.execute("SELECT * FROM training_jobs WHERE id=?",(job_id,)).fetchone()
        return dict(r) if r else None

def add_feedback(prediction_id, actual_label):
    with connection() as db:
        found=db.execute("SELECT id FROM predictions WHERE id=?",(prediction_id,)).fetchone()
        if not found:
            raise ValueError("Unknown prediction ID")
        db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?)",(prediction_id,actual_label,utcnow()))

def summary():
    with connection() as db:
        pending=db.execute("SELECT count(*) FROM uploads WHERE status='pending'").fetchone()[0]
        trained=db.execute("SELECT count(*) FROM uploads WHERE status='trained'").fetchone()[0]
        total=db.execute("SELECT count(*) FROM predictions").fetchone()[0]
        avg=db.execute("SELECT avg(latency_ms) FROM predictions").fetchone()[0]
        recent=[dict(x) for x in db.execute("SELECT prediction,probability,latency_ms,created_at FROM predictions ORDER BY created_at DESC LIMIT 100")]
        feedback=[dict(x) for x in db.execute("SELECT p.prediction,f.actual_label FROM predictions p JOIN feedback f ON p.id=f.prediction_id ORDER BY f.created_at DESC LIMIT 500")]
        prod=None
        if feedback:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            actual=[int(x["actual_label"]=="abnormal") for x in feedback]
            pred=[int(x["prediction"]=="abnormal") for x in feedback]
            prod={"labeled_count":len(feedback), "accuracy":round(float(accuracy_score(actual,pred)),4), "precision":round(float(precision_score(actual,pred,zero_division=0)),4), "recall":round(float(recall_score(actual,pred,zero_division=0)),4), "f1":round(float(f1_score(actual,pred,zero_division=0)),4)}
        return {"production_feedback_metrics":prod,"pending_uploads":pending,"trained_uploads":trained,"prediction_count":total,"avg_latency_ms":round(avg,2) if avg else None,"recent_predictions":recent}
