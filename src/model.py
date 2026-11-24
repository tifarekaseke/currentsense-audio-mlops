"""Custom audio feature classifier with saved weights and incremental updates.
Trains an MLP with Adam, L2 regularization and validation-based early stopping.
"""
from __future__ import annotations
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, classification_report)
from .preprocessing import FEATURE_NAMES, FEATURE_VERSION, list_labeled_audio, features_for

MODEL_PATH=Path(os.getenv('MODEL_PATH','models/current_sense_audio.pkl'))
OUTPUT_DIR=Path(os.getenv('OUTPUT_DIR','outputs'))
CLASSES=np.array([0,1], dtype=int)
LABELS=['normal','abnormal']

def metrics_for(y,prob):
    pred=(prob>=0.5).astype(int)
    values={
        'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'loss':float(log_loss(y,np.column_stack([1-prob,prob]),labels=CLASSES)),
        'confusion_matrix':confusion_matrix(y,pred,labels=CLASSES).tolist(),
        'classification_report':classification_report(y,pred,labels=CLASSES,target_names=LABELS,zero_division=0,output_dict=True)
    }
    if len(set(y))>1:
        values['roc_auc']=float(roc_auc_score(y,prob))
    else:
        values['roc_auc']=None
    return values

def _best_fit(Xtrain,ytrain,Xval,yval,model=None,epochs=90,patience=12):
    """Warm-start via partial_fit. Monitor validation loss; keep best weights."""
    model=model if model is not None else MLPClassifier(hidden_layer_sizes=(64,32),alpha=0.01,
                            learning_rate_init=0.001,solver='adam',random_state=42,
                            max_iter=1,early_stopping=False)
    rng=np.random.default_rng(42)
    best=None
    best_loss=float('inf')
    stale=0
    history=[]
    for epoch in range(epochs):
        order=rng.permutation(len(Xtrain))
        model.partial_fit(Xtrain[order],ytrain[order],classes=CLASSES)
        p=model.predict_proba(Xval)[:,1]
        val_loss=log_loss(yval,np.column_stack([1-p,p]),labels=CLASSES)
        val_f1=f1_score(yval,(p>=0.5).astype(int),zero_division=0)
        history.append({'epoch':epoch+1,'validation_loss':round(float(val_loss),6),
                        'validation_f1':round(float(val_f1),6)})
        if val_loss < best_loss - 0.0001:
            best_loss=float(val_loss)
            best=copy.deepcopy(model)
            stale=0
        else:
            stale+=1
        if stale>=patience:
            break
    return best if best is not None else model,history

def _save_atomic(bundle):
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    temp=MODEL_PATH.with_name(MODEL_PATH.name+'.tmp')
    joblib.dump(bundle,temp,compress=3)
    os.replace(temp,MODEL_PATH)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}; run python -m src.model first')
    b=joblib.load(MODEL_PATH)
    if b['feature_version']!=FEATURE_VERSION:
        raise ValueError('Incompatible model feature version')
    return b

def predict_audio(source, bundle=None):
    from .preprocessing import compute_features
    b=bundle if bundle is not None else load_model()
    feats=compute_features(source)
    X=b['scaler'].transform(feats.reshape(1,-1))
    prob=float(b['classifier'].predict_proba(X)[0,1])
    return {'label':'abnormal' if prob>=0.5 else 'normal',
            'probability_abnormal':prob,'confidence':max(prob,1-prob),
            'model_version':b['version'],
            'audio_features':{name:float(feats[i]) for i,name in enumerate(FEATURE_NAMES)}}

def train_initial(data_root='data', synthetic_fixture=False):
    root=Path(data_root)
    recs=list_labeled_audio(root)
    split={s:[r for r in recs if r['split']==s] for s in ('train','val','test')}
    for s in split:
        if len(split[s])<4 or len({r['label'] for r in split[s]})<2:
            raise ValueError(f'Need multiple labeled normal and abnormal .wav recordings in {root/s}')
    Xtr,ytr,train_recs=features_for(split['train'])
    Xv,yv,_=features_for(split['val'])
    Xt,yt,_=features_for(split['test'])
    scaler=StandardScaler().fit(Xtr)
    clf,hist=_best_fit(scaler.transform(Xtr),ytr,scaler.transform(Xv),yv)
    val=metrics_for(yv,clf.predict_proba(scaler.transform(Xv))[:,1])
    test=metrics_for(yt,clf.predict_proba(scaler.transform(Xt))[:,1])
    timestamp=datetime.now(timezone.utc).isoformat()
    bundle={'classifier':clf,'scaler':scaler,'version':'audio-v1','created_at':timestamp,
            'feature_version':FEATURE_VERSION,'synthetic_fixture':bool(synthetic_fixture),
            'validation_metrics':val,'test_metrics':test,'history':hist,
            'train_file_count':len(train_recs),'retraining_runs':0,'data_root':str(root)}
    _save_atomic(bundle)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    info={'created_at':timestamp,'model_version':'audio-v1','synthetic_fixture':bool(synthetic_fixture),
          'train_records':len(Xtr),'validation_records':len(Xv),'test_records':len(Xt),
          'validation':val,'test':test,'features':FEATURE_NAMES,'history':hist}
    (OUTPUT_DIR/'evaluation.json').write_text(json.dumps(info,indent=2))
    generate_insights(recs)
    generate_plots(info)
    print(f"Saved {MODEL_PATH}, test accuracy={test['accuracy']:.3f}, F1={test['f1']:.3f}; synthetic_fixture={synthetic_fixture}")
    return info

def generate_insights(records):
    import pandas as pd
    train_records=[r for r in records if r['split']=='train']
    X,y,_=features_for(train_records)
    df=pd.DataFrame(X,columns=FEATURE_NAMES)
    df['class']=['abnormal' if v==1 else 'normal' for v in y]
    k=['rms_mean','spectral_centroid_mean','zero_crossing_rate_mean']
    out={'sample_size':len(df),'counts':df['class'].value_counts().to_dict(),
         'feature_stats':{n:df.groupby('class')[n].agg(['mean','std']).round(4).to_dict(orient='index') for n in k},
         'interpretations':{
             'rms_mean':'Average audio energy; compare loudness of normal and abnormal recordings without assuming loudness diagnoses a fault.',
             'spectral_centroid_mean':'Frequency-weighted spectral center; higher values generally mean relatively more high-frequency energy.',
             'zero_crossing_rate_mean':'Rate of waveform sign changes; may reflect differences in high-frequency or noisy components.'}}
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/'feature_insights.json').write_text(json.dumps(out,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,name in zip(axes,k):
            for label in ('normal','abnormal'):
                ax.hist(df.loc[df['class']==label,name],bins=15,alpha=0.6,label=label)
            ax.set_title(name.replace('_',' ').title())
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR/'three_audio_features.png',dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    return out

def generate_plots(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    cm=np.array(report['test']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False)
    ax.set_title('Held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'confusion_matrix.png',dpi=150)
    plt.close(fig)
    h=report['history']
    fig,ax=plt.subplots(figsize=(6,3.5))
    ax.plot([x['epoch'] for x in h],[x['validation_loss'] for x in h])
    ax.set(xlabel='Epoch',ylabel='Validation loss',title='Validation-based early stopping')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'learning_curve.png',dpi=150)
    plt.close(fig)

def retrain_pending():
    """Continue existing saved model weights. New uploads are combined with original train and prior uploads.
    Fixed ORIGINAL StandardScaler is retained to avoid changing coordinates under warm-start weights.
    Immutable validation/test files remain excluded from fit.
    """
    from .database import pending_uploads, mark_trained, connection
    old=load_model()
    pending=pending_uploads()
    if not pending:
        raise ValueError('No pending labeled uploads to retrain on')
    if old.get('synthetic_fixture'):
        raise ValueError('Synthetic demonstration model cannot be incrementally retrained for real evaluation. First train_initial on real MIMII recordings.')
    data_root=old.get('data_root','data')
    base=list_labeled_audio(data_root,splits=('train',))
    with connection() as db:
        uploaded=[dict(row) for row in db.execute("SELECT * FROM uploads WHERE status IN ('trained','pending')")]
    recs=base+[{'path':r['path'],'label':r['label'],'split':'train'} for r in uploaded]
    validation=list_labeled_audio(data_root,splits=('val',))
    Xtr,ytr,_=features_for(recs)
    Xv,yv,_=features_for(validation)
    Xtr=old['scaler'].transform(Xtr)
    Xv=old['scaler'].transform(Xv)
    clf,hist=_best_fit(Xtr,ytr,Xv,yv,model=copy.deepcopy(old['classifier']),epochs=40,patience=9)
    newval=metrics_for(yv,clf.predict_proba(Xv)[:,1])
    old_f1=float(old['validation_metrics']['f1'])
    if newval['f1'] < old_f1 - 0.05:
        raise ValueError(f'Retraining rejected: validation F1 dropped from {old_f1:.3f} to {newval["f1"]:.3f}')
    version=f"audio-v{old['retraining_runs']+2}"
    updated={**old,'classifier':clf,'version':version,'validation_metrics':newval,
             'retraining_runs':old['retraining_runs']+1,'train_file_count':len(recs),
             'last_retrained_at':datetime.now(timezone.utc).isoformat(),'history':hist}
    _save_atomic(updated)
    mark_trained([r['id'] for r in pending])
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/'last_retraining.json').write_text(json.dumps({'version':version,
        'new_records':len(pending),'validation':newval,'history':hist},indent=2))
    return {'version':version,'new_records':len(pending),'validation_metrics':newval}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--synthetic-fixture',action='store_true')
    args=parser.parse_args()
    train_initial(args.data_root,synthetic_fixture=args.synthetic_fixture)
