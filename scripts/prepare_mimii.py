"""Organize legitimate labeled MIMII WAV files into immutable train/val/test splits.
Download and extract a MIMII archive yourself: https://zenodo.org/records/3384388
Example: python scripts/prepare_mimii.py --source /path/to/extracted/6_dB_fan --machine-id id_00 --max-per-class 120
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',required=True,help='Extracted folder containing normal/ and abnormal/ WAV recordings')
p.add_argument('--machine-id',default='id_00')
p.add_argument('--max-per-class',type=int,default=120)
p.add_argument('--dest',default='data')
p.add_argument('--overwrite',action='store_true')
args=p.parse_args()
root=Path(args.source)
dest=Path(args.dest)
if not root.exists(): p.error(f'Source not found: {root}')
if any((dest/x).rglob('*.wav') for x in ('train','val','test')) and not args.overwrite:
    p.error('Target already has WAV files. To avoid mixing experiments, pass --overwrite after backing up data.')
rng=random.Random(42)
manifest=[]
for label in ('normal','abnormal'):
    candidates=[f for f in root.rglob('*.wav') if f.parent.name==label and args.machine_id in f.parts]
    if len(candidates)<15:
        p.error(f'Only {len(candidates)} {label} recordings for {args.machine_id}; need at least 15')
    candidates=sorted(candidates)
    rng.shuffle(candidates)
    candidates=candidates[:args.max_per_class] if args.max_per_class>0 else candidates
    n=len(candidates)
    n_test=max(2,round(n*.15))
    n_val=max(2,round(n*.15))
    for i,source in enumerate(candidates):
        split='test' if i<n_test else ('val' if i<n_test+n_val else 'train')
        target=dest/split/label/f'{args.machine_id}_{source.name}'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        manifest.append({'split':split,'label':label,'source':str(source),
                         'relative_path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(dest/'dataset_manifest.json').write_text(json.dumps({'dataset':'MIMII public 1.0',
    'source_doi':'10.5281/zenodo.3384388','machine_id':args.machine_id,'seed':42,
    'split_method':'random recording-level split stratified within class, not machine/session independent',
    'records':manifest},indent=2))
print('Copied',len(manifest),'REAL recordings into',dest)
for split in ('train','val','test'):
 print(split,len([r for r in manifest if r['split']==split]))
print('Important: random recording split does not prove generalization to unseen machines.')
