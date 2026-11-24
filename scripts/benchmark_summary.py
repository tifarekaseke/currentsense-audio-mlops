"""Aggregate REAL Locust `_stats.csv` files; never fabricate benchmarking results."""
from pathlib import Path
import csv
rows=[]
for p in sorted(Path('outputs').glob('locust_*_stats.csv')):
    with p.open(newline='') as f:
        records=list(csv.DictReader(f))
    x=next((r for r in records if r.get('Name')=='Aggregated'),None)
    if not x:
        x=next((r for r in records if r.get('Name')=='/predict'),None)
    if x:
        rows.append({'file':p.name,'request_count':x.get('Request Count'),
            'failure_count':x.get('Failure Count'),'rps':x.get('Requests/s'),
            'median_ms':x.get('Median Response Time'),
            'average_ms':x.get('Average Response Time'),
            'p95_ms':x.get('95%'), 'p99_ms':x.get('99%')})
if not rows:
    raise SystemExit('No measured Locust CSVs yet: run scripts/run_load_tests.sh on a Docker host.')
path=Path('outputs/locust_comparison.csv')
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(path.read_text())
