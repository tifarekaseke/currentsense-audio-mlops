#!/usr/bin/env bash
set -euo pipefail
# Run from repository root AFTER genuine audio training, with Docker Compose and Locust installed.
# Compares 1, 2, 3 inference containers at the same workloads. Writes measured CSVs.
mkdir -p outputs
export RETRAIN_ENABLED=false
for replicas in 1 2 3; do
  docker compose up -d --build --scale api="$replicas"
  sleep 12
  for users in 10 25 50; do
    echo "Measuring $replicas replicas at $users concurrent users"
    locust -f locustfile.py --headless --host http://localhost:8000 \
      -u "$users" -r 5 -t 60s --csv "outputs/locust_${replicas}containers_${users}users" \
      --only-summary
  done
done
python scripts/benchmark_summary.py
