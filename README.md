# COVID MLOps Lab (End-to-End)

This repository is a hands-on MLOps exercise built around a real-world COVID patient dataset.

## Project Goal
Predict **mortality risk** (`DIED: 1/0`) using admission-time features (demographics + comorbidities),
while ensuring:
- reproducible pipelines
- experiment tracking (MLflow)
- tests + CI
- deployment via FastAPI + Docker

## Dataset
Place the dataset here:

`data/raw/Covid Data.csv`

> Source file used in this lab: `/mnt/data/Covid Data.csv` (local to this environment)

## Phases (you will build step-by-step)
1. Project framing + repo scaffold ✅
2. Data loading + label creation + missing-code handling
3. Baseline model training + evaluation
4. Stronger model + threshold tuning
5. MLflow tracking + artifacts
6. FastAPI inference service
7. Dockerization
8. CI (GitHub Actions) + tests
9. Monitoring stubs + extensions (optional: DVC, registry)

## Quickstart (will be fully enabled after Phase 2)
```bash
make setup
make train
make evaluate
make serve

