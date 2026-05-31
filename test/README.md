# Test payloads for the /predict endpoint

Real engineered feature vectors (the 42 model inputs) plus negative-test cases.
Regenerate any time with:

```bash
uv run --no-sync python test/generate_payloads.py
```

## Files

| File | Expected result |
|------|-----------------|
| `predict_not_readmitted.json` | `200` → `prediction: 0` ("not readmitted <30 days") |
| `predict_readmitted.json` | `200` → `prediction: 1` ("readmitted <30 days") |
| `invalid_missing_features.json` | `422` (missing required features) |
| `invalid_empty.json` | `422` (features must not be empty) |

## Run the service, then test

```bash
# 1. start the API (needs models/winner_random_forest.joblib from main.py)
uv run --no-sync uvicorn src.api.app:app --port 8000

# 2. valid prediction
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @test/predict_readmitted.json

# 3. negative test (expect HTTP 422)
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @test/invalid_missing_features.json
```
