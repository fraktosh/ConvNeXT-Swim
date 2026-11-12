python -u scripts/train_adaptive.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u scripts/evaluate_adaptive.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u scripts/profile_inference_adaptive.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
