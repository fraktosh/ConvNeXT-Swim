python -u scripts/train_fixed_hybrid.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u scripts/evaluate_fixed_hybrid.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u scripts/profile_inference_fixed_hybrid.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
