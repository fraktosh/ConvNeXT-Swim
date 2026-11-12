# Phase 2 — CNN-Only Object Detection Baseline

## Objective
Build a reproducible CNN-only object detector using ConvNeXt-Tiny multi-scale representations.

This is the control condition for the later hybrid and adaptive systems.

## Dataset
Use MS COCO for the main benchmark if it is available and practical in the environment.

Do not download enormous datasets without first checking disk space and asking only if necessary.

Use standard COCO train/validation splits.

Do NOT use license-plate/ANPR datasets.

## Architecture
Build:

Input
  -> ConvNeXt-Tiny
  -> P3/P4/P5
  -> multi-scale detection head
  -> class + box predictions

Choose a lightweight, clearly documented detection head suitable for the RTX 3050.

The head must remain identical, or as close as scientifically possible, in later hybrid/adaptive comparisons.

## Training
Create configurable:
- image size
- batch size
- optimizer
- learning rate
- scheduler
- epochs
- augmentation
- seed
- mixed precision

Do not hard-code these into the model.

## Required metrics
Record:
- AP@[0.5:0.95]
- AP50
- AP75
- AP small
- AP medium
- AP large
- parameter count
- FLOPs/GMACs
- inference latency
- FPS/throughput
- GPU memory

## Reproducibility
Save:
- YAML configuration
- checkpoint
- training log
- validation metrics
- final summary JSON

## Deliverables
Create:
- `src/models/cnn_detector.py`
- `src/models/detection_head.py`
- `src/data/coco.py`
- `scripts/train_cnn_baseline.py`
- `scripts/evaluate_cnn_baseline.py`
- `configs/cnn_baseline.yaml`
- `results/cnn_baseline.json`
- `results/cnn_baseline_report.md`

## Important
Do not tune the baseline specifically to make the proposed method look better.

The baseline exists as a genuine scientific control.

If full COCO training is too expensive, implement the full pipeline and run a small smoke-test subset first. Clearly label smoke-test results as debugging results and do not treat them as final scientific results.
