# Phase 1 — ConvNeXt-Tiny Multi-Scale Feature Extraction

## Objective
Establish a verified ConvNeXt-Tiny hierarchical feature extractor. Do not implement adaptive routing yet.

## Tasks

### 1. Environment
Verify:
- Python
- PyTorch
- CUDA availability
- GPU name
- timm version

Save the environment information to:
`results/environment.txt`

### 2. Load ConvNeXt-Tiny
Use the timm ConvNeXt-Tiny implementation with pretrained ImageNet weights if available.

Use a feature-extraction configuration rather than a classification head.

### 3. Feature hierarchy
For a dummy input of shape:
`[1, 3, 640, 640]`

extract all hierarchical feature maps.

Print and save:
- tensor shape
- channel count
- spatial resolution
- downsampling ratio

Identify the candidate P3/P4/P5 levels corresponding approximately to:
- 1/8
- 1/16
- 1/32

Do not assume these indices until verified.

### 4. Profiling
Measure:
- parameter count
- GPU memory after model creation
- inference time over multiple warm-up and measured iterations
- throughput

Use CUDA synchronization when timing on GPU.

### 5. Visualization
Create a simple feature-activation visualization for at least one sample:
- input image
- selected P3/P4/P5 activation summaries

Do not alter the model to produce this visualization.

### 6. Deliverables
Create:
- `src/backbone/convnext_features.py`
- `scripts/run_phase1.py`
- `configs/phase1.yaml`
- `results/phase1_feature_shapes.json`
- `results/phase1_report.md`
- `results/phase1_feature_visualization.png`

### Acceptance criteria
Phase 1 is complete only when:
- ConvNeXt-Tiny loads successfully;
- a forward pass succeeds on GPU;
- P3/P4/P5 shapes are verified;
- profiling runs successfully;
- outputs are saved;
- no model architecture has been changed.

Do not proceed to training until this is complete.
