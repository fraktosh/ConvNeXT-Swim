# Prof. Pal Research Project — Master Instructions for Antigravity

## Research title
Adaptive Allocation of Local and Global Representations for Efficient Object Detection

## Research direction
Implement and experimentally investigate a region-adaptive CNN–Transformer object-detection framework based on the research direction approved by Prof. Srikanta Pal:

> A region-adaptive CNN–Transformer framework that dynamically determines the appropriate level of feature processing from the multi-scale features extracted by ConvNeXt-Tiny.

## Core idea
Use ConvNeXt-Tiny as the hierarchical CNN backbone. At selected multi-scale feature levels, a learned allocation mechanism should decide how much additional Transformer-based contextual processing a region receives.

The intended processing choices are:
1. CNN only
2. CNN + lightweight Transformer processing
3. CNN + deeper Transformer processing

The goal is NOT to claim that CNN–Transformer hybrids or routing are themselves novel. The research question is whether region/scale/complexity-aware allocation of local versus global processing can improve the accuracy–efficiency trade-off under controlled computational budgets.

## User environment
- Windows / PowerShell
- Python 3.10.11
- PyTorch 2.5.1+cu121
- CUDA 12.1
- NVIDIA RTX 3050 laptop GPU
- `timm` may need to be installed
- Do not replace or downgrade the existing PyTorch/CUDA installation unless a genuine compatibility problem is demonstrated.

## Mandatory research constraints
- Do NOT include ANPR, license-plate detection, number-plate recognition, or license-plate datasets in this research project.
- Do NOT silently change the research question.
- Do NOT claim novelty such as "first", "novel", or "no previous work" without explicit evidence.
- Do NOT immediately build the complete adaptive system before establishing baselines.
- Every major experimental result must be reproducible from a saved configuration.
- Keep baseline and proposed-model code modular so components can be compared fairly.
- Do not overwrite datasets, source images, checkpoints, or experiment results.
- Use fixed random seeds where appropriate.
- Record package versions, GPU information, model configuration, dataset split, input resolution, batch size, training epochs, optimizer, learning rate, and evaluation metrics.

## Required experimental progression
Phase 1: Environment and ConvNeXt-Tiny multi-scale feature extraction.
Phase 2: CNN-only object-detection baseline.
Phase 3: Fixed CNN + Transformer baseline using Swin.
Phase 4: Region-adaptive allocation with three processing levels.
Phase 5: Computational-budget experiments, ablations, latency/FLOPs analysis, and final comparison.

Do not skip phases.

## Expected hierarchy
For a 640x640 input, investigate the hierarchical outputs around:
- P3: 1/8 resolution
- P4: 1/16 resolution
- P5: 1/32 resolution

Do not hard-code channel dimensions until they are verified from the actual installed ConvNeXt-Tiny implementation.

## Engineering requirements
Create a clean project structure such as:

project/
  configs/
  src/
    backbone/
    transformer/
    routing/
    models/
    losses/
    data/
    evaluation/
  experiments/
  checkpoints/
  results/
  logs/
  scripts/
  README.md

Use modular PyTorch code.

## First action
Before implementing the model:
1. Inspect the current Python environment.
2. Verify PyTorch/CUDA/GPU.
3. Install only missing dependencies.
4. Verify `timm`.
5. Run the ConvNeXt-Tiny feature extraction test.
6. Save the resulting feature shapes and environment information.

## Deliverables
At every phase, produce:
- runnable code
- configuration file
- README/instructions
- saved logs
- a concise experiment report
- no fabricated results

If an experiment fails, diagnose it and fix the implementation rather than reporting an invented result.

## Scientific discipline
The project is intended to become a research paper. Therefore:
- preserve baseline comparability;
- use matched input sizes and datasets;
- report parameter count, FLOPs/GMACs, latency, throughput, GPU memory where possible, and detection accuracy;
- separate development/debugging experiments from final reported experiments;
- maintain an experiment registry/table with model name, configuration, seed, dataset, metrics, and checkpoint path.

Start with Phase 1 and proceed sequentially.
