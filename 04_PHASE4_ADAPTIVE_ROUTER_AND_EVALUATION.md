# Phase 4 — Region-Adaptive CNN–Transformer Framework

## Objective
Implement the research contribution approved by Prof. Srikanta Pal:

A region-adaptive CNN–Transformer framework that dynamically determines the appropriate level of feature processing from the multi-scale features extracted by ConvNeXt-Tiny.

## Starting point
Only begin this phase after Phase 1–3 are working.

## Architecture

Input
  |
ConvNeXt-Tiny
  |
P3 / P4 / P5
  |
Region/feature allocation module
  |
+-------------------------------+
| Processing choices            |
| 1. CNN only                   |
| 2. CNN + lightweight Swin     |
| 3. CNN + deeper Swin          |
+-------------------------------+
  |
Adaptive feature fusion
  |
Same detection head as baselines

## Router inputs
Investigate learned routing using information available from the feature maps.

Potential signals:
- object-scale distribution
- local/regional density
- feature uncertainty
- feature-level identity
- activation statistics
- local feature complexity

Do not assume every signal is necessary. Make them individually ablatable.

## Routing
Start with a soft differentiable routing mechanism for training.

A Gumbel-Softmax / straight-through estimator may be investigated for learning discrete choices.

At inference:
- produce a hard routing decision;
- skip unselected branches where technically possible;
- verify that skipped branches actually reduce measured computation/latency.

Do NOT merely compute all three branches and multiply their outputs by zeros. That would not demonstrate conditional computation.

## Computational budget
Introduce a configurable compute-cost term.

The optimization should encourage:
- additional Transformer processing when it provides sufficient expected detection benefit;
- CNN-only processing when extra global computation is unnecessary.

The exact mathematical formulation should be documented before final experiments.

## Critical baselines
Compare against:
1. CNN-only
2. fixed CNN + Swin
3. adaptive model without compute-budget loss
4. adaptive model with compute-budget loss

Where practical, also compare against relevant prior dynamic-computation methods identified during the literature review.

## Ablation studies
At minimum investigate:
- P3 only routing
- P4 only routing
- P3 + P4 routing
- P3 + P4 + P5 routing if feasible
- scale signal removed
- density signal removed
- uncertainty signal removed
- feature-level signal removed
- no budget loss
- different budget strengths

## Evaluation
Report:
- AP@[0.5:0.95]
- AP50
- AP75
- AP small/medium/large
- parameters
- FLOPs/GMACs
- actual latency
- FPS
- peak GPU memory
- effective executed computation if measurable
- routing distribution

## Routing analysis
Produce plots/tables showing:
- percentage of regions assigned to CNN-only
- percentage assigned to lightweight Transformer
- percentage assigned to deeper Transformer
- allocation by P3/P4/P5
- allocation by object scale
- allocation by scene/region complexity
- accuracy versus compute budget

This analysis is scientifically important because the research is about allocation, not merely final mAP.

## Fair comparison
Whenever possible, compare models under approximately matched:
- input resolution
- training schedule
- dataset
- detection head
- parameter budget
- computational budget

If exact matching is impossible, report the mismatch explicitly.

## Deliverables
Create:
- `src/routing/adaptive_allocator.py`
- `src/losses/compute_budget_loss.py`
- `src/models/adaptive_hybrid_detector.py`
- `scripts/train_adaptive.py`
- `scripts/evaluate_adaptive.py`
- `scripts/run_ablation.py`
- configuration YAML files for every experiment
- `results/adaptive_results.json`
- `results/ablation_results.json`
- `results/routing_analysis.json`
- plots in `results/figures/`
- `results/final_experiment_report.md`

## Final research report
The final report must contain:
1. Research question
2. Experimental setup
3. Baselines
4. Proposed architecture
5. Training details
6. Main results
7. Accuracy-efficiency comparison
8. Ablation results
9. Routing analysis
10. Failure cases
11. Limitations
12. Reproducibility information

## Absolutely prohibited
Do not fabricate:
- mAP
- FPS
- FLOPs
- latency
- ablation results
- comparisons
- literature claims

If something cannot be measured, state that it could not be measured.

## Final goal
Determine empirically whether adaptive allocation of local CNN and global Transformer processing across regions and hierarchical feature levels provides a better accuracy–efficiency trade-off than fixed CNN and fixed CNN–Transformer architectures.

Do not declare the hypothesis successful unless the experiments support it.
