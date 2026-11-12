# Phase 3 — Fixed CNN + Swin Transformer Hybrid Baseline

## Objective
Establish a fixed hybrid baseline before introducing adaptive routing.

This experiment answers:
"Does adding Transformer contextual processing at a fixed feature level improve detection, and at what computational cost?"

## Backbone
Start from the verified ConvNeXt-Tiny P3/P4/P5 representations.

Do not replace ConvNeXt-Tiny.

## Transformer
Use Swin Transformer components.

The implementation may use:
- Hugging Face Transformers, or
- a compatible timm Swin implementation,

but document the choice and version.

Do not use a giant Swin model that makes the RTX 3050 experiment impractical.

## Fixed hybrid variants
Implement at least:

A. CNN-only
- ConvNeXt features
- no Transformer

B. CNN + Swin at P4
- P3 unchanged
- P4 receives Swin processing
- P5 unchanged

C. CNN + Swin at P3/P4
- P3 receives lightweight Swin processing
- P4 receives Swin processing
- P5 unchanged

If computationally practical, also test:
D. CNN + Swin at P4/P5

Keep the detection head consistent.

## Important tensor handling
Document and test every conversion:
CNN feature:
`B x C x H x W`

Transformer token representation:
`B x N x C`

and back:
`B x C x H x W`

Verify that spatial dimensions and channel dimensions are preserved correctly.

## Profiling
For every variant measure:
- parameters
- FLOPs/GMACs
- latency
- throughput
- peak GPU memory
- detection accuracy

## Deliverables
Create:
- `src/transformer/swin_block.py`
- `src/models/fixed_hybrid_detector.py`
- `scripts/train_fixed_hybrid.py`
- `scripts/evaluate_fixed_hybrid.py`
- `configs/fixed_hybrid_p4.yaml`
- `configs/fixed_hybrid_p3p4.yaml`
- `results/fixed_hybrid_results.json`
- `results/fixed_hybrid_report.md`

## Scientific purpose
These fixed variants are essential controls.

The adaptive method must eventually be compared against:
1. CNN-only at similar accuracy/compute conditions;
2. fixed CNN+Transformer at similar compute;
3. adaptive CNN+Transformer.

Do not call the fixed hybrid "our proposed method."
