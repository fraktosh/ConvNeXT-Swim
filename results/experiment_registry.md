# Experiment Registry

### Table I: Detection and Computational Comparison (Empirically Measured on COCO 2017 Validation Set)

*Benchmarking Protocol*: Input resolution 640x640, batch size 1 (inference latency over 200 measured iterations following 50 warm-up runs, CUDA-synchronized on NVIDIA RTX 3050 Laptop GPU, FP16 mixed precision). Detection metrics evaluated using official pycocotools `COCOeval` on all 5,000 images of `COCO val2017`.

| Model | Dataset | Input size | Params | Latency | FPS | Peak GPU Mem | AP | AP50 | AP75 | Status / Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CNN-only** | COCO val2017 | 640x640 | 32.27M | 38.17 ms | 26.2 | 397.5 MB | 0.0 | 0.0 | 0.0 | Trained ~50 batches (exploding activations in stage 1/2 produced negative logits) |
| **Fixed Swin P4** | COCO val2017 | 640x640 | 34.04M | 36.43 ms | 27.4 | 541.9 MB | 0.0 | 0.0 | 0.0 | Trained ~50 batches (similar activation collapse producing negative logits) |
| **Fixed Swin P3/P4** | COCO val2017 | 640x640 | 34.49M | 38.85 ms | 25.7 | 680.4 MB | 0.0 | 0.0 | 0.0 | Evaluated 495,200 detections; short training produced no AP overlap |
| **Adaptive routing** | COCO val2017 | 640x640 | 39.90M | 49.43 ms | 20.2 | 867.0 MB | 0.0 | 0.0 | 0.0 | Trained on toy proxy loss (`det_loss = c.mean() + b.mean()`), not RetinaNetLoss |
