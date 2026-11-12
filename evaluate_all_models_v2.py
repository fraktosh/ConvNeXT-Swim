"""COCO val2017 evaluation for all four detectors.

Changes versus evaluate_all_models.py:
  * ImageNet normalization is applied at evaluation (the original eval transform
    only did to_tensor; training uses ImageNet mean/std per DOCUMENTATION.md Entry 5).
    Use --no_normalize to reproduce the old behaviour for an A/B comparison.
  * --models selects which models to evaluate.
  * --max_images evaluates only the first N images (quick check). Quick/A-B runs
    write to results/quick_*.json and never touch the final result files.
  * The transform is a module-level class so DataLoader workers can pickle it on Windows.
Everything else (decoding, NMS, top-100, COCOeval) is unchanged.
"""
import os
import sys
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
import torchvision.ops as box_ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.cnn_detector import CNNDetector
from src.models.fixed_hybrid_detector import FixedHybridDetector
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.retinanet_loss import RetinaNetLoss
from src.data.coco import RealCOCODataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EvalTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, img, target):
        w, h = img.size
        img = TF.resize(img, (640, 640))
        img = TF.to_tensor(img)
        if self.normalize:
            img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        img_id = target[0]['image_id'] if len(target) > 0 else -1
        return img, {'image_id': img_id, 'orig_size': (h, w)}


def collate_fn(batch):
    return tuple(zip(*batch))


def get_postprocessor(device):
    loss_fn = RetinaNetLoss(num_classes=80).to(device)

    def postprocess(cls_logits, bbox_preds, image_shapes):
        anchors = loss_fn.generate_anchors(cls_logits)
        B = cls_logits[0].shape[0]

        cls_flat, bbox_flat = [], []
        for c, b in zip(cls_logits, bbox_preds):
            c = c.view(B, -1, 80, c.shape[2], c.shape[3]).permute(0, 3, 4, 1, 2).reshape(B, -1, 80)
            b = b.view(B, -1, 4, b.shape[2], b.shape[3]).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
            cls_flat.append(c)
            bbox_flat.append(b)

        cls_flat = torch.cat(cls_flat, dim=1).sigmoid()
        bbox_flat = torch.cat(bbox_flat, dim=1)

        results = []
        for i in range(B):
            scores = cls_flat[i]
            boxes = loss_fn.box_coder.decode_single(bbox_flat[i], anchors)

            orig_h, orig_w = image_shapes[i]
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 640) * (orig_w / 640.0)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 640) * (orig_h / 640.0)

            # xyxy to xywh
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            keep = scores > 0.05
            boxes_all, scores_all, labels_all = [], [], []
            for cls_id in range(80):
                cls_keep = keep[:, cls_id]
                if not cls_keep.any():
                    continue
                s = scores[cls_keep, cls_id]
                b = boxes[cls_keep]
                b_xyxy = torch.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], dim=1)
                nms_keep = box_ops.nms(b_xyxy, s, 0.5)
                boxes_all.append(b[nms_keep])
                scores_all.append(s[nms_keep])
                labels_all.append(torch.full((len(nms_keep),), cls_id, dtype=torch.int64, device=device))

            if len(boxes_all) > 0:
                img_boxes = torch.cat(boxes_all)
                img_scores = torch.cat(scores_all)
                img_labels = torch.cat(labels_all)
                if len(img_scores) > 100:
                    img_scores, topk = img_scores.topk(100)
                    img_boxes = img_boxes[topk]
                    img_labels = img_labels[topk]
                results.append({'boxes': img_boxes, 'scores': img_scores, 'labels': img_labels})
            else:
                results.append({
                    'boxes': torch.empty((0, 4), device=device),
                    'scores': torch.empty((0,), device=device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=device)
                })
        return results
    return postprocess


def evaluate_model(model_name, model, checkpoint_path, dataset, dataloader, cocoGt, device, restrict_ids, normalized):
    n_images = len(dataloader.dataset)
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name} on COCO 2017 Validation Set ({n_images} images, normalize={normalized})...")
    print(f"{'=' * 60}")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint {checkpoint_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    model.eval()
    postprocess = get_postprocessor(device)
    coco_results = []
    seen_ids = set()

    start_eval = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images_tensor = torch.stack(images).to(device)
            with torch.amp.autocast('cuda'):
                out = model(images_tensor)
                cls_logits, bbox_preds = out[0], out[1]

            image_shapes = [t['orig_size'] for t in targets]
            results = postprocess(cls_logits, bbox_preds, image_shapes)

            for t, r in zip(targets, results):
                img_id = t['image_id']
                if img_id == -1:
                    continue
                seen_ids.add(int(img_id))
                boxes = r['boxes'].cpu().numpy()
                scores = r['scores'].cpu().numpy()
                labels = r['labels'].cpu().numpy()

                for b, s, l in zip(boxes, scores, labels):
                    coco_cat_id = dataset.contiguous_id_to_coco_id[int(l)]
                    coco_results.append({
                        "image_id": int(img_id),
                        "category_id": int(coco_cat_id),
                        "bbox": [round(float(coord), 2) for coord in b],
                        "score": round(float(s), 4)
                    })

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - start_eval
                print(f"Processed [{batch_idx + 1}/{len(dataloader)}] batches "
                      f"({len(coco_results)} predictions so far, {elapsed:.1f}s)")

    print(f"Total predictions generated: {len(coco_results)}")

    if len(coco_results) > 0:
        cocoDt = cocoGt.loadRes(coco_results)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        if restrict_ids:
            cocoEval.params.imgIds = sorted(seen_ids)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
    else:
        print("No predictions exceeded score threshold 0.05. Metrics are 0.")
        stats = [0.0] * 12

    def pct(x):
        return round(float(x) * 100, 2) if x is not None else 0.0

    return {
        "Model": model_name,
        "Dataset": "COCO val2017",
        "Images evaluated": len(seen_ids) if restrict_ids else n_images,
        "ImageNet normalization at eval": normalized,
        "AP": pct(stats[0]), "AP50": pct(stats[1]), "AP75": pct(stats[2]),
        "APs": pct(stats[3]), "APm": pct(stats[4]), "APl": pct(stats[5]),
        "Total Predictions": len(coco_results)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='cnn,p4,p3p4,adaptive',
                    help='comma list from: cnn,p4,p3p4,adaptive')
    ap.add_argument('--max_images', type=int, default=0, help='0 = all 5000 images')
    ap.add_argument('--no_normalize', action='store_true', help='old behaviour (no ImageNet normalization)')
    args = ap.parse_args()

    normalize = not args.no_normalize
    quick = args.max_images > 0 or args.no_normalize
    tag = f"_n{args.max_images or 'all'}_{'norm' if normalize else 'raw'}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    val_dir = 'datasets/coco/val2017'
    val_ann = 'datasets/coco/annotations/instances_val2017.json'

    print("Loading COCO ground-truth annotations...")
    cocoGt = COCO(val_ann)
    dataset = RealCOCODataset(root=val_dir, annFile=val_ann, transform=EvalTransform(normalize))
    loader_ds = Subset(dataset, list(range(args.max_images))) if args.max_images > 0 else dataset
    dataloader = DataLoader(loader_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)

    registry = {
        'cnn': ("CNN-only", lambda: CNNDetector(num_classes=80, pretrained=False),
                "checkpoints/cnn_baseline/final_model.pth", "results/cnn_eval_results.json"),
        'p4': ("Fixed Swin P4", lambda: FixedHybridDetector(variant='P4', num_classes=80, pretrained=False),
               "checkpoints/fixed_p4/final_model.pth", "results/fixed_p4_eval_results.json"),
        'p3p4': ("Fixed Swin P3/P4", lambda: FixedHybridDetector(variant='P3P4', num_classes=80, pretrained=False),
                 "checkpoints/fixed_p3p4/final_model.pth", "results/fixed_p3p4_eval_results.json"),
        'adaptive': ("Adaptive routing", lambda: AdaptiveHybridDetector(num_classes=80, pretrained=False),
                     "checkpoints/adaptive/final_model.pth", "results/adaptive_eval_results.json"),
    }
    keys = [k.strip() for k in args.models.split(',') if k.strip()]
    for k in keys:
        if k not in registry:
            raise SystemExit(f"Unknown model key '{k}'. Choose from {list(registry)}")

    all_metrics = {}
    for k in keys:
        name, ctor, ckpt, out_file = registry[k]
        model = ctor().to(device)
        m = evaluate_model(name, model, ckpt, dataset, dataloader, cocoGt, device,
                           restrict_ids=args.max_images > 0, normalized=normalize)
        all_metrics[name] = m
        if quick:
            out_file = f"results/quick_{k}{tag}.json"
        with open(out_file, 'w') as f:
            json.dump(m, f, indent=4)
        print(f"Saved evaluation metrics to {out_file}")
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY OF DETECTION METRICS")
    print("=" * 60)
    print(json.dumps(all_metrics, indent=2))

    if not quick and len(keys) == len(registry):
        with open('results/all_eval_summary.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print("Saved results/all_eval_summary.json")
    else:
        print("Quick/partial run: results/all_eval_summary.json was NOT modified.")


if __name__ == '__main__':
    main()
