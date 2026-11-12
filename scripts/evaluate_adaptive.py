import os
import sys
import yaml
import json
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.data.coco import RealCOCODataset
from scripts.evaluate_all_models import eval_transform, collate_fn, get_postprocessor

def evaluate():
    print("Evaluating Adaptive-Hybrid Detector on COCO val2017...")
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = config['data']['val_dir']
    val_ann = config['data']['val_ann']
    
    dataset = RealCOCODataset(root=val_dir, annFile=val_ann, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
    cocoGt = COCO(val_ann)
    
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=False).to(device)
    model_path = os.path.join(config['training']['output_dir'], 'final_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint from {model_path}")
    else:
        print(f"Warning: Checkpoint {model_path} not found.")
        
    model.eval()
    postprocess = get_postprocessor(device)
    coco_results = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
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
                    
    print(f"Total predictions generated: {len(coco_results)}")
    if len(coco_results) > 0:
        cocoDt = cocoGt.loadRes(coco_results)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
    else:
        stats = [0.0] * 12
        
    metrics = {
        "Dataset": "COCO val2017",
        "Model": config['model']['name'],
        "AP": round(float(stats[0]) * 100, 2) if stats[0] is not None else 0.0,
        "AP50": round(float(stats[1]) * 100, 2) if stats[1] is not None else 0.0,
        "AP75": round(float(stats[2]) * 100, 2) if stats[2] is not None else 0.0,
        "APs": round(float(stats[3]) * 100, 2) if stats[3] is not None else 0.0,
        "APm": round(float(stats[4]) * 100, 2) if stats[4] is not None else 0.0,
        "APl": round(float(stats[5]) * 100, 2) if stats[5] is not None else 0.0,
        "Total Predictions": len(coco_results)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/adaptive_eval_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Evaluation finished! Saved to results/adaptive_eval_results.json")
    return metrics

if __name__ == '__main__':
    evaluate()
