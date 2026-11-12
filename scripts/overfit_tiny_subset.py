import os
import sys
import yaml
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.cnn_detector import CNNDetector
from src.data.coco import RealCOCODataset, coco_train_transform, coco_eval_transform, collate_fn
from src.data.coco_categories import CONTIGUOUS_TO_COCO_ID
from src.losses.retinanet_loss import RetinaNetLoss
from scripts.evaluate_all_models import get_postprocessor

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nPhase 4: Minimal Reproducible Training Test (Overfit 8 Images)\n{'='*60}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    val_dir = 'datasets/coco/val2017'
    val_ann = 'datasets/coco/annotations/instances_val2017.json'
    
    # 8 images with ground truth objects
    cocoGt = COCO(val_ann)
    all_img_ids = cocoGt.getImgIds()
    selected_img_ids = []
    for img_id in all_img_ids:
        ann_ids = cocoGt.getAnnIds(imgIds=img_id, iscrowd=0)
        if len(ann_ids) > 0:
            selected_img_ids.append(img_id)
        if len(selected_img_ids) >= 8:
            break
            
    print(f"Selected {len(selected_img_ids)} training images with GT objects: {selected_img_ids}")
    
    # Create dataset with selected images
    train_dataset = RealCOCODataset(root=val_dir, annFile=val_ann, transform=coco_train_transform)
    # Filter dataset.ids to match selected_img_ids
    train_dataset.ids = selected_img_ids
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = CNNDetector(num_classes=80, pretrained=True).to(device)
    loss_fn = RetinaNetLoss(num_classes=80).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    epochs = 40
    print(f"\nStarting training overfit test for {epochs} epochs...")
    initial_loss = None
    
    for epoch in range(epochs):
        model.train()
        epoch_cls, epoch_box = 0.0, 0.0
        
        for images, targets in dataloader:
            images = torch.stack(images).to(device)
            targets = [{k: (v.to(device) if hasattr(v, 'to') else v) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                cls_logits, bbox_preds = model(images)
                cls_loss, box_loss = loss_fn(cls_logits, bbox_preds, targets)
                loss = cls_loss + box_loss
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_cls += cls_loss.item()
            epoch_box += box_loss.item()
            
        avg_loss = epoch_cls + epoch_box
        if initial_loss is None:
            initial_loss = avg_loss
            print(f"Epoch 0 (Initial): Cls={epoch_cls:.4f}, Box={epoch_box:.4f}, Total={avg_loss:.4f}")
            
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}: Cls={epoch_cls:.4f}, Box={epoch_box:.4f}, Total={avg_loss:.4f}")
            
    print(f"\nLoss reduction: {initial_loss:.4f} -> {avg_loss:.4f} ({(1 - avg_loss/initial_loss)*100:.1f}% reduction)")
    
    # Save checkpoint
    os.makedirs('checkpoints/diagnostic_test', exist_ok=True)
    save_path = 'checkpoints/diagnostic_test/overfit_tiny_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, save_path)
    print(f"Saved diagnostic checkpoint to {save_path}")
    
    # Evaluate on the 8 images
    print("\nRunning COCOeval on the 8 overfitted images...")
    eval_dataset = RealCOCODataset(root=val_dir, annFile=val_ann, transform=coco_eval_transform)
    eval_dataset.ids = selected_img_ids
    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    postprocess = get_postprocessor(device)
    coco_results = []
    
    with torch.no_grad():
        for images, targets in eval_loader:
            images = torch.stack(images).to(device)
            with torch.amp.autocast('cuda'):
                cls_logits, bbox_preds = model(images)
            image_shapes = [t['orig_size'] for t in targets]
            results = postprocess(cls_logits, bbox_preds, image_shapes)
            
            for t, r in zip(targets, results):
                img_id = t['image_id']
                boxes = r['boxes'].cpu().numpy()
                scores = r['scores'].cpu().numpy()
                labels = r['labels'].cpu().numpy()
                for b, s, l in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": int(img_id),
                        "category_id": int(CONTIGUOUS_TO_COCO_ID[int(l)]),
                        "bbox": [round(float(c), 2) for c in b],
                        "score": round(float(s), 4)
                    })
                    
    print(f"Generated {len(coco_results)} predictions on 8 images.")
    if len(coco_results) > 0:
        cocoDt = cocoGt.loadRes(coco_results)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = selected_img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
        mAP = stats[0] * 100
        print(f"\n=> Overfit Validation Result: AP = {mAP:.2f}%, AP50 = {stats[1]*100:.2f}%, AP75 = {stats[2]*100:.2f}%")
        if mAP > 0.0:
            print("SUCCESS: Model achieves NONZERO AP on the training subset!")
        else:
            print("Predictions generated, but IoU overlap below 0.50 threshold.")
    else:
        print("No predictions exceeded score threshold.")

if __name__ == '__main__':
    main()
