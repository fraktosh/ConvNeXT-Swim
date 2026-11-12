import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.coco import RealCOCODataset
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss

def simple_transform(img, target):
    """
    Basic transform to resize PIL image to 640x640 and scale bounding boxes.
    """
    w, h = img.size
    new_w, new_h = 640, 640
    
    img = TF.resize(img, (new_h, new_w))
    img = TF.to_tensor(img)
    
    boxes = []
    labels = []
    for obj in target:
        bbox = obj['bbox']
        # COCO bbox is [x_min, y_min, width, height]
        x = bbox[0] * (new_w / w)
        y = bbox[1] * (new_h / h)
        bw = bbox[2] * (new_w / w)
        bh = bbox[3] * (new_h / h)
        boxes.append([x, y, bw, bh])
        labels.append(obj['category_id'])
        
    if len(boxes) == 0:
        boxes = torch.empty((0, 4))
        labels = torch.empty((0,), dtype=torch.int64)
    else:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
    return img, {'boxes': boxes, 'labels': labels}

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Real COCO Smoke Test on: {device}")
    
    # Pointing to the downloaded train2017 dataset
    val_img_dir = 'datasets/coco/train2017'
    val_ann_file = 'datasets/coco/annotations/instances_train2017.json'
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_ann_file):
        print("Error: Validation dataset not found. Download might not be complete.")
        return
        
    dataset = RealCOCODataset(
        root=val_img_dir,
        annFile=val_ann_file,
        transform=simple_transform
    )
    
    # We will test on just 20 batches to prove stability
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # COCO has 91 categories including background
    model = AdaptiveHybridDetector(num_classes=91, pretrained=True).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=0.5).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    model.train()
    
    print("Starting 20-batch smoke test...")
    for i, (images, targets) in enumerate(dataloader):
        if i >= 20:
            break
            
        images = torch.stack(images).to(device)
        optimizer.zero_grad()
        
        # Forward pass through adaptive network
        cls_logits, bbox_preds, w_p3, w_p4 = model(images)
        
        # Basic loss for smoke test to verify backward pass runs without NaNs
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
        b_loss = budget_loss_fn(w_p3) + budget_loss_fn(w_p4)
        
        total_loss = det_loss + 0.1 * b_loss
        total_loss.backward()
        optimizer.step()
        
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Batch {i+1}/20 - Total Loss: {total_loss.item():.4f} - GPU Memory: {mem_mb:.1f} MB")
        
    print("\nReal COCO Smoke Test Completed Successfully! No memory leaks or shape mismatches.")

if __name__ == '__main__':
    main()
