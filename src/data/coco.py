import os
import copy
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as TF

from src.data.coco_categories import (
    COCO_CATEGORY_IDS,
    COCO_ID_TO_CONTIGUOUS,
    CONTIGUOUS_TO_COCO_ID,
    coco_id_to_contiguous,
    contiguous_to_coco_id
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def coco_train_transform(img, target, image_size: Tuple[int, int] = (640, 640)) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Standard training transform for COCO:
    1. Resizes image to image_size (640, 640).
    2. Converts to Tensor in range [0, 1].
    3. Normalizes with ImageNet mean and std.
    4. Filters invalid annotations (iscrowd == 1, area <= 0, width <= 0, height <= 0).
    5. Converts bounding boxes [x, y, w, h] to [x1, y1, x2, y2] scaled to image_size.
    6. Clamps box coordinates to [0, width] and [0, height].
    7. Maps category_id to contiguous [0, 79] label.
    """
    w, h = img.size
    new_h, new_w = image_size
    
    img = TF.resize(img, (new_h, new_w))
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    boxes = []
    labels = []
    
    for obj in target:
        # Ignore crowd annotations during training
        if obj.get('iscrowd', 0) == 1:
            continue
            
        bbox = obj.get('bbox', [])
        if len(bbox) != 4:
            continue
            
        bw, bh = bbox[2], bbox[3]
        if bw <= 0 or bh <= 0:
            continue
            
        x1 = bbox[0] * (new_w / w)
        y1 = bbox[1] * (new_h / h)
        x2 = (bbox[0] + bw) * (new_w / w)
        y2 = (bbox[1] + bh) * (new_h / h)
        
        # Clamp to image boundaries
        x1 = max(0.0, min(float(new_w), float(x1)))
        y1 = max(0.0, min(float(new_h), float(y1)))
        x2 = max(0.0, min(float(new_w), float(x2)))
        y2 = max(0.0, min(float(new_h), float(y2)))
        
        if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
            raw_cat_id = obj['category_id']
            # If already converted or in COCO format:
            cont_id = coco_id_to_contiguous(raw_cat_id) if raw_cat_id in COCO_ID_TO_CONTIGUOUS else raw_cat_id
            if 0 <= cont_id < 80:
                boxes.append([x1, y1, x2, y2])
                labels.append(cont_id)
                
    if len(boxes) == 0:
        boxes_t = torch.empty((0, 4), dtype=torch.float32)
        labels_t = torch.empty((0,), dtype=torch.int64)
    else:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        
    return img, {'boxes': boxes_t, 'labels': labels_t}

def coco_eval_transform(img, target, image_size: Tuple[int, int] = (640, 640)) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Evaluation transform:
    Resizes and normalizes the image; returns image metadata (image_id, original_size).
    """
    w, h = img.size
    new_h, new_w = image_size
    img = TF.resize(img, (new_h, new_w))
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    img_id = target[0]['image_id'] if len(target) > 0 and 'image_id' in target[0] else -1
    return img, {'image_id': img_id, 'orig_size': (h, w)}

class RealCOCODataset(CocoDetection):
    """
    Dataset wrapper around official torchvision CocoDetection.
    Integrates verified contiguous category mapping.
    """
    def __init__(self, root: str, annFile: str, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        
        # Canonical mappings from src.data.coco_categories
        self.coco_ids = COCO_CATEGORY_IDS
        self.coco_id_to_contiguous_id = COCO_ID_TO_CONTIGUOUS
        self.contiguous_id_to_coco_id = CONTIGUOUS_TO_COCO_ID

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        image_id = self.ids[index]
        target = copy.deepcopy(target)
        
        # Ensure image_id is stored in target objects
        for t in target:
            t['image_id'] = image_id
            
        if self.transform is not None:
            img, target = self.transform(img, target)
        else:
            # Fallback default: ensure category_id is contiguous
            for t in target:
                if t['category_id'] in self.coco_id_to_contiguous_id:
                    t['category_id'] = self.coco_id_to_contiguous_id[t['category_id']]
                    
        # Ensure image_id is preserved in target dictionary if transformed into dict
        if isinstance(target, dict) and 'image_id' not in target:
            target['image_id'] = image_id
            
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
