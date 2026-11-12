import os
import sys
import unittest
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.coco import coco_train_transform, coco_eval_transform, RealCOCODataset
from src.data.coco_categories import COCO_ID_TO_CONTIGUOUS, CONTIGUOUS_TO_COCO_ID

class TestDatasetTransform(unittest.TestCase):
    def test_train_transform_box_conversion_and_clamping(self):
        # Create a mock 1000x500 image
        img = Image.new('RGB', (1000, 500), color=(128, 128, 128))
        target = [
            # Valid box: person (coco_id=1)
            {'bbox': [100, 50, 200, 150], 'category_id': 1, 'iscrowd': 0},
            # Box with category > 79: toothbrush (coco_id=90)
            {'bbox': [300, 100, 100, 50], 'category_id': 90, 'iscrowd': 0},
            # Crowd annotation: should be skipped
            {'bbox': [50, 50, 100, 100], 'category_id': 3, 'iscrowd': 1},
            # Degenerate box: zero width, should be skipped
            {'bbox': [200, 200, 0, 50], 'category_id': 1, 'iscrowd': 0},
            # Box exceeding bounds: should be clamped
            {'bbox': [900, 400, 200, 200], 'category_id': 80, 'iscrowd': 0} # toaster (coco_id=80)
        ]
        
        img_t, target_t = coco_train_transform(img, target, image_size=(640, 640))
        
        # Check image shape
        self.assertEqual(img_t.shape, (3, 640, 640))
        
        # Expected boxes: 3 valid boxes (crowd and zero-width skipped)
        boxes = target_t['boxes']
        labels = target_t['labels']
        self.assertEqual(len(boxes), 3)
        self.assertEqual(len(labels), 3)
        
        # Check box 0: [100, 50, 200, 150] in (1000, 500) -> scaled to (640, 640)
        # x1 = 100 * (640/1000) = 64, y1 = 50 * (640/500) = 64
        # x2 = (100+200) * 0.64 = 192, y2 = (50+150) * 1.28 = 256
        self.assertAlmostEqual(boxes[0, 0].item(), 64.0, places=2)
        self.assertAlmostEqual(boxes[0, 1].item(), 64.0, places=2)
        self.assertAlmostEqual(boxes[0, 2].item(), 192.0, places=2)
        self.assertAlmostEqual(boxes[0, 3].item(), 256.0, places=2)
        self.assertEqual(labels[0].item(), COCO_ID_TO_CONTIGUOUS[1]) # person
        
        # Check box 1: category 90 (toothbrush)
        self.assertEqual(labels[1].item(), COCO_ID_TO_CONTIGUOUS[90])
        self.assertTrue(0 <= labels[1].item() < 80)
        
        # Check box 2: category 80 (toaster) clamped to 640
        self.assertLessEqual(boxes[2, 2].item(), 640.0)
        self.assertLessEqual(boxes[2, 3].item(), 640.0)
        self.assertEqual(labels[2].item(), COCO_ID_TO_CONTIGUOUS[80])

    def test_empty_target(self):
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        img_t, target_t = coco_train_transform(img, [], image_size=(640, 640))
        self.assertEqual(target_t['boxes'].shape, (0, 4))
        self.assertEqual(target_t['labels'].shape, (0,))

    def test_real_dataset_loader(self):
        val_ann = 'datasets/coco/annotations/instances_val2017.json'
        val_dir = 'datasets/coco/val2017'
        if not (os.path.exists(val_ann) and os.path.exists(val_dir)):
            self.skipTest("COCO val set not found")
            
        dataset = RealCOCODataset(root=val_dir, annFile=val_ann, transform=coco_train_transform)
        self.assertEqual(len(dataset), 5000)
        
        # Check first 5 samples
        for i in range(5):
            img_t, target_t = dataset[i]
            self.assertEqual(img_t.shape, (3, 640, 640))
            if len(target_t['labels']) > 0:
                self.assertTrue(torch.all(target_t['labels'] >= 0))
                self.assertTrue(torch.all(target_t['labels'] < 80))
                boxes = target_t['boxes']
                self.assertTrue(torch.all(boxes[:, 2] > boxes[:, 0]))
                self.assertTrue(torch.all(boxes[:, 3] > boxes[:, 1]))

if __name__ == '__main__':
    unittest.main()
