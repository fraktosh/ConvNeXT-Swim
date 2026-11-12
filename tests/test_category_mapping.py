import os
import sys
import unittest
import json
import torch
from pycocotools.coco import COCO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.coco_categories import (
    COCO_CATEGORY_IDS,
    COCO_CATEGORY_NAMES,
    COCO_ID_TO_CONTIGUOUS,
    CONTIGUOUS_TO_COCO_ID,
    NUM_CLASSES,
    coco_id_to_contiguous,
    contiguous_to_coco_id
)

class TestCOCOCategoryMapping(unittest.TestCase):
    def setUp(self):
        self.val_ann = 'datasets/coco/annotations/instances_val2017.json'
        if os.path.exists(self.val_ann):
            self.coco = COCO(self.val_ann)
        else:
            self.coco = None

    def test_total_classes(self):
        """Test exactly 80 categories exist."""
        self.assertEqual(len(COCO_CATEGORY_IDS), 80)
        self.assertEqual(len(COCO_CATEGORY_NAMES), 80)
        self.assertEqual(NUM_CLASSES, 80)

    def test_bijection(self):
        """Test bijection between COCO IDs and contiguous IDs."""
        self.assertEqual(len(COCO_ID_TO_CONTIGUOUS), 80)
        self.assertEqual(len(CONTIGUOUS_TO_COCO_ID), 80)
        
        # Test range of contiguous IDs is strictly [0, 79]
        contiguous_set = set(COCO_ID_TO_CONTIGUOUS.values())
        self.assertEqual(contiguous_set, set(range(80)))
        
        # Test round-trip identity
        for coco_id in COCO_CATEGORY_IDS:
            cont_id = coco_id_to_contiguous(coco_id)
            self.assertTrue(0 <= cont_id < 80)
            self.assertEqual(contiguous_to_coco_id(cont_id), coco_id)
            
        for cont_id in range(80):
            coco_id = contiguous_to_coco_id(cont_id)
            self.assertIn(coco_id, COCO_CATEGORY_IDS)
            self.assertEqual(coco_id_to_contiguous(coco_id), cont_id)

    def test_categories_over_79_not_merged(self):
        """Confirm categories with original IDs > 79 are distinct and not clamped."""
        over_79 = [
            (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (84, 'book'),
            (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddy bear'),
            (89, 'hair drier'), (90, 'toothbrush')
        ]
        assigned_contiguous = []
        for cat_id, expected_name in over_79:
            cont_id = coco_id_to_contiguous(cat_id)
            self.assertTrue(0 <= cont_id < 80, f"ID {cat_id} mapped out of bounds: {cont_id}")
            self.assertNotIn(cont_id, assigned_contiguous, f"Category {cat_id} collision at contiguous {cont_id}")
            assigned_contiguous.append(cont_id)
            
            # Verify name
            self.assertEqual(COCO_CATEGORY_NAMES[cont_id], expected_name)
            
        # Verify all 10 have unique contiguous labels
        self.assertEqual(len(set(assigned_contiguous)), len(over_79))

    def test_against_official_coco_annotations(self):
        """Confirm definition matches instances_val2017.json exactly."""
        if self.coco is None:
            self.skipTest("COCO annotation file not found")
            
        coco_cat_ids = sorted(self.coco.getCatIds())
        self.assertEqual(COCO_CATEGORY_IDS, coco_cat_ids)
        
        for cat_id in coco_cat_ids:
            cat_obj = self.coco.loadCats(cat_id)[0]
            cont_id = coco_id_to_contiguous(cat_id)
            self.assertEqual(COCO_CATEGORY_NAMES[cont_id], cat_obj['name'])

    def test_invalid_category_error(self):
        """Confirm mapping rejects invalid category IDs (gaps and out-of-range)."""
        gaps = [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91, 100]
        for gap_id in gaps:
            with self.assertRaises(KeyError):
                coco_id_to_contiguous(gap_id)
                
        with self.assertRaises(KeyError):
            contiguous_to_coco_id(-1)
        with self.assertRaises(KeyError):
            contiguous_to_coco_id(80)

    def test_multi_category_batch_conversion(self):
        """Test realistic multi-category target conversions."""
        raw_coco_labels = [1, 3, 80, 90, 44, 18] # person, car, toaster, toothbrush, bottle, horse
        mapped = [coco_id_to_contiguous(c) for c in raw_coco_labels]
        self.assertTrue(all(0 <= m < 80 for m in mapped))
        recovered = [contiguous_to_coco_id(m) for m in mapped]
        self.assertEqual(recovered, raw_coco_labels)

if __name__ == '__main__':
    unittest.main()
