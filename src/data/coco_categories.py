"""
Canonical COCO Category Definitions and Contiguous ID Mapping.

Official MS COCO defines 80 object categories whose IDs range non-contiguously from 1 to 90.
This module provides a bijective, verified mapping between official COCO category IDs and
0-indexed contiguous training labels [0, 79].
"""

from typing import Dict, List, Tuple

# Official 80 category IDs in MS COCO (sorted ascending)
COCO_CATEGORY_IDS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

COCO_CATEGORY_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Bijective mappings
COCO_ID_TO_CONTIGUOUS: Dict[int, int] = {
    coco_id: cont_id for cont_id, coco_id in enumerate(COCO_CATEGORY_IDS)
}

CONTIGUOUS_TO_COCO_ID: Dict[int, int] = {
    cont_id: coco_id for cont_id, coco_id in enumerate(COCO_CATEGORY_IDS)
}

CONTIGUOUS_TO_NAME: Dict[int, str] = {
    i: name for i, name in enumerate(COCO_CATEGORY_NAMES)
}

COCO_ID_TO_NAME: Dict[int, str] = {
    coco_id: name for coco_id, name in zip(COCO_CATEGORY_IDS, COCO_CATEGORY_NAMES)
}

NUM_CLASSES: int = len(COCO_CATEGORY_IDS) # 80

def coco_id_to_contiguous(coco_id: int) -> int:
    """Map official COCO category ID (1-90) to contiguous training label [0, 79]."""
    if coco_id not in COCO_ID_TO_CONTIGUOUS:
        raise KeyError(f"Invalid COCO category ID: {coco_id}. Must be one of {COCO_CATEGORY_IDS}")
    return COCO_ID_TO_CONTIGUOUS[coco_id]

def contiguous_to_coco_id(cont_id: int) -> int:
    """Map contiguous prediction label [0, 79] back to official COCO category ID (1-90)."""
    if cont_id not in CONTIGUOUS_TO_COCO_ID:
        raise KeyError(f"Invalid contiguous label: {cont_id}. Must be in range [0, {NUM_CLASSES - 1}]")
    return CONTIGUOUS_TO_COCO_ID[cont_id]
