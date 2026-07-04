"""
VOC to COCO Format Converter for DETR
======================================
Converts PASCAL VOC 2007+2012 annotations to COCO JSON format.

Training set: VOC2007 trainval + VOC2012 trainval (~16,551 images)
Validation set: VOC2007 test (~4,952 images)

Output structure (required by DETR):
    voc_coco_format/
        train2017/              <- training images
        val2017/                <- validation images
        annotations/
            instances_train2017.json
            instances_val2017.json

Usage:
    cd DETR-PROJECT
    python scripts/voc2coco.py
"""

import os
import sys
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ==============================================================
# CONFIGURATION — adjust these paths if your layout differs
# ==============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # DETR-PROJECT/

VOC_ROOT = PROJECT_ROOT / "data" / "VOCdevkit"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "voc_coco_format"

# Source directories
VOC2007_DIR = VOC_ROOT / "VOC2007"
VOC2012_DIR = VOC_ROOT / "VOC2012"

# Output directories (DETR hardcodes these exact names)
TRAIN_IMG_DIR = OUTPUT_ROOT / "train2017"
VAL_IMG_DIR = OUTPUT_ROOT / "val2017"
ANN_DIR = OUTPUT_ROOT / "annotations"

# ==============================================================
# VOC class names → COCO category IDs (1-indexed)
# ==============================================================
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Map class name → category ID (1-indexed for COCO compatibility)
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


def read_image_ids(split_file: Path) -> list:
    """Read image IDs from a VOC split file (e.g., trainval.txt)."""
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def parse_voc_annotation(xml_path: Path) -> dict:
    """
    Parse a single VOC XML annotation file.
    Returns dict with: filename, width, height, list of objects (class, bbox).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        # Skip difficult objects (standard VOC practice)
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        class_name = obj.find("name").text
        if class_name not in CLASS_TO_ID:
            print(f"  Warning: Unknown class '{class_name}' in {xml_path}, skipping")
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert VOC [xmin, ymin, xmax, ymax] → COCO [x, y, width, height]
        coco_bbox = [
            xmin,                # x (left)
            ymin,                # y (top)
            xmax - xmin,         # width
            ymax - ymin,         # height
        ]

        # Sanity check: skip invalid boxes
        if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
            print(f"  Warning: Invalid bbox in {xml_path}: {coco_bbox}, skipping")
            continue

        area = coco_bbox[2] * coco_bbox[3]

        objects.append({
            "class_name": class_name,
            "category_id": CLASS_TO_ID[class_name],
            "bbox": coco_bbox,
            "area": area,
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def build_coco_dataset(image_ids: list, voc_dirs: list, output_img_dir: Path, split_name: str) -> dict:
    """
    Build a COCO-format dataset dict and copy images.

    Args:
        image_ids: list of (image_id_str, voc_dir) tuples
        voc_dirs: not used (included in image_ids)
        output_img_dir: where to copy images (train2017/ or val2017/)
        split_name: "train" or "val" (for logging)

    Returns:
        COCO-format dict with 'images', 'annotations', 'categories'
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CLASS_TO_ID.items()],
    }

    image_global_id = 0
    annotation_global_id = 0
    skipped_no_file = 0
    skipped_no_annotations = 0
    copied_images = 0

    print(f"\nProcessing {split_name} split ({len(image_ids)} images)...")

    for idx, (img_id, voc_dir) in enumerate(image_ids):
        # Find annotation XML
        xml_path = voc_dir / "Annotations" / f"{img_id}.xml"
        if not xml_path.exists():
            skipped_no_file += 1
            continue

        # Parse annotation
        ann_data = parse_voc_annotation(xml_path)

        # Skip images with no valid objects (after filtering difficult ones)
        if len(ann_data["objects"]) == 0:
            skipped_no_annotations += 1
            continue

        # Find source image
        src_img_path = voc_dir / "JPEGImages" / ann_data["filename"]
        if not src_img_path.exists():
            # Try with .jpg extension
            src_img_path = voc_dir / "JPEGImages" / f"{img_id}.jpg"
            if not src_img_path.exists():
                skipped_no_file += 1
                continue

        # Use a unique filename to avoid collisions between VOC2007 and VOC2012
        # (some image IDs overlap between the two datasets)
        voc_year = "voc2007" if "VOC2007" in str(voc_dir) else "voc2012"
        dst_filename = f"{voc_year}_{img_id}.jpg"
        dst_img_path = output_img_dir / dst_filename

        # Copy image (skip if already exists)
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        copied_images += 1

        # Add image entry
        image_global_id += 1
        coco["images"].append({
            "id": image_global_id,
            "file_name": dst_filename,
            "width": ann_data["width"],
            "height": ann_data["height"],
        })

        # Add annotation entries
        for obj in ann_data["objects"]:
            annotation_global_id += 1
            coco["annotations"].append({
                "id": annotation_global_id,
                "image_id": image_global_id,
                "category_id": obj["category_id"],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0,
            })

        # Progress logging
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(image_ids)} images...")

    print(f"  Done: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"  Copied: {copied_images} images")
    if skipped_no_file > 0:
        print(f"  Skipped (missing file): {skipped_no_file}")
    if skipped_no_annotations > 0:
        print(f"  Skipped (no valid objects): {skipped_no_annotations}")

    return coco


def main():
    print("=" * 60)
    print("VOC to COCO Converter for DETR")
    print("=" * 60)

    # Verify source directories exist
    for d in [VOC2007_DIR, VOC2012_DIR]:
        if not d.exists():
            print(f"ERROR: {d} not found!")
            sys.exit(1)

    # Create output directories (should already exist, but just in case)
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Build TRAINING set: VOC2007 trainval + VOC2012 trainval
    # -------------------------------------------------------
    train_ids = []

    # VOC2007 trainval
    voc07_trainval = read_image_ids(VOC2007_DIR / "ImageSets" / "Main" / "trainval.txt")
    train_ids.extend([(img_id, VOC2007_DIR) for img_id in voc07_trainval])
    print(f"VOC2007 trainval: {len(voc07_trainval)} image IDs")

    # VOC2012 trainval
    voc12_trainval = read_image_ids(VOC2012_DIR / "ImageSets" / "Main" / "trainval.txt")
    train_ids.extend([(img_id, VOC2012_DIR) for img_id in voc12_trainval])
    print(f"VOC2012 trainval: {len(voc12_trainval)} image IDs")

    print(f"Total training IDs: {len(train_ids)}")

    train_coco = build_coco_dataset(train_ids, None, TRAIN_IMG_DIR, "train")

    # Save training annotations
    train_json_path = ANN_DIR / "instances_train2017.json"
    with open(train_json_path, "w") as f:
        json.dump(train_coco, f)
    print(f"Saved: {train_json_path}")

    # -------------------------------------------------------
    # Build VALIDATION set: VOC2007 test
    # -------------------------------------------------------
    val_ids = []

    # VOC2007 test
    voc07_test = read_image_ids(VOC2007_DIR / "ImageSets" / "Main" / "test.txt")
    val_ids.extend([(img_id, VOC2007_DIR) for img_id in voc07_test])
    print(f"\nVOC2007 test: {len(voc07_test)} image IDs")

    val_coco = build_coco_dataset(val_ids, None, VAL_IMG_DIR, "val")

    # Save validation annotations
    val_json_path = ANN_DIR / "instances_val2017.json"
    with open(val_json_path, "w") as f:
        json.dump(val_coco, f)
    print(f"Saved: {val_json_path}")

    # -------------------------------------------------------
    # Final verification
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check image counts
    n_train_imgs = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
    n_val_imgs = len(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"Images in train2017/:  {n_train_imgs}")
    print(f"Images in val2017/:    {n_val_imgs}")

    # Check annotation counts
    with open(train_json_path) as f:
        train_data = json.load(f)
    with open(val_json_path) as f:
        val_data = json.load(f)

    print(f"Train JSON - images: {len(train_data['images'])}, annotations: {len(train_data['annotations'])}, categories: {len(train_data['categories'])}")
    print(f"Val JSON   - images: {len(val_data['images'])}, annotations: {len(val_data['annotations'])}, categories: {len(val_data['categories'])}")

    # Check category IDs are 1-indexed
    train_cat_ids = set(a["category_id"] for a in train_data["annotations"])
    val_cat_ids = set(a["category_id"] for a in val_data["annotations"])
    print(f"Train category ID range: {min(train_cat_ids)} to {max(train_cat_ids)}")
    print(f"Val category ID range:   {min(val_cat_ids)} to {max(val_cat_ids)}")

    # Check bbox format (sample)
    sample_ann = train_data["annotations"][0]
    bbox = sample_ann["bbox"]
    print(f"Sample bbox [x, y, w, h]: {bbox}")
    assert bbox[2] > 0 and bbox[3] > 0, "ERROR: bbox width/height should be positive!"

    # Check file_name consistency
    sample_img = train_data["images"][0]
    sample_path = TRAIN_IMG_DIR / sample_img["file_name"]
    assert sample_path.exists(), f"ERROR: {sample_path} not found!"

    print("\n✓ All checks passed! Dataset is ready for DETR training.")
    print(f"\nNext step: use --coco_path {OUTPUT_ROOT.relative_to(PROJECT_ROOT)} when running DETR")


if __name__ == "__main__":
    main()
