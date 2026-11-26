"""Data prep step 1: convert the Annotated Potholes dataset to YOLO format.

Reads Pascal VOC XML from ``potholes/annotated-images`` and the split list in
``potholes/splits.json`` (train / test; test is used as val), and writes images
plus YOLO box labels to ``potholes_yolo/``.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# ── EDIT THESE PATHS to match your structure ──────────────────────
POTHOLE_DIR     = "./potholes"
ANNOTATED_DIR   = "./potholes/annotated-images"
SPLITS_JSON     = "./potholes/splits.json"
OUTPUT_DIR      = "./potholes_yolo"
# ──────────────────────────────────────────────────────────────────

# Pothole = class 0 in its own dataset (will be remapped to 0 in unified)
CLASS_MAP = {"pothole": 0}

def parse_voc_xml(xml_path: str):
    """Parse Pascal VOC XML and return (image_filename, width, height, list of boxes)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size     = root.find("size")
    width    = int(size.find("width").text)
    height   = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert to YOLO format (normalized cx, cy, w, h)
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        w  = (xmax - xmin) / width
        h  = (ymax - ymin) / height

        class_id = CLASS_MAP.get(name, 0)  # default to 0 if name varies
        boxes.append((class_id, cx, cy, w, h))

    return filename, boxes

def convert_pothole_dataset():
    out = Path(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    # splits.json has "train" and "test" — map test → val (80/20 split)
    split_map = {}
    for xml_name in splits.get("train", []):
        split_map[xml_name] = "train"
    for xml_name in splits.get("test", []):
        split_map[xml_name] = "val"   # use test split as val

    annotated = Path(ANNOTATED_DIR)
    converted  = 0
    skipped    = 0

    for xml_file in annotated.glob("*.xml"):
        split = split_map.get(xml_file.name, "train")  # default train if not in splits

        try:
            filename, boxes = parse_voc_xml(str(xml_file))
        except Exception as e:
            print(f"  Skipping {xml_file.name}: {e}")
            skipped += 1
            continue

        # Find the corresponding image
        img_path = None
        candidates = [annotated / filename] + [
            annotated / (Path(filename).stem + ext)
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        ]
        for candidate in candidates:
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # A label without an image is useless for training, so skip it
            print(f"  Image not found for {xml_file.name}, skipping")
            skipped += 1
            continue
        shutil.copy(img_path, out / "images" / split / img_path.name)

        # Write YOLO label
        label_path = out / "labels" / split / (img_path.stem + ".txt")
        with open(label_path, "w") as lf:
            for box in boxes:
                lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        converted += 1

    # Write data.yaml for pothole-only dataset
    import yaml
    # No "path" key: Ultralytics then uses the folder holding data.yaml,
    # which keeps the dataset portable across machines.
    data_yaml = {
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "pothole"},
    }
    with open(out / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nPothole conversion complete:")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped}")
    for split in ["train", "val", "test"]:
        imgs = len(list((out / "images" / split).iterdir()))
        lbls = len(list((out / "labels" / split).iterdir()))
        print(f"  {split:6s} → {imgs} images, {lbls} labels")

if __name__ == "__main__":
    convert_pothole_dataset()