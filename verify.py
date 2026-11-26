"""Data prep step 5: sanity-check ``unified_dataset`` before training.

Prints data.yaml and image / label counts per split, and creates empty label
files for images that have none (YOLO treats them as background).
"""
from pathlib import Path
import yaml

DATASET_DIR = "./unified_dataset"

with open(f"{DATASET_DIR}/data.yaml") as f:
    cfg = yaml.safe_load(f)

print("data.yaml contents:")
print(yaml.dump(cfg, default_flow_style=False))

ds = Path(DATASET_DIR)
for split in ["train", "val", "test"]:
    imgs = list((ds / "images" / split).glob("*.*"))
    lbls = list((ds / "labels" / split).glob("*.txt"))
    empty_lbls = [l for l in lbls if l.stat().st_size == 0]
    print(f"{split:6s}: {len(imgs):4d} images | {len(lbls):4d} labels | {len(empty_lbls):4d} empty labels")

    # Check for images without labels
    img_stems = {i.stem for i in imgs}
    lbl_stems = {l.stem for l in lbls}
    no_label  = img_stems - lbl_stems
    if no_label:
        print(f"         ⚠ {len(no_label)} images have no label file — creating empty ones")
        for stem in no_label:
            (ds / "labels" / split / f"{stem}.txt").touch()