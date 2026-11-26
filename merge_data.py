"""Data prep step 2: merge crack-seg and potholes_yolo into ``unified_dataset``.

Copies images and labels into images/{train,val,test} and labels/{...},
prefixes file names with the source name to avoid clashes, remaps class ids to
0 = crack, 1 = pothole, and writes ``unified_dataset/data.yaml``.
"""
import shutil, yaml
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────
CRACK_SEG_DIR  = "./crack-seg"
POTHOLE_YOLO   = "./potholes_yolo"      # output from Step 1
OUTPUT_DIR     = "./unified_dataset"
# ──────────────────────────────────────────────────────────────────

# Unified class map
CLASS_MAP = {
    "crack":   0,
    "pothole": 1,
}

SOURCES = [
    {
        "dir":     CRACK_SEG_DIR,
        "classes": {0: "crack"},     # crack-seg only has class 0 = crack
    },
    {
        "dir":     POTHOLE_YOLO,
        "classes": {0: "pothole"},   # pothole dataset only has class 0 = pothole
    },
]

def merge():
    out = Path(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        src      = Path(source["dir"])
        src_cls  = source["classes"]   # {old_id: class_name}
        prefix   = src.name.replace("-", "_").replace(" ", "_")

        if not src.exists():
            print(f"SKIPPING {source['dir']} — folder not found")
            continue

        print(f"\nMerging: {source['dir']}")

        for split_src, split_dst in [
            ("train", "train"),
            ("valid", "val"),
            ("val",   "val"),
            ("test",  "test"),
        ]:
            img_dir = src / "images" / split_src
            lbl_dir = src / "labels" / split_src

            if not img_dir.exists():
                continue

            img_files = [
                f for f in img_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            print(f"  {split_src:6s} → {len(img_files)} images")

            for img_file in img_files:
                # Prefix filename to avoid collisions between datasets
                new_img_name = f"{prefix}_{img_file.name}"
                new_lbl_name = f"{prefix}_{img_file.stem}.txt"

                # Copy image
                shutil.copy(
                    img_file,
                    out / "images" / split_dst / new_img_name
                )

                # Remap and copy label
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                new_lbl_path = out / "labels" / split_dst / new_lbl_name

                if lbl_file.exists():
                    new_lines = []
                    with open(lbl_file) as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_id    = int(parts[0])
                            cls_name  = src_cls.get(old_id, "crack")
                            new_id    = CLASS_MAP[cls_name]
                            new_lines.append(
                                f"{new_id} " + " ".join(parts[1:])
                            )
                    with open(new_lbl_path, "w") as of:
                        of.write("\n".join(new_lines))
                else:
                    # No label = background image, write empty file
                    new_lbl_path.touch()

    # Write unified data.yaml
    # No "path" key: Ultralytics then uses the folder holding data.yaml,
    # which keeps the dataset portable across machines.
    data_yaml = {
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_MAP),
        "names": {v: k for k, v in CLASS_MAP.items()},
    }
    with open(out / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Final summary
    print("\n── Unified dataset summary ──")
    for split in ["train", "val", "test"]:
        imgs = len(list((out / "images" / split).iterdir()))
        lbls = len(list((out / "labels" / split).iterdir()))
        print(f"  {split:6s} → {imgs} images, {lbls} labels")
    print(f"\ndata.yaml written to: {OUTPUT_DIR}/data.yaml")
    print("Ready to train.")

if __name__ == "__main__":
    merge()