"""Data prep step 3: convert box labels to polygons.

Converts YOLO bbox labels (class cx cy w h) → polygon format
(class x1y1 x2y1 x2y2 x1y2) so all labels are consistent
for YOLOv8-seg training.
"""
from pathlib import Path

DATASET_DIR = "./unified_dataset"
POTHOLE_PREFIX = "potholes_yolo_"  # prefix added during merge

def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> str:
    """Convert normalized bbox to 4-point polygon (clockwise from top-left)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Clamp to [0, 1]
    x1, x2 = max(0.0, x1), min(1.0, x2)
    y1, y2 = max(0.0, y1), min(1.0, y2)

    # 4-point polygon: top-left, top-right, bottom-right, bottom-left
    return f"{x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"

def is_bbox_format(parts: list[str]) -> bool:
    """True if line has exactly 5 values (class + 4 bbox coords)."""
    return len(parts) == 5

def fix_labels(dataset_dir: str):
    ds = Path(dataset_dir)
    total_converted = 0
    total_skipped   = 0
    total_files     = 0

    for split in ["train", "val", "test"]:
        lbl_dir = ds / "labels" / split
        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))
        print(f"\n{split}: checking {len(label_files)} files...")

        for lbl_path in label_files:
            with open(lbl_path) as f:
                lines = f.read().splitlines()

            if not lines or all(l.strip() == "" for l in lines):
                continue

            new_lines    = []
            file_changed = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if is_bbox_format(parts):
                    # Convert bbox → polygon
                    cls_id = parts[0]
                    cx, cy, w, h = float(parts[1]), float(parts[2]), \
                                   float(parts[3]), float(parts[4])
                    poly_coords = bbox_to_polygon(cx, cy, w, h)
                    new_lines.append(f"{cls_id} {poly_coords}")
                    file_changed  = True
                    total_converted += 1
                else:
                    # Already polygon format — keep as-is
                    new_lines.append(line)
                    total_skipped += 1

            if file_changed:
                with open(lbl_path, "w") as f:
                    f.write("\n".join(new_lines))
                total_files += 1

    print(f"\n── Conversion summary ────────────────────────")
    print(f"  Files updated    : {total_files}")
    print(f"  Lines converted  : {total_converted}  (bbox → polygon)")
    print(f"  Lines kept as-is : {total_skipped}   (already polygon)")
    print(f"\nAll labels are now in polygon format. Ready to train.")

if __name__ == "__main__":
    fix_labels(DATASET_DIR)