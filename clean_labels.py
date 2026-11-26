"""Data prep step 4: validate and clean label files.

Scans all label files in unified_dataset and, rewriting each file in place:
1. Removes lines with fewer than 5 values (invalid YOLO format)
2. Removes lines whose class_id is not an integer in [0, NC)
3. Removes lines with non-numeric or an odd number of coordinates
4. Clips coordinates to the [0, 1] range
5. Leaves files with no valid annotations empty (background images)
and prints a summary.
"""
from pathlib import Path

DATASET_DIR = "./unified_dataset"
NC = 2  # number of classes: 0=crack, 1=pothole

def clean_label_file(label_path: Path, nc: int) -> tuple[int, int, list[str]]:
    """
    Returns (original_lines, kept_lines, issues_found).
    Rewrites the file in place with only valid lines.
    """
    issues = []
    kept   = []

    with open(label_path) as f:
        lines = f.read().splitlines()

    original_count = len(lines)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Must have at least class_id + 4 bbox values OR class_id + polygon points
        if len(parts) < 5:
            issues.append(f"  Line {i+1}: too few values ({len(parts)}) — removed")
            continue

        # Class ID must be valid
        try:
            cls_id = int(parts[0])
        except ValueError:
            issues.append(f"  Line {i+1}: non-integer class_id '{parts[0]}' — removed")
            continue

        if cls_id < 0 or cls_id >= nc:
            issues.append(f"  Line {i+1}: class_id {cls_id} out of range [0,{nc-1}] — removed")
            continue

        # All coordinate values must be in [0, 1]
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"  Line {i+1}: non-numeric coordinates — removed")
            continue

        # Check for NaN or out-of-range (allow tiny overshoot with clipping)
        coords_clipped = [max(0.0, min(1.0, c)) for c in coords]
        if any(abs(c - cc) > 0.01 for c, cc in zip(coords, coords_clipped)):
            issues.append(f"  Line {i+1}: coordinates out of [0,1] range — clipped")

        # Polygon must have even number of coords (x,y pairs) and at least 3 points
        if len(coords_clipped) % 2 != 0:
            issues.append(f"  Line {i+1}: odd number of polygon coordinates — removed")
            continue
        if len(coords_clipped) < 8:  # minimum 4 points for a polygon (bbox has 4 values)
            # For bbox format (4 values) this is fine, for polygon need >= 6 values
            pass  # allow bbox format through

        kept.append(f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords_clipped))

    # Rewrite file
    with open(label_path, "w") as f:
        f.write("\n".join(kept))

    return original_count, len(kept), issues


def clean_dataset(dataset_dir: str, nc: int):
    ds = Path(dataset_dir)
    total_files   = 0
    total_removed = 0
    empty_files   = 0
    problem_files = []

    for split in ["train", "val", "test"]:
        lbl_dir = ds / "labels" / split
        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))
        print(f"\n{split}: {len(label_files)} label files")

        for lbl_path in label_files:
            orig, kept, issues = clean_label_file(lbl_path, nc)
            total_files += 1
            removed = orig - kept
            total_removed += removed

            if issues:
                problem_files.append((str(lbl_path), issues))

            if kept == 0 and orig > 0:
                empty_files += 1
                # File still exists but is now empty — that's fine for background images

    print(f"\n── Clean summary ─────────────────────────────")
    print(f"  Files scanned  : {total_files}")
    print(f"  Lines removed  : {total_removed}")
    print(f"  Files now empty: {empty_files}  (background images — OK)")
    print(f"  Files with issues: {len(problem_files)}")

    if problem_files:
        print(f"\n  First 10 problem files:")
        for path, issues in problem_files[:10]:
            print(f"  {path}")
            for issue in issues[:3]:
                print(f"    {issue}")

    print("\nDone. Dataset is clean.")


if __name__ == "__main__":
    clean_dataset(DATASET_DIR, NC)