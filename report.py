"""Save the results for one image to ``reports/``.

Writes five files named ``<image>_<timestamp>_*``: annotated JPG, heatmap JPG,
JSON report, CSV (one row per detection) and an ethics TXT summary.
"""
import json, csv, cv2
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

def generate_report(image_path, annotated_image, heatmap_image,
                    detections, ethics_report, explanation_text,
                    output_dir="reports"):
    """Write all report files and return a dict of their paths."""
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(str(image_path)).stem
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Annotated image
    ann_path = str(out / f"{stem}_{ts}_annotated.jpg")
    cv2.imwrite(ann_path, annotated_image)

    # 2. Heatmap image
    heat_path = str(out / f"{stem}_{ts}_heatmap.jpg")
    cv2.imwrite(heat_path, heatmap_image)

    # 3. JSON report
    report_data = {
        "source_image":    str(image_path),
        "processed_at":    ts,
        "total_defects":   len(detections),
        "severity_summary": ethics_report.severity_distribution,
        "class_summary":    ethics_report.class_distribution,
        "detections":       [asdict(d) for d in detections],
        "explainability":   explanation_text,
        "ethics": {
            "flagged_for_review":  ethics_report.flagged_for_review,
            "bias_warnings":       ethics_report.bias_warnings,
            "disclaimer":          ethics_report.disclaimer,
            "dataset_provenance":  ethics_report.dataset_provenance,
            "model_version":       ethics_report.model_version,
        },
    }
    json_path = str(out / f"{stem}_{ts}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 4. CSV
    csv_path = str(out / f"{stem}_{ts}_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name","confidence","severity","area_px","area_cm2",
                    "flagged_for_review","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                    "severity_reason","confidence_reason","timestamp"])
        for d in detections:
            w.writerow([
                d.class_name, f"{d.confidence:.3f}", d.severity,
                f"{d.area_px:.1f}", f"{d.area_cm2:.2f}", d.flagged_for_review,
                *[f"{v:.1f}" for v in d.bbox_xyxy],
                d.severity_reason, d.confidence_reason, d.timestamp,
            ])

    # 5. Ethics text
    eth_path = str(out / f"{stem}_{ts}_ethics.txt")
    with open(eth_path, "w", encoding="utf-8") as f:
        f.write(f"ETHICS & TRANSPARENCY REPORT\n{'='*50}\n")
        f.write(f"Model version    : {ethics_report.model_version}\n")
        f.write(f"Processed at     : {ts}\n")
        f.write(f"Total detections : {ethics_report.total_detections}\n")
        f.write(f"Flagged          : {ethics_report.flagged_for_review}\n")
        f.write(f"Severity dist.   : {ethics_report.severity_distribution}\n")
        f.write(f"\nDisclaimer:\n{ethics_report.disclaimer}\n")
        if ethics_report.bias_warnings:
            f.write(f"\nBias Warnings:\n")
            for w in ethics_report.bias_warnings:
                f.write(f"  - {w}\n")
        f.write(f"\nDataset provenance:\n{ethics_report.dataset_provenance}\n")
        f.write(f"\nExplainability:\n{explanation_text}\n")
        f.write(f"\nAudit trail: {ethics_report.audit_trail_path}\n")

    print(f"\nReports saved to: {output_dir}/")
    return {"annotated": ann_path, "heatmap": heat_path,
            "json": json_path, "csv": csv_path, "ethics": eth_path}