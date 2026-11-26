"""Ethics and transparency checks for a set of detections.

Produces an ``EthicsReport`` with: how many detections need human review,
class / severity distributions, simple bias warnings (class mix far from the
training mix), a disclaimer, and dataset provenance.
"""
from dataclasses import dataclass
from collections import Counter

from detector import AUDIT_LOG_PATH, REVIEW_THRESHOLD

DATASET_PROVENANCE = (
    "Model trained on: Ultralytics crack-seg (4,029 images, AGPL-3.0) + "
    "Annotated Potholes Dataset by Chitholian (665 images, Public Domain). "
    "Known limitations: dataset primarily covers daylight conditions on paved "
    "roads. Performance may degrade on unpaved roads, wet surfaces, night-time "
    "images, or road types not represented in training data (e.g. highways vs local roads)."
)

EXPECTED_DIST = {"crack": 0.65, "pothole": 0.35}

@dataclass
class EthicsReport:
    total_detections:     int
    flagged_for_review:   int
    class_distribution:   dict
    severity_distribution: dict
    bias_warnings:        list
    disclaimer:           str
    dataset_provenance:   str
    model_version:        str
    audit_trail_path:     str

def generate_ethics_report(detections, model_version="yolov8n-seg-v1",
                            audit_log_path=AUDIT_LOG_PATH):
    """Build an ``EthicsReport`` for the detections of one image."""
    total    = len(detections)
    flagged  = sum(1 for d in detections if d.flagged_for_review)
    cls_dist = dict(Counter(d.class_name for d in detections))
    sev_dist = dict(Counter(d.severity   for d in detections))

    # Bias checks
    bias_warnings = []
    if total > 0:
        for cls, expected in EXPECTED_DIST.items():
            actual = cls_dist.get(cls, 0) / total
            if actual > expected * 2.5:
                bias_warnings.append(
                    f"OVER-DETECTION: '{cls}' at {actual:.0%} vs expected {expected:.0%}. "
                    f"Possible false positives - check image quality or lighting."
                )
            elif actual < expected * 0.15 and expected > 0.1:
                bias_warnings.append(
                    f"UNDER-DETECTION: '{cls}' at {actual:.0%} vs expected ~ {expected:.0%}. "
                    f"Model may be missing this defect type in this image."
                )

    if flagged > 0:
        disclaimer = (
            f"DISCLAIMER: {flagged}/{total} detection(s) have confidence < {REVIEW_THRESHOLD:.2f} "
            f"and are FLAGGED FOR MANDATORY HUMAN REVIEW. These detections must NOT "
            f"be used for repair prioritisation without verification by a qualified engineer."
        )
    elif total == 0:
        disclaimer = "No defects detected. If defects are visible, try lowering the confidence threshold."
    else:
        disclaimer = (
            f"All {total} detection(s) meet the confidence threshold. "
            f"Severity scores are ADVISORY - final repair decisions require human judgment."
        )

    return EthicsReport(
        total_detections=total,
        flagged_for_review=flagged,
        class_distribution=cls_dist,
        severity_distribution=sev_dist,
        bias_warnings=bias_warnings,
        disclaimer=disclaimer,
        dataset_provenance=DATASET_PROVENANCE,
        model_version=model_version,
        audit_trail_path=audit_log_path,
    )