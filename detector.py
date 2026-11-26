"""Road-defect detection core.

Flow for one image (``RoadDefectDetector.detect``):

    BGR image -> YOLOv8-seg -> per instance: mask polygon, box, class, confidence
              -> area (Shapely polygon area, falls back to box area; cm^2 if a
                 pixels-per-cm scale is set)
              -> severity + written reason (``classify_severity``)
              -> review flag when confidence < REVIEW_THRESHOLD
              -> one JSON line appended to the audit log

Settings come from environment variables (or a ``.env`` file): MODEL_PATH,
CONFIDENCE_THRESHOLD, REVIEW_THRESHOLD, AUDIT_LOG_PATH.
"""
from dotenv import load_dotenv
import os, cv2, time, json
import numpy as np
from dataclasses import dataclass, asdict
from shapely.geometry import Polygon

load_dotenv()

MODEL_PATH           = os.getenv("MODEL_PATH", "./runs/segment/runs/road_defect_seg6/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
REVIEW_THRESHOLD     = float(os.getenv("REVIEW_THRESHOLD", 0.40))
AUDIT_LOG_PATH       = os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")

CLASS_NAMES = {0: "crack", 1: "pothole"}

SEVERITY_THRESHOLDS = {
    "crack":   {"low": 300,  "moderate": 1500},
    "pothole": {"low": 800,  "moderate": 3000},
}

SEVERITY_COLORS = {
    "low":      (0, 200, 0),
    "moderate": (0, 165, 255),
    "high":     (0, 0, 220),
}

@dataclass
class Detection:
    class_id:           int
    class_name:         str
    confidence:         float
    bbox_xyxy:          list
    mask_polygon:       list
    area_px:            float
    area_cm2:           float
    severity:           str
    severity_reason:    str
    confidence_reason:  str
    flagged_for_review: bool
    timestamp:          str
    image_path:         str

# ── Severity logic ────────────────────────────────────────────────────────────
def classify_severity(class_name, area_px, confidence):
    """Map a defect's pixel area to low / moderate / high, with a written reason."""
    t     = SEVERITY_THRESHOLDS.get(class_name, {"low": 300, "moderate": 1500})
    low_t = t["low"]
    hi_t  = t["moderate"]

    if area_px < low_t:
        sev    = "low"
        reason = (f"Defect pixel area ({area_px:.0f}px^2) is below the low-severity "
                  f"threshold ({low_t}px^2) for '{class_name}'. Routine monitoring advised.")
    elif area_px < hi_t:
        sev    = "moderate"
        reason = (f"Defect pixel area ({area_px:.0f}px^2) is between low ({low_t}px^2) "
                  f"and high ({hi_t}px^2) thresholds. Schedule inspection within 30 days.")
    else:
        sev    = "high"
        reason = (f"Defect pixel area ({area_px:.0f}px^2) exceeds high-severity threshold "
                  f"({hi_t}px^2) for '{class_name}'. Immediate repair recommended.")

    if confidence < REVIEW_THRESHOLD and sev == "high":
        reason += (f" Severity kept at HIGH, but model confidence ({confidence:.2f}) "
                   f"is below {REVIEW_THRESHOLD}: mandatory human verification required.")
    return sev, reason

def confidence_reason(class_name, confidence, area_px):
    """Explain the model's confidence level in plain language."""
    if confidence >= 0.75:
        quality = "high"
        basis   = "strong texture contrast and clear boundary edges"
    elif confidence >= 0.50:
        quality = "moderate"
        basis   = "partial texture match with some boundary ambiguity"
    else:
        quality = "low"
        basis   = "weak signal due to possible shadow, occlusion, or noise"
    return (f"Confidence {confidence:.2f} ({quality}). Model identified '{class_name}' "
            f"based on {basis}. Mask area: {area_px:.0f}px^2.")

# ── Main detector class ───────────────────────────────────────────────────────
class RoadDefectDetector:
    """Wraps the YOLOv8-seg model: detection, annotation and audit logging."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        px_per_cm=None,
        audit_log_path=AUDIT_LOG_PATH,
    ):
        from ultralytics import YOLO
        self.model              = YOLO(model_path)
        self.conf_threshold     = confidence_threshold
        self.px_per_cm          = px_per_cm
        self.audit_log_path     = audit_log_path

    def detect(self, image_bgr, image_path="unknown"):
        """Run the model on a BGR image and return a list of ``Detection``."""
        results    = self.model(image_bgr, conf=self.conf_threshold, verbose=False)
        detections = []

        if results[0].masks is None:
            self._write_audit([], image_path)
            return detections

        masks   = results[0].masks.xy
        boxes   = results[0].boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        bboxes  = boxes.xyxy.cpu().numpy()

        for mask_pts, cls_id, conf, bbox in zip(masks, classes, confs, bboxes):
            class_name = CLASS_NAMES.get(int(cls_id), "crack")

            # Area from segmentation mask polygon
            area_px = 0.0
            if len(mask_pts) >= 3:
                try:
                    area_px = Polygon(mask_pts).area
                except Exception:
                    pass
            if area_px == 0.0:
                area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            area_cm2 = (area_px / (self.px_per_cm ** 2)) if self.px_per_cm else 0.0
            sev, sev_reason  = classify_severity(class_name, area_px, float(conf))
            conf_reason_str  = confidence_reason(class_name, float(conf), area_px)

            detections.append(Detection(
                class_id=int(cls_id),
                class_name=class_name,
                confidence=float(conf),
                bbox_xyxy=bbox.tolist(),
                mask_polygon=mask_pts.tolist() if len(mask_pts) else [],
                area_px=float(area_px),
                area_cm2=float(area_cm2),
                severity=sev,
                severity_reason=sev_reason,
                confidence_reason=conf_reason_str,
                flagged_for_review=bool(float(conf) < REVIEW_THRESHOLD),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                image_path=str(image_path),
            ))

        self._write_audit(detections, image_path)
        return detections

    def annotate_image(self, image, detections):
        """Draw masks, boxes and labels, coloured by severity, on a copy of the image."""
        vis = image.copy()
        for det in detections:
            color = SEVERITY_COLORS.get(det.severity, (128, 128, 128))
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

            # Mask overlay
            if det.mask_polygon:
                pts     = np.array(det.mask_polygon, dtype=np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
                cv2.polylines(vis, [pts], True, color, 2)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            flag   = " [REVIEW]" if det.flagged_for_review else ""
            label  = (f"{det.class_name} | {det.severity.upper()} "
                      f"| {det.confidence:.2f} | {det.area_px:.0f}px^2{flag}")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return vis

    def _write_audit(self, detections, image_path):
        entry = {
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_path":      str(image_path),
            "conf_threshold":  self.conf_threshold,
            "total":           len(detections),
            "detections":      [asdict(d) for d in detections],
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")