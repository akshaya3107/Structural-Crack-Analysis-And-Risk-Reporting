"""EigenCAM heatmap and plain-language explanation for YOLOv8-seg.

Flow:
    forward hooks on backbone layers 0-9 -> run the image through the model
    -> take the deepest 4-D activation -> first principal component (EigenCAM),
       sign-corrected so hot = strong activation
    -> resize, colour-map and blend onto the image
    -> per detection: mean CAM value inside its box -> "visual support" text

If no activation can be used, falls back to a confidence-weighted Gaussian
heatmap built from the detection boxes.
"""
import cv2
import numpy as np
import torch

def generate_explanation(yolo_model, image_bgr, detections, save_dir="explanations"):
    """Return a dict with the blended heatmap, its saved path and a text summary."""
    import os, time
    os.makedirs(save_dir, exist_ok=True)

    h, w      = image_bgr.shape[:2]
    img_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    # ── Safe activation extraction ────────────────────────────────────────────
    activations = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # Only capture tensor outputs, skip tuples/lists from detection heads
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                activations.append(output.detach().float().cpu())
        return hook_fn

    # Register hooks only on Conv2d layers in backbone (safe layers)
    hooks   = []
    model   = yolo_model.model
    for i, layer in enumerate(model.model):
        # Only hook backbone layers (0-9), skip detection head layers
        if i <= 9:
            for submodule in layer.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    hooks.append(submodule.register_forward_hook(make_hook(f"layer{i}")))
                    break  # one hook per layer is enough

    # Run inference with hooks active
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    # Match the model's device and precision (e.g. CUDA / fp16), otherwise the
    # first layer fails and we silently fall back to the box-only heatmap.
    try:
        param  = next(model.parameters())
        tensor = tensor.to(device=param.device, dtype=param.dtype)
    except StopIteration:
        pass
    with torch.no_grad():
        try:
            yolo_model.model(tensor)
        except Exception:
            pass  # activations may still be captured before any error

    # Remove all hooks immediately after inference
    for h_ in hooks:
        h_.remove()

    # ── Build CAM from captured activations ───────────────────────────────────
    cam_resized = None

    # Use the deepest activation captured (most semantic)
    for act in reversed(activations):
        try:
            acts = act[0].numpy()       # (C, H_feat, W_feat)
            if acts.shape[0] == 0 or acts.shape[1] == 0:
                continue

            # EigenCAM: first principal component
            reshaped = acts.reshape(acts.shape[0], -1).astype(np.float64)
            energy   = np.abs(reshaped).mean(axis=0)   # overall activation strength per pixel
            reshaped -= reshaped.mean(axis=1, keepdims=True)

            try:
                _, _, Vt     = np.linalg.svd(reshaped, full_matrices=False)
                cam_flat     = Vt[0].astype(np.float32)
            except np.linalg.LinAlgError:
                cam_flat     = reshaped.mean(axis=0).astype(np.float32)

            # SVD returns the component with an arbitrary sign; orient it so that
            # high values follow strong activations (otherwise the map can be inverted).
            if np.dot(cam_flat - cam_flat.mean(), energy - energy.mean()) < 0:
                cam_flat = -cam_flat

            cam = cam_flat.reshape(acts.shape[1], acts.shape[2])
            cam = np.maximum(cam, 0)

            if cam.max() > 0:
                cam = cam / cam.max()

            cam_resized = cv2.resize(cam, (w, h))
            break   # successfully built CAM — stop
        except Exception:
            continue

    # Fallback: detection-box based heatmap if CAM failed
    if cam_resized is None or cam_resized.max() == 0:
        cam_resized = _bbox_fallback_cam(detections, h, w)

    # ── Blend heatmap onto image ──────────────────────────────────────────────
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * np.clip(cam_resized, 0, 1)), cv2.COLORMAP_JET
    )
    blended = cv2.addWeighted(image_bgr, 0.55, heatmap_color, 0.45, 0)

    # Save
    ts           = time.strftime("%Y%m%d_%H%M%S")
    heatmap_path = os.path.join(save_dir, f"eigencam_{ts}.jpg")
    cv2.imwrite(heatmap_path, blended)

    # ── Build explanation text ────────────────────────────────────────────────
    max_act  = float(cam_resized.max())
    mean_act = float(cam_resized.mean())
    peak_loc = np.unravel_index(cam_resized.argmax(), cam_resized.shape)
    peak_x   = peak_loc[1] / w * 100
    peak_y   = peak_loc[0] / h * 100
    attn_pat = "concentrated" if mean_act < 0.25 else "distributed"

    det_texts = []
    for i, det in enumerate(detections, 1):
        # ROI activation inside detection bbox
        x1 = max(0, int(det.bbox_xyxy[0]))
        y1 = max(0, int(det.bbox_xyxy[1]))
        x2 = min(w - 1, int(det.bbox_xyxy[2]))
        y2 = min(h - 1, int(det.bbox_xyxy[3]))
        roi_act = float(cam_resized[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0

        support = (
            "strongly supported by visual evidence"   if roi_act > 0.50 else
            "moderately supported by visual evidence" if roi_act > 0.25 else
            "weakly supported - review recommended"
        )
        flag = ("\n  FLAGGED FOR HUMAN REVIEW - confidence below threshold."
                if det.flagged_for_review else "")

        det_texts.append(
            f"Detection {i} - {det.class_name.upper()} ({det.severity} severity)\n"
            f"  Confidence     : {det.confidence:.2f}\n"
            f"  Area           : {det.area_px:.0f} px^2"
            + (f" = {det.area_cm2:.1f} cm^2" if det.area_cm2 else "") + "\n"
            f"  CAM activation : {roi_act:.3f} in detection region\n"
            f"  Visual support : {support}\n"
            f"  Why detected   : {det.confidence_reason}\n"
            f"  Severity basis : {det.severity_reason}"
            + flag
        )

    summary = (
        f"EigenCAM Explainability Report\n"
        f"{'-' * 45}\n"
        f"Peak activation : {max_act:.3f} at ({peak_x:.0f}%, {peak_y:.0f}%) of image\n"
        f"Mean activation : {mean_act:.3f} - {attn_pat} attention pattern\n"
        f"Total detections: {len(detections)}\n\n"
        + "\n\n".join(det_texts)
    )

    return {
        "heatmap_path":    heatmap_path,
        "heatmap_bgr":     blended,
        "summary":         summary,
        "peak_activation": max_act,
        "cam_array":       cam_resized,
    }


def _bbox_fallback_cam(detections, h, w):
    """
    Fallback: build a heatmap purely from detection bounding boxes
    weighted by confidence. Used when CAM extraction fails.
    """
    cam = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        x1 = max(0, int(det.bbox_xyxy[0]))
        y1 = max(0, int(det.bbox_xyxy[1]))
        x2 = min(w - 1, int(det.bbox_xyxy[2]))
        y2 = min(h - 1, int(det.bbox_xyxy[3]))
        if x2 > x1 and y2 > y1:
            # Gaussian-like weight - stronger at centre of bbox
            roi           = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            cy            = (y2 - y1) // 2
            cx            = (x2 - x1) // 2
            sigma         = max(cy, cx, 1)
            Y, X          = np.ogrid[:y2-y1, :x2-x1]
            gauss         = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            cam[y1:y2, x1:x2] = np.maximum(cam[y1:y2, x1:x2], gauss * det.confidence)

    if cam.max() > 0:
        cam /= cam.max()
    return cam