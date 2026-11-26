"""Streamlit web app for road-defect detection.

Flow:
    upload image -> "Analyse Image"
      -> detector.detect()            masks, area, severity, review flags, audit log
      -> detector.annotate_image()    tab 1: annotated result + severity counts
      -> explainability.generate_explanation()   tab 2: EigenCAM heatmap (optional)
      -> tab 3: per-detection report
      -> ethics.generate_ethics_report()          tab 4: disclaimer, bias, provenance
      -> report.generate_report()     saves JPG / JSON / CSV / TXT to reports/ (optional)

Run:  streamlit run app.py
"""
import os

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

# Fix Windows UTF-8 — env var only, never rewrap sys.stdout in Streamlit
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"]       = "1"

MODEL_PATH = os.getenv("MODEL_PATH",
                       "./runs/segment/runs/road_defect_seg6/weights/best.pt")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide",
)

# ── Load model once (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    from detector import RoadDefectDetector
    return RoadDefectDetector(model_path=MODEL_PATH)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛣️ Road Defect Detection System")
st.markdown(
    "Detects **potholes** and **cracks** with severity scoring, "
    "area measurement, EigenCAM explainability, and ethical AI reporting."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                             help="Lower = more detections, higher = fewer but more certain")
    px_per_cm   = st.number_input("Pixels per cm (0 = skip cm² calc)", 0.0, 1000.0, 0.0,
                                   help="Measure a known reference object in your image")
    save_report = st.checkbox("Save report to /reports folder", value=True)
    show_heatmap = st.checkbox("Generate EigenCAM heatmap", value=True)

    st.markdown("---")
    st.markdown("**⚖️ Ethical Use Notice**")
    st.caption(
        "This tool provides advisory outputs only. "
        "All repair decisions must be made by qualified engineers. "
        "Flagged detections require human verification."
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a road image", type=["jpg", "jpeg", "png"],
    help="Upload a photo of a road surface to detect defects"
)

if uploaded is not None:
    # Convert to BGR
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")
        st.stop()
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    if st.button("🔍 Analyse Image", type="primary", use_container_width=True):
        with st.spinner("Running detection..."):

            # ── Detection ─────────────────────────────────────────────────────
            detector = load_detector()
            detector.conf_threshold = conf_thresh
            detector.px_per_cm = float(px_per_cm) if px_per_cm > 0 else None

            detections = detector.detect(image_bgr, image_path=uploaded.name)
            annotated  = detector.annotate_image(image_bgr, detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        from ethics import generate_ethics_report
        ethics = generate_ethics_report(detections)
        xai = None  # filled in by the EigenCAM tab when enabled

        # ── Results tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📸 Annotated Result",
            "🔥 EigenCAM Heatmap",
            "📋 Detection Report",
            "⚖️ Ethics & Transparency",
        ])

        with tab1:
            st.image(annotated_rgb, caption="Detected defects", use_container_width=True)

            # Severity summary
            if detections:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total defects",   len(detections))
                col2.metric("🔴 High",   sum(1 for d in detections if d.severity == "high"))
                col3.metric("🟠 Moderate", sum(1 for d in detections if d.severity == "moderate"))
                col4.metric("🟢 Low",    sum(1 for d in detections if d.severity == "low"))
            else:
                st.success("✅ No defects detected. Try lowering the confidence threshold if defects are visible.")

        with tab2:
            if show_heatmap:
                with st.spinner("Generating EigenCAM heatmap..."):
                    from explainability import generate_explanation
                    xai = generate_explanation(detector.model, image_bgr, detections)
                    heatmap_rgb = cv2.cvtColor(xai["heatmap_bgr"], cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, caption="EigenCAM — red/warm areas = model focus regions",
                         use_container_width=True)
                st.text_area("Full XAI analysis", xai["summary"], height=300)
            else:
                st.info("Enable 'Generate EigenCAM heatmap' in the sidebar to see this.")

        with tab3:
            if not detections:
                st.info("No detections to report.")
            else:
                sev_emoji = {"low": "🟢", "moderate": "🟠", "high": "🔴"}
                for i, d in enumerate(detections, 1):
                    emoji = sev_emoji.get(d.severity, "⚪")
                    flag  = " 🚩 FLAGGED FOR REVIEW" if d.flagged_for_review else ""

                    with st.expander(
                        f"{emoji} Detection {i}: {d.class_name.title()} — "
                        f"{d.severity.upper()}{flag}", expanded=True
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence", f"{d.confidence:.2f}")
                        c2.metric("Area (px²)",  f"{d.area_px:.0f}")
                        c3.metric("Area (cm²)",  f"{d.area_cm2:.1f}" if d.area_cm2 else "N/A")

                        if d.flagged_for_review:
                            st.error("⚠️ Confidence below threshold — mandatory human review required.")

                        st.markdown("**Why the model detected this:**")
                        st.info(d.confidence_reason)

                        st.markdown("**Severity reasoning:**")
                        st.warning(d.severity_reason)

                        st.markdown("**Bounding box:**")
                        st.code(f"x1={d.bbox_xyxy[0]:.0f}, y1={d.bbox_xyxy[1]:.0f}, "
                                f"x2={d.bbox_xyxy[2]:.0f}, y2={d.bbox_xyxy[3]:.0f}")

        with tab4:
            st.subheader("Disclaimer")
            if ethics.flagged_for_review > 0:
                st.error(ethics.disclaimer)
            else:
                st.success(ethics.disclaimer)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Class distribution")
                st.json(ethics.class_distribution)
            with col2:
                st.subheader("Severity distribution")
                st.json(ethics.severity_distribution)

            if ethics.bias_warnings:
                st.subheader("⚠️ Bias Warnings")
                for w in ethics.bias_warnings:
                    st.warning(w)

            st.subheader("Dataset Provenance")
            st.info(ethics.dataset_provenance)

            st.subheader("Audit Trail")
            st.code(f"Logged to: {ethics.audit_trail_path}")

        # ── Save reports ──────────────────────────────────────────────────────
        if save_report:
            with st.spinner("Saving reports..."):
                from report import generate_report
                xai_text = xai["summary"] if xai else "Heatmap disabled"
                heatmap_save = xai["heatmap_bgr"] if xai else image_bgr
                paths = generate_report(
                    image_path=uploaded.name,
                    annotated_image=annotated,
                    heatmap_image=heatmap_save,
                    detections=detections,
                    ethics_report=ethics,
                    explanation_text=xai_text,
                )
            st.success("✅ Reports saved to /reports folder")
            with st.expander("📁 Saved file paths"):
                for k, v in paths.items():
                    st.code(v)
