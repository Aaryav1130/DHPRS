"""
Streamlit Web Dashboard for Doctor's Handwritten Prescription Recognition System (DHPRS).
Features live image upload, side-by-side inspection, structured clinical table, HITL verification, and JSON export.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'api' can be imported by Streamlit runner
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.config import (
    DEFAULT_MODEL_NAME,
    DEVICE,
    TEST_SAMPLES_DIR,
    PROCESSED_DATA_DIR,
)
from src.pipeline.orchestrator import PrescriptionPipeline
from src.postprocessing.drug_matcher import DrugMatcher

# Streamlit Page Configuration
st.set_page_config(
    page_title="DHPRS - Doctor's Handwritten Prescription Recognition",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Healthcare Theme CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .badge-verified {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-caution {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-review {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the singleton pipeline instance."""
    return PrescriptionPipeline()


@st.cache_resource
def get_drug_matcher():
    """Initializes and caches the drug matcher."""
    return DrugMatcher()


pipeline = get_pipeline()
drug_matcher = get_drug_matcher()

# Sidebar Navigation & System Stats
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-history.png", width=64)
    st.title("DHPRS Console")
    st.markdown("**Doctor's Handwritten Prescription Recognition System**")
    st.markdown("---")

    st.subheader("System Telemetry")
    st.write(f"**Hardware Device:** `{DEVICE}`")
    st.write(f"**TrOCR Model:** `{DEFAULT_MODEL_NAME.split('/')[-1]}`")
    st.write(f"**Loaded Drugs:** `{len(drug_matcher.drug_list):,}`")

    st.markdown("---")
    st.markdown("### Navigation")
    app_mode = st.radio(
        "Choose Mode:",
        ["Prescription Scanner", "Drug Lexicon Tester", "Architecture & Papers"],
    )

# Mode 1: Prescription Scanner
if app_mode == "Prescription Scanner":
    st.markdown('<div class="main-header">🏥 Handwritten Prescription Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a physical prescription photo or scan to extract and validate clinical entities with Transformer OCR.</div>',
        unsafe_allow_html=True,
    )

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Prescription Image (JPG, PNG, JPEG):",
            type=["jpg", "jpeg", "png"],
        )

    with col_sample:
        st.markdown("**Or Pick a Demo Prescription:**")
        sample_files = list(TEST_SAMPLES_DIR.glob("*.*"))
        sample_options = ["None"] + [f.name for f in sample_files]
        chosen_sample = st.selectbox("Preloaded Test Samples:", sample_options)

    # Resolve active image
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
    elif chosen_sample != "None":
        sample_path = TEST_SAMPLES_DIR / chosen_sample
        active_image = Image.open(sample_path).convert("RGB")

    if active_image is not None:
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.subheader("1. Original Prescription Scan")
            st.image(active_image, use_container_width=True)

        with col_img2:
            st.subheader("2. Pipeline Analysis")
            process_btn = st.button("🚀 Process Prescription with TrOCR", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing through 5-stage pipeline (Deskew -> Segmentation -> TrOCR -> NLP Matching)..."):
                # Run pipeline
                output = pipeline.process(active_image, save_artifacts=True)

                with col_img2:
                    if output.annotated_image_path and Path(output.annotated_image_path).exists():
                        annotated_img = Image.open(output.annotated_image_path)
                        st.image(annotated_img, caption="Detected Lines & OCR Overlays", use_container_width=True)
                    else:
                        st.info("Processed without bounding box overlay.")

            # Metrics Ribbon
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Total Medicines Found", output.total_medications_detected)
            with m2:
                conf_pct = f"{int(output.overall_confidence * 100)}%"
                st.metric("Overall Confidence", conf_pct)
            with m3:
                st.metric("Skew Correction", f"{output.deskew_angle_corrected:+.1f}°")
            with m4:
                status_color = "🟢" if output.safety_status == "VERIFIED" else "🟡" if output.safety_status == "CAUTION" else "🔴"
                st.metric("Safety Assessment", f"{status_color} {output.safety_status}")

            if output.needs_manual_review:
                st.warning("⚠️ **Human-in-the-Loop Review Recommended:** Certain low-confidence handwriting or ambiguous drug names require pharmacist confirmation.")

            # Clinical Findings Ribbon
            if hasattr(output, "clinical_notes") and output.clinical_notes:
                st.markdown("### 🩺 Clinical Findings & Patient Observations")
                for note in output.clinical_notes:
                    st.info(f"📌 {note}")

            # Structured Prescription Table
            st.markdown("### 📋 Digital Prescription Output")

            if output.medications:
                table_data = []
                for med in output.medications:
                    table_data.append({
                        "Line": med.line_index,
                        "Raw OCR Transcription": med.raw_ocr_text,
                        "Validated Drug Name": med.verified_drug_name,
                        "Dosage Strength": med.dosage_strength or "—",
                        "Dosage Form": med.dosage_form or "—",
                        "Frequency": med.frequency or "—",
                        "Duration": med.duration or "—",
                        "Instructions": med.instructions or "—",
                        "Confidence": f"{int(med.final_confidence * 100)}%",
                        "Status": "⚠️ Review Required" if med.needs_review else "✅ Verified",
                    })

                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Export Options
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    json_str = json.dumps(output.to_dict(), indent=2, default=str)
                    st.download_button(
                        label="📥 Download Structured JSON",
                        data=json_str,
                        file_name=f"{output.prescription_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_exp2:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Clinical CSV",
                        data=csv_data,
                        file_name=f"{output.prescription_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.info("No legible text lines detected. Please ensure the prescription is well-lit and in focus.")

# Mode 2: Drug Lexicon Tester
elif app_mode == "Drug Lexicon Tester":
    st.markdown('<div class="main-header">💊 Pharmaceutical Lexicon & Fuzzy Matcher</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Test the rapid fuzzy matching and phonetic disambiguation engine on messy handwriting OCR typos.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        test_query = st.text_input("Enter noisy drug candidate string (e.g. 'Amoxcilin', 'Paracitamol', 'Metformn'):", value="Amoxcilin 500mg")
        threshold_val = st.slider("Acceptance Threshold (%):", min_value=50.0, max_value=95.0, value=75.0, step=1.0)

        match_btn = st.button("🔍 Match Drug", type="primary")

    if match_btn or test_query:
        res = drug_matcher.match(test_query, threshold=threshold_val, use_online_fallback=True)

        st.markdown("### Match Results")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Input Query", res.original_query)
        with r2:
            st.metric("Matched Drug", res.matched_name)
        with r3:
            st.metric("Similarity Score", f"{res.similarity_score:.1f}%")
        with r4:
            st.metric("Source", res.source.replace("_", " ").title())

        if res.is_confident:
            st.success(f"✅ High-confidence match verified: **{res.matched_name}** ({res.similarity_score:.1f}% match via {res.source})")
        else:
            st.warning(f"⚠️ Low confidence: Matched candidate is **{res.matched_name}** with {res.similarity_score:.1f}% similarity. Pharmacist review required.")

# Mode 3: Architecture & Papers
elif app_mode == "Architecture & Papers":
    st.markdown('<div class="main-header">📚 Technical Architecture & Research References</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Engineering design, scientific citations, and evaluators reference guide.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### 5-Stage System Architecture
    ```
    [Raw Photo / Scan]
           │
           ▼
    1. Preprocessing (OpenCV)
       • Morphological Background Division (Shadow Removal)
       • CLAHE (Contrast Limited Adaptive Histogram Equalization)
       • Deskewing via Minimum Area Bounding Rectangles
           │
           ▼
    2. Text Segmentation (Morphological Contours)
       • Horizontal Dilation Kernel (Height=3, Width=25)
       • Natural Human Reading Order Spatial Sorter
       • Aspect-Ratio Preserving Letterboxing Canvas (384x384)
           │
           ▼
    3. Handwritten Text Recognition (Microsoft TrOCR)
       • Vision Transformer (BEiT) Encoder
       • Autoregressive Causal Decoder (RoBERTa)
       • Cross-Attention Visual Grounding
           │
           ▼
    4. Post-Processing & Verification (RapidFuzz + NLM RxNorm)
       • Clinical Regex Entity Parser (Dosage, Form, Frequency, Duration)
       • Fuzzy Drug Dictionary Matcher (WHO EML + Indian/BD Lexicons)
       • Medical Abbreviation Expansion (OD, BD, TDS, 1+0+1, AC, PC)
           │
           ▼
    5. Clinical Structured Output & Safety Verification
       • Per-line & Global Confidence Scoring
       • Human-in-the-Loop (HITL) Safety Flagging
       • Structured JSON / CSV / PDF Export
    ```

    ### Key Scientific Citations (2024–2026):
    1. **MIRAGE (2024-2025):** *Multimodal Identification and Recognition of Annotations in Indian General Prescriptions* — 743K images, 82% accuracy. [arXiv:2410.09729](https://arxiv.org/abs/2410.09729)
    2. **Multi-Head TrOCR (2024):** *Accurate Extraction of Medicine from Handwritten Prescriptions* — Achieved Character Error Rate (CER) of 1.4%. [arXiv:2412.18199](https://arxiv.org/abs/2412.18199)
    3. **E-TrOCR (ICDAR 2025):** *Extended-TrOCR: Enhancing Transformer-Based OCR for Complex Handwriting*. [arXiv:2406.19234](https://arxiv.org/abs/2406.19234)
    4. **TrOCR (Microsoft Research):** *Transformer-based Optical Character Recognition with Pre-trained Models*. [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)
    """)
