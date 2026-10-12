# 🏥 DHPRS: Doctor's Handwritten Prescription Recognition System

An accurate, industry-standard, 100% free and open-source system for digitizing, recognizing, and validating handwritten medical prescriptions using **Transformer-based Optical Character Recognition (Microsoft TrOCR)**, **Adaptive Preprocessing**, **Computer Vision Segmentation**, and **Clinical Entity NLP**.

---

## 🌟 Key Features

1. **5-Stage Medical Document Pipeline**:
   - **Stage 1 (Preprocessing)**: Non-uniform illumination normalization via morphological background division, contrast enhancement via CLAHE on faint ink strokes, and automated deskewing via minimum area bounding rectangles.
   - **Stage 2 (Segmentation)**: Fast contour-based line and word bounding box extraction with natural reading-order spatial sorting and aspect-preserving letterboxing canvas (384×384 px).
   - **Stage 3 (Recognition)**: Vision Transformer (ViT/BEiT) + Autoregressive Language Transformer (RoBERTa) via **Microsoft TrOCR**.
   - **Stage 4 (Medical NLP & Post-Processing)**:
     - Regex entity parsing: extracts dosage strength (e.g. `500mg`, `10ml`), dosage form (`Tab`, `Cap`, `Syp`), frequency (`TDS`, `BD`, `OD`), duration (`5 days`, `2 weeks`), and instructions (`before food`, `after meals`).
     - South Asian clinical frequency parsing: supports `1+0+1`, `1+1+1`, `0+0+1` notation.
     - C++ accelerated fuzzy string matching via **RapidFuzz** against offline WHO, Indian Pharmacopoeia, and Bangladesh pharmaceutical lexicons.
     - Online fallback to free **NLM RxNav / RxNorm REST API** (`rxnav.nlm.nih.gov`).
   - **Stage 5 (Safety Verification & Structured Output)**: Per-item and global confidence scoring, automated Human-in-the-Loop (HITL) review flagging for low-confidence handwriting, and structured clinical JSON export.

2. **Dual User Interfaces**:
   - **Interactive Web Dashboard (Streamlit)**: Live prescription upload, side-by-side visual inspection, bounding box overlay with confidence badges, interactive drug lexicon tester, and one-click JSON/CSV download.
   - **High-Performance REST API (FastAPI)**: Automatic OpenAPI / Swagger UI documentation at `http://127.0.0.1:8000/docs`.

3. **100% Free & Open-Source Stack**:
   - Zero paid APIs or closed models.
   - Ready-to-run **Google Colab Notebook** (`notebooks/03_colab_trocr_finetuning.ipynb`) for fine-tuning TrOCR on free T4 GPUs.
   - Built-in integration with the **RxHandBD** prescription dataset from Zenodo (Record 18478741).

---

## 📁 Project Directory Structure

```
D:\DHPRS\
├── .venv/                      # Python 3.12 virtual environment
├── data/
│   ├── raw/                    # Uploaded raw prescription images
│   ├── processed/              # Preprocessed & annotated overlay images
│   ├── crops/                  # Segmented text line / word crops
│   ├── datasets/               # Downloaded datasets (RxHandBD, AMR, etc.)
│   ├── dictionaries/           # Offline pharmaceutical lexicons (WHO, India, BD)
│   └── test_samples/           # Benchmark prescriptions for demos and tests
├── models/
│   ├── checkpoints/            # Fine-tuned model weights
│   └── cache/                  # Hugging Face local cache
├── src/
│   ├── config.py               # Central paths, device detection, hyperparameters
│   ├── preprocessing/          # Stage 1: Illumination, CLAHE, Deskew, Sauvola
│   ├── segmentation/           # Stage 2: Contour detection, Reading-order sorter
│   ├── recognition/            # Stage 3: TrOCR inference, Aspect padding, Dataset
│   ├── postprocessing/         # Stage 4: Abbreviations, Regex entities, RapidFuzz
│   ├── pipeline/               # Stage 5: End-to-end orchestrator
│   └── utils/                  # Visualizer (overlays, side-by-side)
├── api/
│   ├── app.py                  # FastAPI application with Swagger UI
│   └── schemas.py              # Pydantic v2 schemas
├── ui/
│   └── app.py                  # Streamlit healthcare dashboard
├── notebooks/
│   └── 03_colab_trocr_finetuning.ipynb  # Free Colab T4 GPU fine-tuning notebook
├── tests/                      # Automated unit tests for all stages
├── scripts/
│   ├── download_datasets.py    # Zenodo downloader & sample generator
│   ├── run_api.bat             # 1-click FastAPI launch script
│   └── run_ui.bat              # 1-click Streamlit launch script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Windows 10/11
- Python 3.12 64-bit

### 2. Activate Virtual Environment
```powershell
cd D:\DHPRS
.\.venv\Scripts\Activate.ps1
```

### 3. Launch the Healthcare Web Dashboard
Double-click `scripts\run_ui.bat` or run:
```powershell
streamlit run ui/app.py
```
The browser will automatically open at **`http://localhost:8501`**.

### 4. Launch the Backend API & Interactive Swagger UI
Double-click `scripts\run_api.bat` or run:
```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000/docs`** to test endpoints live.

---

## 🧪 Running Automated Tests
Run pytest across all pipeline stages:
```powershell
pytest tests/ -v
```

---

## 🔬 Training on Google Colab (Free T4 GPU)

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Upload the notebook from `D:\DHPRS\notebooks\03_colab_trocr_finetuning.ipynb`.
3. Set Runtime to **T4 GPU** (Runtime -> Change runtime type -> T4 GPU).
4. Run all cells:
   - Mounts Google Drive
   - Automatically downloads the **RxHandBD** Zenodo dataset (Record 18478741)
   - Fine-tunes `microsoft/trocr-base-stage1` using `Seq2SeqTrainer`
   - Evaluates Character Error Rate (CER)
   - Saves the final model checkpoint to Google Drive (`/content/drive/MyDrive/DHPRS_Models/trocr-rxhandbd-best`)
5. Download the saved folder and place it in `D:\DHPRS\models\checkpoints\trocr-rxhandbd-best`.

---

## 📚 Key Research References & Citations

1. **TrOCR (Microsoft Research):**
   *Li et al.*, "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models", [arXiv:2109.10282](https://arxiv.org/abs/2109.10282).
2. **MIRAGE (2024–2025):**
   *Bhowmik et al.*, "MIRAGE: Multimodal Identification and Recognition of Annotations in Indian General Prescriptions", [arXiv:2410.09729](https://arxiv.org/abs/2410.09729).
3. **Multi-Head TrOCR (2024):**
   *Rehman et al.*, "Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions", [arXiv:2412.18199](https://arxiv.org/abs/2412.18199).
4. **E-TrOCR (ICDAR 2025):**
   *Chen et al.*, "Extended-TrOCR: Enhancing Transformer-Based OCR for Complex Handwriting", [arXiv:2406.19234](https://arxiv.org/abs/2406.19234).
5. **RxHandBD Dataset:**
   *Md. Masudul Islam*, "RxHandBD: A Handwritten Prescription Word Image Dataset", [Zenodo: 18478741](https://doi.org/10.5281/zenodo.18478741).

---

## 📄 License
MIT License — 100% Free for academic and personal use.
