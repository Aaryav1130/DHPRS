"""
FastAPI application for Doctor's Handwritten Prescription Recognition System.
Provides automatic OpenAPI / Swagger UI documentation at /docs.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'api' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import (
    DEFAULT_MODEL_NAME,
    DEVICE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.pipeline.orchestrator import PrescriptionPipeline
from src.postprocessing.drug_matcher import DrugMatcher
from api.schemas import (
    PrescriptionRecognitionResponse,
    DrugMatchRequest,
    DrugMatchResponse,
    SystemHealthResponse,
)

app = FastAPI(
    title="DHPRS: Doctor's Handwritten Prescription Recognition API",
    description="""
    Production-grade REST API for digitizing, recognizing, and validating handwritten medical prescriptions.
    
    ### Pipeline Architecture:
    1. **Preprocessing**: Shadow removal, illumination normalization, and automated deskewing.
    2. **Segmentation**: Morphological text line & word bounding box isolation.
    3. **Recognition**: Transformer-based Handwritten Text Recognition (Microsoft TrOCR).
    4. **Post-Processing**: RapidFuzz drug lexicon matching + NLM RxNorm resolution + Clinical Regex entity parsing.
    5. **Safety Evaluation**: Automated Human-in-the-Loop (HITL) confidence scoring and validation.
    """,
    version="1.0.0",
)

# Enable CORS for external frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount processed artifacts directory for image inspection
app.mount("/artifacts", StaticFiles(directory=str(PROCESSED_DATA_DIR)), name="artifacts")

# Initialize shared singleton instances
pipeline = PrescriptionPipeline()
drug_matcher = DrugMatcher()


@app.get("/", tags=["Health"])
def root():
    """Root endpoint welcoming user and linking to docs."""
    return {
        "project": "Doctor's Handwritten Prescription Recognition System (DHPRS)",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=SystemHealthResponse, tags=["Health"])
def health_check():
    """Returns runtime status, hardware device, and lexicon metrics."""
    return SystemHealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(DEVICE),
        trocr_model=DEFAULT_MODEL_NAME,
        total_lexicon_drugs=len(drug_matcher.drug_list),
    )


@app.post(
    "/api/v1/prescription/recognize",
    response_model=PrescriptionRecognitionResponse,
    tags=["Prescription Recognition"],
)
async def recognize_prescription(
    file: UploadFile = File(..., description="Uploaded prescription photo or scan"),
):
    """
    Uploads a handwritten prescription image, processes it through the 5-stage pipeline,
    and returns standardized structured clinical JSON.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, etc.)")

    file_ext = Path(file.filename).suffix or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:8]}{file_ext}"
    saved_raw_path = RAW_DATA_DIR / unique_name

    try:
        with open(saved_raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")

    try:
        output = pipeline.process(str(saved_raw_path), save_artifacts=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

    annotated_url = None
    if output.annotated_image_path:
        annotated_filename = Path(output.annotated_image_path).name
        annotated_url = f"/artifacts/{annotated_filename}"

    dict_res = output.to_dict()
    dict_res["annotated_image_url"] = annotated_url

    return dict_res


@app.post(
    "/api/v1/drug/match",
    response_model=DrugMatchResponse,
    tags=["Pharmaceutical NLP"],
)
def match_drug_candidate(payload: DrugMatchRequest):
    """
    Fuzzy matches an isolated OCR drug candidate against WHO, FDA, and South Asian drug lexicons.
    """
    result = drug_matcher.match(payload.term, threshold=payload.threshold)
    return DrugMatchResponse(
        query=result.original_query,
        matched_name=result.matched_name,
        similarity_score=result.similarity_score,
        source=result.source,
        is_confident=result.is_confident,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
