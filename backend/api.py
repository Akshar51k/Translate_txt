import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Resolve base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging configuration to write to backend.log in the backend folder
log_file = os.path.join(BASE_DIR, "backend.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Silence verbose huggingface/transformers logs to keep log clean
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

from detector import LanguageDetector
from translator_engine import TranslationEngine

# Models will be loaded inside lifespan
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    logger.info("Initializing models on startup...")
    try:
        # Automatically detect if GPU (CUDA) is available
        import ctranslate2
        try:
            cuda_available = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda_available = False
        device_option = "cuda" if cuda_available else "cpu"
        device_display = "GPU (CUDA)" if device_option == "cuda" else "CPU"
        logger.info(f"Detected device: {device_display}")

        models["detector"] = LanguageDetector()
        models["translator"] = TranslationEngine(device=device_option)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load models during startup: {e}", exc_info=True)
        raise e
    yield
    # Clean up
    logger.info("Shutting down api...")
    models.clear()

app = FastAPI(
    title="PolyglotTranslate API",
    description="API backend for high-performance multilingual text translation",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    paragraphs: List[str] = Field(..., description="List of paragraphs to translate")
    confidence_threshold: float = Field(0.60, ge=0.0, le=1.0, description="fastText confidence threshold")
    batch_size: int = Field(4, ge=1, le=64, description="Translation batch size")

class ParagraphMetrics(BaseModel):
    total: int
    to_translate: int
    english_unchanged: int
    low_confidence_or_other: int

class ParagraphComparison(BaseModel):
    num: int
    original: str
    translated: str
    lang: str
    code: Optional[str] = None
    confidence: float
    action: str
    reason: Optional[str] = None

class TranslationResponse(BaseModel):
    translated_text: str
    metrics: ParagraphMetrics
    comparison_data: List[ParagraphComparison]
    device_used: str
    elapsed_seconds: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "detector_loaded": "detector" in models,
        "translator_loaded": "translator" in models,
        "device_used": models["translator"].device_used if "translator" in models else None,
        "device_fallback": models["translator"].device_fallback if "translator" in models else False
    }

@app.post("/translate", response_model=TranslationResponse)
def translate_text(req: TranslationRequest):
    if "detector" not in models or "translator" not in models:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    detector = models["detector"]
    translator = models["translator"]

    start_time = time.time()
    paragraphs = req.paragraphs
    confidence_threshold = req.confidence_threshold
    batch_size = req.batch_size

    logger.info(f"Received request to translate {len(paragraphs)} paragraphs with batch_size={batch_size}")

    # Step 1: Run Language Detection
    detection_results: List[Tuple[str, float, Optional[str], str]] = []
    for p in paragraphs:
        stripped_p = p.strip()
        if not stripped_p:
            detection_results.append(("en", 1.0, None, "English"))
        else:
            # Clean markdown headers/decorations for better language detection accuracy
            clean_detect_text = stripped_p
            clean_detect_text = clean_detect_text.lstrip("#").strip()
            clean_detect_text = clean_detect_text.strip("*_-`")
            
            iso_code, confidence = detector.detect(clean_detect_text)
            nllb_code = detector.get_nllb_code(iso_code)
            lang_name = detector.get_language_name(iso_code)
            detection_results.append((iso_code, confidence, nllb_code, lang_name))

    # Step 2: Identify segments that require translation
    paragraphs_to_translate: List[str] = []
    indices_to_translate: List[int] = []
    src_langs_to_translate: List[str] = []

    for idx, p in enumerate(paragraphs):
        stripped_p = p.strip()
        if not stripped_p:
            continue
            
        iso_code, confidence, nllb_code, _ = detection_results[idx]
        
        if iso_code != "en" and confidence >= confidence_threshold and nllb_code is not None:
            paragraphs_to_translate.append(stripped_p)
            indices_to_translate.append(idx)
            src_langs_to_translate.append(nllb_code)

    # Calculate metrics
    total_paragraphs = len(paragraphs)
    to_translate_count = len(paragraphs_to_translate)
    english_count = sum(1 for idx, (iso, conf, _, _) in enumerate(detection_results) if iso == "en" and paragraphs[idx].strip())
    low_conf_or_unsupported = total_paragraphs - to_translate_count - english_count

    # Step 3: Run Batch Translation
    translated_paragraphs_map: Dict[int, str] = {}
    if to_translate_count > 0:
        logger.info(f"Starting batch translation of {to_translate_count} paragraphs...")
        batch_results = translator.translate_batch(
            texts=paragraphs_to_translate,
            src_langs=src_langs_to_translate,
            batch_size=batch_size
        )
        # Store results mapped to original indices
        for idx, result in zip(indices_to_translate, batch_results):
            translated_paragraphs_map[idx] = result

    # Step 4: Reassemble final document & comparison data
    final_paragraphs: List[str] = []
    comparison_data = []

    for idx, original_p in enumerate(paragraphs):
        iso_code, confidence, nllb_code, lang_name = detection_results[idx]
        
        if idx in translated_paragraphs_map:
            translated_text = translated_paragraphs_map[idx]
            final_paragraphs.append(translated_text)
            comparison_data.append({
                "num": idx + 1,
                "original": original_p,
                "translated": translated_text,
                "lang": lang_name,
                "code": nllb_code,
                "confidence": confidence,
                "action": "Translated"
            })
        else:
            final_paragraphs.append(original_p)
            action_label = "Skipped" if original_p.strip() else "Empty"
            
            if not original_p.strip():
                reason = "Empty Paragraph"
            elif iso_code == "en":
                reason = "English"
            elif confidence < confidence_threshold:
                reason = f"Low Confidence ({lang_name} @ {confidence * 100:.0f}%)"
            else:
                reason = "Unsupported Language"
                
            comparison_data.append({
                "num": idx + 1,
                "original": original_p,
                "translated": original_p,
                "lang": lang_name if original_p.strip() else "N/A",
                "code": "eng_Latn" if iso_code == "en" else None,
                "confidence": confidence if original_p.strip() else 0.0,
                "action": action_label,
                "reason": reason
            })

    output_content = "\n\n".join(final_paragraphs)
    elapsed_time = time.time() - start_time
    logger.info(f"Request completed in {elapsed_time:.2f} seconds.")

    return {
        "translated_text": output_content,
        "metrics": {
            "total": total_paragraphs,
            "to_translate": to_translate_count,
            "english_unchanged": english_count,
            "low_confidence_or_other": low_conf_or_unsupported
        },
        "comparison_data": comparison_data,
        "device_used": translator.device_used,
        "elapsed_seconds": round(elapsed_time, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
