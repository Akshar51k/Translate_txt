import logging
import os
from typing import Optional, List, Tuple, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# Resolve base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging configuration to write to app.log in the project folder
log_file = os.path.join(BASE_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Silence verbose huggingface/transformers logs to keep app.log clean
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

from detector import LanguageDetector, LANGUAGE_MAP
from translator_engine import TranslationEngine


# Page configurations and SEO metadata
st.set_page_config(
    page_title="PolyglotTranslate - Multilingual text translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism, modern typography, animations, responsive columns)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Main font overrides */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Custom Header with linear gradient */
.header-container {
    text-align: center;
    padding: 30px 10px 10px 10px;
    margin-bottom: 25px;
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.main-title {
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 50%, #6BCB77 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.subtitle {
    font-size: 1.1rem;
    color: #a0aec0;
    font-weight: 300;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Glassmorphic Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.1);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #4D96FF;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.85rem;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Translation Cards Layout */
.translation-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 15px;
    margin-bottom: 30px;
}

.translation-row {
    background: rgba(255, 255, 255, 0.015);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}

.translation-row:hover {
    background: rgba(255, 255, 255, 0.035);
    border-color: rgba(77, 150, 255, 0.2);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.row-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 10px;
    font-size: 0.85rem;
}

.badges-container {
    display: flex;
    gap: 8px;
    align-items: center;
}

.lang-badge {
    background: linear-gradient(135deg, #4D96FF 0%, #00D2FC 100%);
    color: #ffffff;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    box-shadow: 0 2px 5px rgba(77, 150, 255, 0.2);
}

.eng-badge {
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
}

.conf-badge {
    color: #718096;
    font-weight: 400;
}

.paragraph-num {
    font-weight: 600;
    color: #4D96FF;
    font-family: 'Outfit', sans-serif;
}

.row-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

@media (max-width: 800px) {
    .row-body {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}

.text-box {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e2e8f0;
    white-space: pre-wrap;
}

.box-title {
    font-weight: 600;
    color: #718096;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}

.box-content {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid rgba(255, 255, 255, 0.02);
    min-height: 50px;
}

.box-content.unchanged {
    border-left: 3px solid #718096;
}

.box-content.translated {
    border-left: 3px solid #6BCB77;
    background: rgba(107, 203, 119, 0.03);
}

/* Status box formatting */
.status-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
}
</style>
"""

# Render styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADING -----------------

@st.cache_resource
def load_detector() -> LanguageDetector:
    """
    Caches the fastText language detector. If downloading is needed,
    uses a streamlit-safe container to show progress.
    """
    model_path = os.path.join(BASE_DIR, "models", "lid.176.ftz")
    if not os.path.exists(model_path):
        # Create UI elements for download progress
        with st.status("Initializing language detector model...", expanded=True) as status:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def progress_callback(pct: float):
                progress_bar.progress(pct)
                status_text.text(f"Downloading lid.176.ftz... {pct * 100:.1f}%")
                
            detector = LanguageDetector(model_path=model_path, progress_callback=progress_callback)
            status.update(label="Language detector ready!", state="complete", expanded=False)
            progress_bar.empty()
            status_text.empty()
            return detector
    else:
        return LanguageDetector(model_path=model_path)

@st.cache_resource
def load_translator(device: str) -> TranslationEngine:
    """
    Caches the CTranslate2 translation engine.
    On first run, downloads and converts the public NLLB-200 model to
    CTranslate2 INT8 format locally for low-latency inference.
    """
    with st.status("Initializing translation engine...", expanded=True) as status:
        status_text = st.empty()
        
        def status_callback(msg: str):
            status_text.text(msg)
            
        engine = TranslationEngine(
            device=device,
            progress_callback=status_callback
        )
        status.update(label="Translation engine ready!", state="complete", expanded=False)
        status_text.empty()
        return engine

# ----------------- MAIN UI -----------------

def main():
    # Header Banner
    st.markdown("""
        <div class="header-container">
            <div class="main-title">🌐 PolyglotTranslate</div>
            <div class="subtitle">
                High-performance multilingual text translation engine powered by fastText and CTranslate2 NLLB-200.
                Upload files, auto-detect paragraph languages, and translate everything to English instantly.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Automatically detect if GPU (CUDA) is available
    import ctranslate2
    try:
        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False
    device_option = "cuda" if cuda_available else "cpu"
    device_display = "GPU (CUDA)" if device_option == "cuda" else "CPU"
    
    # Sidebar Panel
    st.sidebar.markdown("### ⚙️ Engine Settings")
    st.sidebar.info(f"🖥️ **Active Device:** {device_display}")
    
    # Confidence threshold for fastText
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="If fastText detects a language with confidence below this threshold, the paragraph is treated as English and kept unchanged."
    )
    
    # Batch size adequate for hosting on Streamlit cloud
    batch_size = 4
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Architecture Details")
    st.sidebar.markdown(
        f"""
        - **Language Detector:** fastText `lid.176.ftz` (lightweight compressed model).
        - **Translation Model:** Meta NLLB-200 Distilled 600M, converted to CTranslate2 INT8 format for efficient CPU/GPU execution.
        - **Active Device:** {device_display}
        - **Batch Size:** {batch_size} (optimized for hosting)
        - **Pipeline:** Paragraph splitting -> parallel language detection -> batch translation of non-English segments -> structure-preserving export.
        """
    )
    
    # Initialize detector and translator models
    try:
        detector = load_detector()
        translator = load_translator(device=device_option)
    except Exception as e:
        st.error(f"Failed to load translation pipeline: {e}")
        st.info("Check your Python environment and try running again.")
        return

    # Main application space
    if getattr(translator, 'device_fallback', False):
        st.warning("⚠️ **CUDA Initialization Failed:** The translation engine was unable to initialize on GPU (CUDA). We have automatically fallen back to **CPU** mode.")

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Select a UTF-8 encoded text file (.txt)", 
        type=["txt"],
        help="The file will be read, split by double newlines into paragraphs, and translated."
    )
    
    if uploaded_file is not None:
        try:
            # Read file as UTF-8
            file_bytes = uploaded_file.read()
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            st.error("Error decoding file: Please ensure the file is encoded in UTF-8 format.")
            return
            
        if not content.strip():
            st.warning("The uploaded file is empty. Please upload a file containing text.")
            return
            
        # Normalize all types of line endings to standard LF (\n) to prevent paragraph splitting issues on Windows
        normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
        
        # Split text into paragraphs (separated by double newlines)
        raw_paragraphs = normalized_content.split("\n\n")
        
        # Filter paragraphs: store original and stripped text
        paragraphs: List[str] = []
        for p in raw_paragraphs:
            # We preserve spacing/empty paragraphs but keep track of non-empty text
            paragraphs.append(p)
            
        st.info(f"Loaded file successfully. Found **{len(paragraphs)}** paragraphs.")
        
        # Trigger Translation Button
        if st.button("🚀 Start Translation Process", type="primary", key="btn_translate"):
            logger.info(f"Start Translation Process triggered for file: {uploaded_file.name}")
            
            # Step 1: Run Language Detection
            detection_results: List[Tuple[str, float, Optional[str], str]] = []
            
            # Fast text is extremely fast, so we can do it in a small spinner
            with st.spinner("Analyzing paragraph languages..."):
                for p in paragraphs:
                    stripped_p = p.strip()
                    if not stripped_p:
                        # Empty paragraph
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
                
                # We translate if:
                # - The language is NOT English
                # - Confidence is above threshold
                # - The language has a mapped NLLB code
                if iso_code != "en" and confidence >= confidence_threshold and nllb_code is not None:
                    paragraphs_to_translate.append(stripped_p)
                    indices_to_translate.append(idx)
                    src_langs_to_translate.append(nllb_code)
            
            # Show Metrics
            total_paragraphs = len(paragraphs)
            to_translate_count = len(paragraphs_to_translate)
            english_count = sum(1 for iso, conf, _, _ in detection_results if iso == "en" and p.strip())
            low_conf_or_unsupported = total_paragraphs - to_translate_count - english_count
            
            logger.info(
                f"Language detection metrics - Total: {total_paragraphs}, "
                f"To Translate: {to_translate_count}, English (Unchanged): {english_count}, "
                f"Low Confidence/Unsupported: {low_conf_or_unsupported}"
            )
            
            # Render visual metrics
            st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{total_paragraphs}</div>
                        <div class="metric-label">Total Paragraphs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #6BCB77;">{to_translate_count}</div>
                        <div class="metric-label">To Translate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #FFD93D;">{english_count}</div>
                        <div class="metric-label">English (Unchanged)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #FF6B6B;">{low_conf_or_unsupported}</div>
                        <div class="metric-label">Low Confidence / Other</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Step 3: Run Batch Translation
            translated_paragraphs_map: Dict[int, str] = {}
            
            if to_translate_count > 0:
                import time
                start_time = time.time()
                logger.info(f"Starting batch translation of {to_translate_count} paragraphs with batch_size={batch_size}...")
                
                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                
                # Perform batch translation with incremental updates
                for batch_idx in range(0, to_translate_count, batch_size):
                    batch_texts = paragraphs_to_translate[batch_idx:batch_idx + batch_size]
                    batch_langs = src_langs_to_translate[batch_idx:batch_idx + batch_size]
                    batch_indices = indices_to_translate[batch_idx:batch_idx + batch_size]
                    
                    pct = batch_idx / to_translate_count
                    progress_bar.progress(pct)
                    progress_text.markdown(f"**Translating paragraphs {batch_idx + 1} to {min(batch_idx + batch_size, to_translate_count)} of {to_translate_count}...**")
                    
                    # Translate this small batch
                    batch_results = translator.translate_batch(
                        texts=batch_texts,
                        src_langs=batch_langs,
                        batch_size=batch_size
                    )
                    
                    # Store results mapped to original indices
                    for idx, result in zip(batch_indices, batch_results):
                        translated_paragraphs_map[idx] = result
                
                elapsed_time = time.time() - start_time
                logger.info(f"Batch translation completed in {elapsed_time:.2f} seconds.")
                progress_bar.progress(1.0)
                progress_text.success(f"🎉 Translation completed successfully in {elapsed_time:.2f} seconds!")
                
            # Step 4: Reassemble final document
            final_paragraphs: List[str] = []
            comparison_data: List[Dict[str, Any]] = []
            
            for idx, original_p in enumerate(paragraphs):
                iso_code, confidence, nllb_code, lang_name = detection_results[idx]
                
                # If paragraph was translated, fetch the translation
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
                    # Unchanged (English, low confidence, or empty)
                    final_paragraphs.append(original_p)
                    action_label = "Skipped" if original_p.strip() else "Empty"
                    
                    # Determine reason for skipping
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
            
            # Combine paragraphs with double newlines
            output_content = "\n\n".join(final_paragraphs)
            
            # Step 5: Render Download Panel
            st.markdown("### 📥 Download Results")
            st.download_button(
                label=f"💾 Download {uploaded_file.name}",
                data=output_content,
                file_name=uploaded_file.name,
                mime="text/plain",
                type="primary",
                key="btn_download"
            )
            
            # Step 6: Render Comparison List
            st.markdown("### 🔍 Paragraph-by-Paragraph Comparison")
            
            # HTML generation for side-by-side comparison
            html_rows = []
            for row in comparison_data:
                # Format empty or skipped vs translated
                if row["action"] == "Empty":
                    # Skip rendering details for empty paragraphs, just list it briefly
                    continue
                    
                confidence_pct = f"{row['confidence'] * 100:.1f}%"
                
                if row["action"] == "Translated":
                    badge_html = f'<span class="lang-badge">{row["lang"]} ({row["code"]})</span>'
                    meta_html = f'<span class="conf-badge">Confidence: {confidence_pct}</span>'
                    original_class = "box-content"
                    translated_class = "box-content translated"
                    translated_content = row["translated"]
                else:
                    badge_html = f'<span class="eng-badge">{row["lang"]}</span>'
                    meta_html = f'<span class="conf-badge">Action: Kept Unchanged ({row["reason"]})</span>'
                    original_class = "box-content unchanged"
                    translated_class = "box-content unchanged"
                    translated_content = "(Unchanged)"
                
                html_rows.append(f"""
                    <div class="translation-row">
                        <div class="row-header">
                            <div>
                                <span class="paragraph-num">Paragraph #{row["num"]}</span>
                                &nbsp;&nbsp;|&nbsp;&nbsp;
                                {badge_html}
                            </div>
                            <div>
                                {meta_html}
                            </div>
                        </div>
                        <div class="row-body">
                            <div class="text-box">
                                <div class="box-title">Original Text</div>
                                <div class="{original_class}">{row["original"]}</div>
                            </div>
                            <div class="text-box">
                                <div class="box-title">English Output</div>
                                <div class="{translated_class}">{translated_content}</div>
                            </div>
                        </div>
                    </div>
                """)
                
            # Render comparison lists
            if html_rows:
                # Remove newlines and carriage returns so Streamlit's markdown parser does not treat indented HTML as code blocks
                clean_html = "".join(html_rows).replace("\n", "").replace("\r", "")
                st.markdown(f'<div class="translation-list">{clean_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No content paragraphs to compare.")
                
if __name__ == "__main__":
    main()
