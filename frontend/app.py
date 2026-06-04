import logging
import os
import requests
from typing import Optional, List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# Resolve base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging configuration
log_file = os.path.join(BASE_DIR, "frontend.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

def main():
    # Header Banner
    st.markdown("""
        <div class="header-container">
            <div class="main-title">🌐 PolyglotTranslate</div>
            <div class="subtitle">
                High-performance decoupled translation interface. Auto-detect paragraph languages and translate everything to English.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Panel
    st.sidebar.markdown("### ⚙️ Connection Settings")
    
    # Backend URL configuration
    backend_url_input = st.sidebar.text_input(
        "Backend API URL",
        value="http://127.0.0.1:8000",
        help="The endpoint where the translation backend (FastAPI) is hosted."
    )
    
    # Automatically rewrite 0.0.0.0 to 127.0.0.1 to handle client routing on Windows
    backend_url = backend_url_input.replace("0.0.0.0", "127.0.0.1")
    
    # Backend health check
    backend_healthy = False
    device_display = "Disconnected"
    try:
        health_resp = requests.get(f"{backend_url.rstrip('/')}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            backend_healthy = health_data.get("status") == "healthy"
            device_used = health_data.get("device_used", "cpu")
            device_display = "GPU (CUDA)" if device_used == "cuda" else "CPU"
            if health_data.get("device_fallback", False):
                device_display += " (Fallback)"
    except Exception:
        pass
        
    if backend_healthy:
        st.sidebar.success(f"🟢 Connected to Backend ({device_display})")
    else:
        st.sidebar.error("🔴 Disconnected from Backend")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="If fastText detects a language with confidence below this threshold, the paragraph is treated as English and kept unchanged."
    )
    
    # Hardcoded Batch size optimized for parallel translation throughput
    batch_size = 8
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Architecture Details")
    st.sidebar.markdown(
        f"""
        - **Language Detector:** fastText `lid.176.ftz` (running on API).
        - **Translation Model:** Meta NLLB-200 Distilled 600M (running on API).
        - **Decoupled Architecture:** Lightweight client UI suitable for Streamlit Cloud.
        """
    )

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Select a UTF-8 encoded text file (.txt)", 
        type=["txt"],
        help="The file will be read, split by double newlines into paragraphs, and sent to the API backend for translation."
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
        paragraphs = [p for p in normalized_content.split("\n\n")]
        
        st.info(f"Loaded file successfully. Found **{len(paragraphs)}** paragraphs.")
        
        # Disable translation button if backend is not reachable
        btn_disabled = not backend_healthy
        
        if btn_disabled:
            st.warning("⚠️ **Backend unreachable:** Please launch the translation API backend or verify the URL settings in the sidebar.")
            
        if st.button("🚀 Start Translation Process", type="primary", key="btn_translate", disabled=btn_disabled):
            logger.info(f"Start Translation Process triggered for file: {uploaded_file.name} to {backend_url}")
            
            with st.spinner("Sending document to translation API... Translation may take a moment depending on file size."):
                try:
                    payload = {
                        "paragraphs": paragraphs,
                        "confidence_threshold": confidence_threshold,
                        "batch_size": batch_size
                    }
                    
                    response = requests.post(
                        f"{backend_url.rstrip('/')}/translate",
                        json=payload,
                        timeout=300  # 5 minute timeout for very large documents
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        translated_text = res_data["translated_text"]
                        metrics = res_data["metrics"]
                        comparison_data = res_data["comparison_data"]
                        device_used = res_data["device_used"]
                        elapsed_seconds = res_data["elapsed_seconds"]
                        
                        st.success(f"🎉 Translation completed successfully on {device_display if backend_healthy else device_used} in {elapsed_seconds:.2f} seconds!")
                        
                        # Render visual metrics
                        st.markdown(f"""
                            <div class="metrics-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{metrics['total']}</div>
                                    <div class="metric-label">Total Paragraphs</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #6BCB77;">{metrics['to_translate']}</div>
                                    <div class="metric-label">To Translate</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #FFD93D;">{metrics['english_unchanged']}</div>
                                    <div class="metric-label">English (Unchanged)</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #FF6B6B;">{metrics['low_confidence_or_other']}</div>
                                    <div class="metric-label">Low Confidence / Other</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Render Download Panel
                        st.markdown("### 📥 Download Results")
                        st.download_button(
                            label=f"💾 Download {uploaded_file.name}",
                            data=translated_text,
                            file_name=uploaded_file.name,
                            mime="text/plain",
                            type="primary",
                            key="btn_download"
                        )
                        
                        # Render Comparison List
                        st.markdown("### 🔍 Paragraph-by-Paragraph Comparison")
                        
                        # HTML generation for side-by-side comparison
                        html_rows = []
                        for row in comparison_data:
                            if row["action"] == "Empty":
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
                            
                        # Render comparison list
                        if html_rows:
                            clean_html = "".join(html_rows).replace("\n", "").replace("\r", "")
                            st.markdown(f'<div class="translation-list">{clean_html}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No content paragraphs to compare.")
                    else:
                        st.error(f"Error from translation server: Status Code {response.status_code}")
                        st.text(response.text)
                        
                except Exception as e:
                    st.error(f"Failed to communicate with the translation backend: {e}")
                    st.info("Please make sure your FastAPI backend server is running and accessible at the specified URL.")

if __name__ == "__main__":
    main()
