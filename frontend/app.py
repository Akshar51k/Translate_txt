import logging
import os
import requests
import time
from logging.handlers import RotatingFileHandler
import streamlit as st
from dotenv import load_dotenv

# Resolve base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging with RotatingFileHandler to prevent disk space exhaustion in production
log_file = os.path.join(BASE_DIR, "frontend.log")
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Setup console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Setup rotating file handler (10MB limit per file, maximum 5 backup files)
file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10 * 1024 * 1024, 
    backupCount=5, 
    encoding="utf-8"
)
file_handler.setFormatter(log_formatter)

# Initialize root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

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
        value="https://outlet-stamina-similarly.ngrok-free.dev",
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
    
    # Hardcoded confidence threshold for fastText language detection
    confidence_threshold = 0.70
    
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

    uploaded_files = st.file_uploader(
        "Select files to translate (.txt, .rtf, .pdf)", 
        type=["txt", "rtf", "pdf"],
        accept_multiple_files=True,
        help="Upload one or more files to translate them sequentially."
    )
    
    if uploaded_files:
        # Initialize translation results storage in session state
        if "translation_results" not in st.session_state:
            st.session_state.translation_results = {}
            
        # Clean up session state keys that are no longer uploaded
        uploaded_keys = {f"{f.name}_{f.size}" for f in uploaded_files}
        st.session_state.translation_results = {
            k: v for k, v in st.session_state.translation_results.items() if k in uploaded_keys
        }
        
        st.info(f"Loaded **{len(uploaded_files)}** files successfully.")
        
        # Disable translation button if backend is not reachable
        btn_disabled = not backend_healthy
        
        if btn_disabled:
            st.warning("⚠️ **Backend unreachable:** Please launch the translation API backend or verify the URL settings in the sidebar.")
            
        if st.button("🚀 Start Translation Process", type="primary", key="btn_translate", disabled=btn_disabled):
            logger.info(f"Start Translation Process triggered for {len(uploaded_files)} files")
            
            # Setup sequential progress bar
            main_progress_bar = st.progress(0.0)
            main_progress_text = st.empty()
            
            for file_idx, file in enumerate(uploaded_files):
                file_key = f"{file.name}_{file.size}"
                main_progress_text.markdown(f"**Processing file {file_idx + 1} of {len(uploaded_files)}: `{file.name}`...**")
                
                try:
                    from file_parser import extract_text
                    content = extract_text(file)
                except Exception as e:
                    st.session_state.translation_results[file_key] = {
                        "success": False,
                        "name": file.name,
                        "error": f"Failed to extract text: {str(e)}"
                    }
                    continue
                    
                if not content.strip():
                    st.session_state.translation_results[file_key] = {
                        "success": False,
                        "name": file.name,
                        "error": "The uploaded file is empty."
                    }
                    continue
                    
                # Normalize and split paragraphs
                normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
                paragraphs = [p for p in normalized_content.split("\n\n")]
                
                with st.spinner(f"Translating `{file.name}` ({len(paragraphs)} paragraphs)..."):
                    try:
                        payload = {
                            "paragraphs": paragraphs,
                            "confidence_threshold": confidence_threshold,
                            "batch_size": batch_size
                        }
                        
                        response = requests.post(
                            f"{backend_url.rstrip('/')}/translate",
                            json=payload,
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            st.session_state.translation_results[file_key] = {
                                "success": True,
                                "name": file.name,
                                "data": response.json()
                            }
                        else:
                            st.session_state.translation_results[file_key] = {
                                "success": False,
                                "name": file.name,
                                "error": f"API Error: Status {response.status_code} - {response.text}"
                            }
                    except Exception as e:
                        st.session_state.translation_results[file_key] = {
                            "success": False,
                            "name": file.name,
                            "error": f"Connection Error: {str(e)}"
                        }
                
                # Update main progress bar
                main_progress_bar.progress((file_idx + 1) / len(uploaded_files))
                
            main_progress_text.success("🎉 All files processed!")
            time.sleep(1)
            main_progress_bar.empty()
            main_progress_text.empty()
            
        # Display Results Section if there is any data processed
        if st.session_state.translation_results:
            st.markdown("### 📊 Translation Results")
            
            for file in uploaded_files:
                file_key = f"{file.name}_{file.size}"
                
                if file_key in st.session_state.translation_results:
                    result = st.session_state.translation_results[file_key]
                    
                    # Set accordion title based on success status
                    expander_title = f"📄 {file.name} "
                    if result["success"]:
                        expander_title += "✅ Ready"
                    else:
                        expander_title += "❌ Failed"
                        
                    with st.expander(expander_title, expanded=True):
                        if not result["success"]:
                            st.error(f"Failed to translate: {result['error']}")
                            continue
                            
                        res_data = result["data"]
                        translated_text = res_data["translated_text"]
                        metrics = res_data["metrics"]
                        comparison_data = res_data["comparison_data"]
                        device_used = res_data["device_used"]
                        elapsed_seconds = res_data["elapsed_seconds"]
                        
                        st.success(f"Translation completed successfully on {device_display if backend_healthy else device_used} in {elapsed_seconds:.2f} seconds!")
                        
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
                        
                        # Individual download button
                        base_name, _ = os.path.splitext(file.name)
                        st.download_button(
                            label=f"💾 Download {base_name}_translated.txt",
                            data=translated_text,
                            file_name=f"{base_name}_translated.txt",
                            mime="text/plain",
                            type="primary",
                            key=f"btn_dl_{file_key}"
                        )
                        
                        # Render Comparison List
                        st.markdown("#### 🔍 Paragraph-by-Paragraph Comparison")
                        
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
                            
                        if html_rows:
                            clean_html = "".join(html_rows).replace("\n", "").replace("\r", "")
                            st.markdown(f'<div class="translation-list">{clean_html}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No content paragraphs to compare.")

if __name__ == "__main__":
    main()
