# PolyglotTranslate 🌐

PolyglotTranslate is a high-performance Streamlit application designed for automatic multilingual text-to-English translation. It splits uploaded documents into paragraphs, automatically detects the language of each paragraph using **fastText (`lid.176.bin`)**, and translates all non-English paragraphs to English using a CTranslate2 optimized, INT8-quantized **NLLB-200 Distilled 600M** model.

## Features

- **Automated Language Identification:** Leverages fastText's lightweight `lid.176.bin` model (covers 176 languages) to instantly identify the language of each paragraph.
- **Low-Latency Translation:** Uses `CTranslate2` with an INT8-quantized `NLLB-200` model for fast translation speeds and low memory footprint.
- **Structure Preserving:** Translates paragraph-by-paragraph, maintaining the original document structure and layout (empty lines, paragraphs, and order).
- **English-Bypassing:** Bypasses translation for paragraphs detected as English to save compute resources.
- **Premium User Interface:** Modern web aesthetics featuring dark mode, glassmorphic metric cards, interactive side-by-side translation comparisons, and progress trackers.
- **Scalable Architecture:** Implements CTranslate2 batch execution, translating multiple paragraphs in parallel.

---

## Codebase Organization

The project is structured as follows:
- **`app.py`**: Streamlit web application interface, custom CSS theme, and translation orchestrator.
- **`detector.py`**: `LanguageDetector` class wrapping fastText, including automatic download of `lid.176.bin` and mapping to NLLB codes.
- **`translator_engine.py`**: `TranslationEngine` class wrapping CTranslate2 and tokenizer batch translation logic.
- **`requirements.txt`**: Package dependencies.

---

## Setup & Installation

### Prerequisites
- **Python 3.11** installed on your system.
- An active internet connection (to download the models on the first run).
- Visual C++ Redistributable (normally pre-installed on Windows).

### 1. Clone or Open Project Folder
Open your terminal (PowerShell / Command Prompt) in the directory containing the project:
```bash
cd c:\Users\khatraks\Downloads\Trans
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory (if it doesn't already exist) and add your Hugging Face authentication token:
```env
HF_TOKEN=your_huggingface_write_token_here
```
*Note: Make sure there are no spaces or quotes around the token itself. If you're accessing public repositories, you can leave it empty or comment it out; sending an invalid token can cause a 401 error on public repos.*

### 3. Create and Activate a Virtual Environment

It is highly recommended to isolate dependencies inside a virtual environment:
```powershell
# Create venv
python -m venv venv

# Activate venv (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate venv (Command Prompt)
.\venv\Scripts\activate.bat
```

### 4. Install Dependencies
Install all required libraries using the pre-configured `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> [!TIP]
> **PyTorch (torch) Installation Size Tip:**
> Streamlit's code-change watcher and the Hugging Face `transformers` codebase require `torch` to be installed in your environment.
> By default, `pip install torch` will download a large package (~1.2 GB) containing CUDA libraries. If you only plan to use **CPU inference**, you can install a lightweight **CPU-only PyTorch** version first (only ~150 MB), then run the requirements install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

*Note: We use `fasttext-predict` instead of the original `fasttext` package because it contains pre-compiled binaries for Windows, preventing compiler errors.*

---

## Running the Application

To start the Streamlit web application:
```bash
streamlit run app.py
```
This will start a local server and automatically open a tab in your default web browser (usually at `http://localhost:8501`).

### First-Run Note 📥
On the very first launch:
1. The app will automatically download the fastText model file (`lid.176.bin`, **~126 MB**) into the `models/` directory.
2. The app will download the CTranslate2 quantized model (`michaelfeil/ct2fast-nllb-200-distilled-600M`, **~1.2 GB**) from Hugging Face.
*This download occurs only once; subsequent launches will load the cached local models immediately.*

---

## How to Use the App

1. **Upload your Text File:** Drag and drop or upload a `.txt` file encoded in UTF-8.
2. **Configure Settings (Sidebar):**
   - **Inference Device:** Select `cpu` (default) or `cuda` if you have a compatible NVIDIA GPU.
   - **Confidence Threshold:** Adjust the minimum language detection confidence. Predictions below this threshold are treated as English and kept unchanged.
   - **Batch Size:** Configure the level of parallel translation (default: 8).
3. **Run:** Click the **🚀 Start Translation Process** button.
4. **Download Result:** Click the **💾 Download output.txt** button to save your translated document.
5. **Analyze:** Inspect the paragraph-by-paragraph comparison cards with detected language tags and confidence.
