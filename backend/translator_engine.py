import os
import sys
import subprocess
import logging
import shutil
import threading
import re
from typing import List, Dict
import ctranslate2
from transformers import AutoTokenizer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Resolve absolute paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_model_dir = os.path.join(BASE_DIR, "models", "nllb-200-ct2-int8")
parent_model_dir = os.path.join(os.path.dirname(BASE_DIR), "models", "nllb-200-ct2-int8")

if os.path.exists(os.path.join(local_model_dir, "model.bin")):
    CT2_MODEL_DIR = local_model_dir
elif os.path.exists(os.path.join(parent_model_dir, "model.bin")):
    CT2_MODEL_DIR = parent_model_dir
    logger.info(f"CTranslate2 model found in parent directory fallback path: {CT2_MODEL_DIR}")
else:
    CT2_MODEL_DIR = local_model_dir

HF_MODEL_ID = "facebook/nllb-200-distilled-600M"


def _convert_model_to_ct2(output_dir: str, progress_callback=None) -> None:
    """
    Converts the public HuggingFace NLLB-200 model to CTranslate2 INT8 format
    and saves it locally. This runs only once on first launch.
    """
    if progress_callback:
        progress_callback(
            "Converting NLLB-200 to CTranslate2 INT8 format "
            "(one-time setup, may take a few minutes)..."
        )

    os.makedirs(output_dir, exist_ok=True)

    # Use the ct2-transformers-converter CLI shipped with ctranslate2
    cmd = [
        "ct2-transformers-converter",
        "--model", HF_MODEL_ID,
        "--output_dir", output_dir,
        "--quantization", "int8",
        "--force",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if progress_callback:
            progress_callback("Model conversion completed successfully!")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fallback: invoke through the Python module interface
        if progress_callback:
            progress_callback("Trying alternative conversion method...")
        try:
            cmd_alt = [
                sys.executable, "-m", "ctranslate2.converters.transformers",
                "--model", HF_MODEL_ID,
                "--output_dir", output_dir,
                "--quantization", "int8",
                "--force",
            ]
            subprocess.run(cmd_alt, capture_output=True, text=True, check=True)
            if progress_callback:
                progress_callback("Model conversion completed successfully!")
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(
                f"Failed to convert model to CTranslate2 format.\n"
                f"Stdout: {e2.stdout}\nStderr: {e2.stderr}"
            ) from e2


class TranslationEngine:
    """
    Manages the CTranslate2 translation engine and Hugging Face tokenizer.
    Handles batch translation of multilingual paragraphs to English with
    low-latency INT8 quantized inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = "int8",
        progress_callback=None,
    ) -> None:
        """
        Initializes the TranslationEngine.

        On first run, downloads the public facebook/nllb-200-distilled-600M model
        and converts it locally to CTranslate2 INT8 format for low-latency inference.
        Subsequent runs load the cached local model instantly.

        Args:
            device: 'cpu' or 'cuda' for GPU acceleration.
            compute_type: CTranslate2 compute type (default 'int8').
            progress_callback: Optional callable for status updates.
        """
        self.device = device
        self.compute_type = compute_type
        self.device_used = device
        self.device_fallback = False
        # Thread safety lock to serialize tokenizer writes (such as self.tokenizer.src_lang)
        self.lock = threading.Lock()

        # --- Step 1: Convert model if not already done ---
        model_bin_path = os.path.join(CT2_MODEL_DIR, "model.bin")
        if not os.path.exists(model_bin_path):
            if progress_callback:
                progress_callback(
                    "First-time setup: downloading and converting NLLB-200 model..."
                )
            try:
                _convert_model_to_ct2(CT2_MODEL_DIR, progress_callback)
            except Exception as e:
                # Clean up output directory on failure to avoid half-converted states
                if os.path.exists(CT2_MODEL_DIR):
                    try:
                        shutil.rmtree(CT2_MODEL_DIR)
                    except Exception:
                        pass
                logger.error(f"Failed to convert NLLB-200 model to CTranslate2 format: {e}", exc_info=True)
                raise e
        else:
            if progress_callback:
                progress_callback("Loading cached CTranslate2 model...")

        # --- Step 2: Load CTranslate2 Translator with CUDA→CPU fallback ---
        if progress_callback:
            progress_callback("Initializing CTranslate2 translation engine...")

        try:
            self.translator = ctranslate2.Translator(
                CT2_MODEL_DIR,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            if self.device == "cuda":
                if progress_callback:
                    progress_callback(
                        "CUDA initialization failed. Falling back to CPU..."
                    )
                try:
                    self.translator = ctranslate2.Translator(
                        CT2_MODEL_DIR,
                        device="cpu",
                        compute_type=self.compute_type,
                    )
                    self.device_used = "cpu"
                    self.device_fallback = True
                except Exception as cpu_err:
                    raise RuntimeError(
                        f"Failed to load on CPU after CUDA failure: {cpu_err}"
                    ) from e
            else:
                raise e

        # --- Step 3: Load tokenizer from the public HuggingFace repo ---
        if progress_callback:
            progress_callback("Loading NLLB-200 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

        if progress_callback:
            progress_callback("Translation engine ready!")

    def translate_batch(
        self,
        texts: List[str],
        src_langs: List[str],
        batch_size: int = 8,
    ) -> List[str]:
        """
        Translates a list of texts from their respective source languages
        to English using CTranslate2 batch inference.

        Args:
            texts: List of paragraphs to translate.
            src_langs: List of NLLB language codes (e.g. 'fra_Latn').
            batch_size: Number of paragraphs to translate in parallel.

        Returns:
            List of translated English strings in the same order.
        """
        if not texts:
            return []

        target_lang = "eng_Latn"
        
        # Adaptive Beam Size configuration based on active hardware to optimize CPU latency vs GPU throughput
        # - GPU (CUDA): beam_size = 4 (maximizes translation quality)
        # - CPU: beam_size = 2 (reduces workload by 2x, providing low latency with high accuracy)
        beam_size = 4 if self.device_used == "cuda" else 2
        
        # Smart Semantic Splitter threshold (in words)
        # Paragraphs under this limit remain untouched to preserve contextual translation quality
        max_words = 200

        flat_texts: List[str] = []
        flat_src_langs: List[str] = []
        
        # Maps original paragraph index to list of indices in flat_texts
        paragraph_map: Dict[int, List[int]] = {}

        # Universal sentence splitter supporting Latin, Cyrillic, Asian, Arabic, and Sanskrit scripts
        # Splits on standard punctuation followed by whitespace/newlines or direct line breaks
        def split_into_sentences(text: str) -> List[str]:
            # Regex splits on:
            # - .!? (Latin/Cyrillic)
            # - 。！？ (Chinese/Japanese/Korean)
            # - । (Hindi/Sanskrit danda)
            # - ؟ (Arabic/Persian question mark)
            # followed by whitespace or line breaks
            sentences = re.split(r'(?<=[.!?。！？।؟])\s+|\n+', text)
            return [s.strip() for s in sentences if s.strip()]

        # ----------------- Step 1: Smart Semantic Splitter (Flattening) -----------------
        for p_idx, (text, lang) in enumerate(zip(texts, src_langs)):
            words = text.split()
            if len(words) <= max_words:
                # If paragraph is within safe limits, translate it as a single chunk
                flat_texts.append(text)
                flat_src_langs.append(lang)
                paragraph_map[p_idx] = [len(flat_texts) - 1]
            else:
                # Split large paragraphs on sentence boundaries
                sentences = split_into_sentences(text)
                chunks = []
                current_chunk = []
                current_count = 0
                
                for s in sentences:
                    s_words = len(s.split())
                    # Group sentences into sub-chunks up to max_words to retain local context
                    if current_count + s_words > max_words:
                        if current_chunk:
                            # Join sentences with space
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [s]
                        current_count = s_words
                    else:
                        current_chunk.append(s)
                        current_count += s_words
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    
                # Safe Fallback: If a single sentence exceeds max_words and contains no punctuation,
                # split strictly by word count to protect the NLLB-200 context limits
                processed_chunks = []
                for chunk in chunks:
                    chunk_words = chunk.split()
                    if len(chunk_words) > max_words:
                        for i in range(0, len(chunk_words), 150):
                            processed_chunks.append(" ".join(chunk_words[i : i + 150]))
                    else:
                        processed_chunks.append(chunk)
                        
                indices = []
                for chunk in processed_chunks:
                    flat_texts.append(chunk)
                    flat_src_langs.append(lang)
                    indices.append(len(flat_texts) - 1)
                paragraph_map[p_idx] = indices

        # ----------------- Step 2: Parallel Batch Translation -----------------
        flat_results: List[str] = []
        
        # Send the entire list of chunks to CTranslate2 in a single call.
        # This allows CTranslate2 to perform global length sorting to minimize padding,
        # distribute execution across physical CPU cores, and run native batching.
        # Uses thread-safety lock because setting self.tokenizer.src_lang is stateful.
        batch_tokens: List[List[str]] = []
        with self.lock:
            for text, src_lang in zip(flat_texts, flat_src_langs):
                self.tokenizer.src_lang = src_lang
                tokens = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.encode(text)
                )
                batch_tokens.append(tokens)

        target_prefixes = [[target_lang] for _ in flat_texts]
        
        try:
            # Execute CTranslate2 batch translation with optimized parameters:
            # - beam_size: Adaptive (4 on GPU, 2 on CPU)
            # - max_batch_size: Configured batch limit
            # - max_decoding_length: 1024 (native NLLB capacity to prevent truncation)
            # - repetition_penalty: 1.1 (discourages repetitive outputs)
            # - no_repeat_ngram_size: 4 (mathematically prevents word repetition looping)
            results = self.translator.translate_batch(
                batch_tokens,
                target_prefix=target_prefixes,
                beam_size=beam_size,
                max_batch_size=batch_size,
                max_decoding_length=1024,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
            )

            # Decode the translations back to string outputs under tokenizer lock
            with self.lock:
                for result in results:
                    translated_tokens = result.hypotheses[0]
                    translated_ids = self.tokenizer.convert_tokens_to_ids(
                        translated_tokens
                    )
                    decoded_text = self.tokenizer.decode(
                        translated_ids, skip_special_tokens=True
                    )
                    flat_results.append(decoded_text.strip())

        except Exception as e:
            logger.error(f"Error during translation batch: {e}", exc_info=True)
            for text in flat_texts:
                flat_results.append(f"[Translation Failed] {text}")

        # ----------------- Step 3: Format-Aware Reassembly (Unflattening) -----------------
        translated_results: List[str] = []
        for p_idx in range(len(texts)):
            chunk_indices = paragraph_map[p_idx]
            chunk_translations = [flat_results[idx] for idx in chunk_indices]
            original_text = texts[p_idx]
            
            # Preserve original visual layout: join with newlines if they existed in the source
            if "\n" in original_text:
                translated_results.append("\n".join(chunk_translations))
            else:
                translated_results.append(" ".join(chunk_translations))

        return translated_results
