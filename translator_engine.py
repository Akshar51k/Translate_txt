import os
import sys
import subprocess
import logging
import shutil
from typing import List, Optional
import ctranslate2
from transformers import AutoTokenizer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Resolve absolute paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CT2_MODEL_DIR = os.path.join(BASE_DIR, "models", "nllb-200-ct2-int8")
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

        translated_results: List[str] = []
        target_lang = "eng_Latn"

        # Process in batches for maximum throughput
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_src_langs = src_langs[i : i + batch_size]

            # Prepare batch tokens
            batch_tokens: List[List[str]] = []
            for text, src_lang in zip(batch_texts, batch_src_langs):
                # Set the source language on the tokenizer dynamically
                self.tokenizer.src_lang = src_lang
                # Tokenize text and get token strings
                tokens = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.encode(text)
                )
                batch_tokens.append(tokens)

            # Target prefix instructs the model to translate to English
            target_prefixes = [[target_lang] for _ in batch_texts]

            try:
                # Run CTranslate2 batch translation
                results = self.translator.translate_batch(
                    batch_tokens,
                    target_prefix=target_prefixes,
                    beam_size=4,
                    max_decoding_length=256,
                )

                # Decode the translations
                for result in results:
                    translated_tokens = result.hypotheses[0]
                    translated_ids = self.tokenizer.convert_tokens_to_ids(
                        translated_tokens
                    )
                    decoded_text = self.tokenizer.decode(
                        translated_ids, skip_special_tokens=True
                    )
                    translated_results.append(decoded_text.strip())

            except Exception as e:
                logger.error(f"Error during translation batch: {e}", exc_info=True)
                for text in batch_texts:
                    translated_results.append(f"[Translation Failed] {text}")

        return translated_results
