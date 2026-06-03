import os
from typing import List, Optional
import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TranslationEngine:
    """
    Manages the CTranslate2 translation engine and Hugging Face tokenizer.
    Handles batch translation of multilingual paragraphs to English.
    """
    def __init__(
        self,
        model_id: str = "michaelfeil/ct2fast-nllb-200-distilled-600M",
        device: str = "cpu",
        compute_type: str = "int8",
        progress_callback = None
    ) -> None:
        """
        Initializes the TranslationEngine.
        Downloads the CTranslate2 model from Hugging Face if not cached,
        then loads the translator and NLLB tokenizer.
        """
        self.device = device
        self.compute_type = compute_type
        
        # Download NLLB-200 CT2 model from Hugging Face
        # snapshot_download automatically checks if the files are already in the cache
        if progress_callback:
            progress_callback("Downloading NLLB-200 model weights (~1.2 GB)...")
        
        try:
            self.model_dir = snapshot_download(repo_id=model_id)
        except Exception as e:
            # If the download fails due to an invalid token/credentials (401 error),
            # fall back to downloading anonymously (guest mode) since the model is public.
            err_msg = str(e).lower()
            if "401" in err_msg or "unauthorized" in err_msg or "authentication" in err_msg or "password" in err_msg:
                if progress_callback:
                    progress_callback("Hugging Face token invalid or unauthorized. Retrying download in guest mode...")
                self.model_dir = snapshot_download(repo_id=model_id, token=False)
            else:
                raise e

        
        if progress_callback:
            progress_callback("Loading NLLB-200 translation engine...")
            
        # Initialize CTranslate2 Translator with CUDA to CPU fallback handling
        self.device_used = self.device
        self.device_fallback = False
        
        try:
            self.translator = ctranslate2.Translator(
                self.model_dir,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            if self.device == "cuda":
                if progress_callback:
                    progress_callback("CUDA initialization failed. Falling back to CPU...")
                try:
                    self.translator = ctranslate2.Translator(
                        self.model_dir,
                        device="cpu",
                        compute_type=self.compute_type
                    )
                    self.device_used = "cpu"
                    self.device_fallback = True
                except Exception as cpu_err:
                    raise RuntimeError(f"Failed to load translation engine on CPU after CUDA initialization failed: {cpu_err}") from e
            else:
                raise e

        
        if progress_callback:
            progress_callback("Loading NLLB-200 tokenizer...")
            
        # Initialize the tokenizer
        # We load the tokenizer of the original model from HF
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    def translate_batch(self, texts: List[str], src_langs: List[str], batch_size: int = 8) -> List[str]:
        """
        Translates a list of texts from their respective source languages to English.
        
        Args:
            texts: List of paragraphs to translate.
            src_langs: List of NLLB language codes matching the texts (e.g. 'fra_Latn').
            batch_size: Number of paragraphs to translate in parallel.
            
        Returns:
            List of translated English strings in the same order.
        """
        if not texts:
            return []
            
        translated_results: List[str] = []
        target_lang = "eng_Latn"
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_src_langs = src_langs[i:i + batch_size]
            
            # Prepare batch tokens
            batch_tokens = []
            for text, src_lang in zip(batch_texts, batch_src_langs):
                # Set the source language on the tokenizer dynamically
                self.tokenizer.src_lang = src_lang
                # Tokenize text and get token strings
                tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
                batch_tokens.append(tokens)
            
            # Set the target prefix for each item in the batch
            # This instructs the model to translate to English
            target_prefixes = [[target_lang] for _ in range(len(batch_texts))]
            
            try:
                # Run translation batch
                results = self.translator.translate_batch(
                    batch_tokens,
                    target_prefix=target_prefixes,
                    beam_size=4,
                    max_decoding_length=256
                )
                
                # Decode the translations
                for result in results:
                    # Get the translated tokens of the top hypothesis
                    translated_tokens = result.hypotheses[0]
                    # Convert token strings back to IDs and decode to text
                    translated_ids = self.tokenizer.convert_tokens_to_ids(translated_tokens)
                    decoded_text = self.tokenizer.decode(translated_ids, skip_special_tokens=True)
                    translated_results.append(decoded_text.strip())
                    
            except Exception as e:
                # In case of an error, fall back to returning original text for this batch
                print(f"Error during translation batch: {e}")
                # We can try to translate individual sentences to isolate, or just fall back
                for text in batch_texts:
                    translated_results.append(f"[Translation Failed] {text}")
                    
        return translated_results
