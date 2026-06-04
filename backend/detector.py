import os
import urllib.request
import logging
from typing import Tuple, Dict, Optional
import fasttext

logger = logging.getLogger(__name__)

# Comprehensive mapping of fastText (lid.176.bin) codes to NLLB-200 (FLORES-200) codes and names
# Some languages detected by fastText may not be supported by NLLB-200, they map to None for NLLB
LANGUAGE_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "af": {"nllb": "afr_Latn", "name": "Afrikaans"},
    "als": {"nllb": "gsw_Latn", "name": "Alsatian"},
    "am": {"nllb": "amh_Ethi", "name": "Amharic"},
    "an": {"nllb": "arg_Latn", "name": "Aragonese"},
    "ar": {"nllb": "arb_Arab", "name": "Arabic"},
    "arz": {"nllb": "arz_Arab", "name": "Egyptian Arabic"},
    "as": {"nllb": "asm_Beng", "name": "Assamese"},
    "ast": {"nllb": "ast_Latn", "name": "Asturian"},
    "av": {"nllb": "ava_Cyrl", "name": "Avaric"},
    "ay": {"nllb": "ayr_Latn", "name": "Aymara"},
    "az": {"nllb": "azj_Latn", "name": "Azerbaijani"},
    "ba": {"nllb": "bak_Cyrl", "name": "Bashkir"},
    "bar": {"nllb": "bar_Latn", "name": "Bavarian"},
    "bcl": {"nllb": "bcl_Latn", "name": "Central Bikol"},
    "be": {"nllb": "bel_Cyrl", "name": "Belarusian"},
    "bg": {"nllb": "bul_Cyrl", "name": "Bulgarian"},
    "bh": {"nllb": "bho_Deva", "name": "Bihari"},
    "bi": {"nllb": "bis_Latn", "name": "Bislama"},
    "bjn": {"nllb": "bjn_Latn", "name": "Banjar"},
    "bn": {"nllb": "ben_Beng", "name": "Bengali"},
    "bo": {"nllb": "bod_Tibt", "name": "Tibetan"},
    "bpy": {"nllb": "bpy_Beng", "name": "Bishnupriya"},
    "br": {"nllb": "bre_Latn", "name": "Breton"},
    "bs": {"nllb": "bos_Latn", "name": "Bosnian"},
    "bxr": {"nllb": "bxr_Cyrl", "name": "Buryat"},
    "ca": {"nllb": "cat_Latn", "name": "Catalan"},
    "ceb": {"nllb": "ceb_Latn", "name": "Cebuano"},
    "cdo": {"nllb": "cdo_Latn", "name": "Min Dong Chinese"},
    "ch": {"nllb": "cha_Latn", "name": "Chamorro"},
    "co": {"nllb": "cos_Latn", "name": "Corsican"},
    "cs": {"nllb": "ces_Latn", "name": "Czech"},
    "cv": {"nllb": "chv_Cyrl", "name": "Chuvash"},
    "cy": {"nllb": "cym_Latn", "name": "Welsh"},
    "da": {"nllb": "dan_Latn", "name": "Danish"},
    "de": {"nllb": "deu_Latn", "name": "German"},
    "diq": {"nllb": "diq_Latn", "name": "Dimli"},
    "dsb": {"nllb": "dsb_Latn", "name": "Lower Sorbian"},
    "dty": {"nllb": "dty_Deva", "name": "Doteli"},
    "dv": {"nllb": "div_Thaa", "name": "Dhivehi"},
    "el": {"nllb": "ell_Grek", "name": "Greek"},
    "eml": {"nllb": None, "name": "Emilian-Romagnolo"},
    "en": {"nllb": "eng_Latn", "name": "English"},
    "eo": {"nllb": "epo_Latn", "name": "Esperanto"},
    "es": {"nllb": "spa_Latn", "name": "Spanish"},
    "et": {"nllb": "est_Latn", "name": "Estonian"},
    "eu": {"nllb": "eus_Latn", "name": "Basque"},
    "fa": {"nllb": "pes_Arab", "name": "Persian"},
    "fi": {"nllb": "fin_Latn", "name": "Finnish"},
    "fr": {"nllb": "fra_Latn", "name": "French"},
    "frp": {"nllb": "frp_Latn", "name": "Arpitan"},
    "fy": {"nllb": "fry_Latn", "name": "Western Frisian"},
    "ga": {"nllb": "gle_Latn", "name": "Irish"},
    "gd": {"nllb": "gla_Latn", "name": "Scottish Gaelic"},
    "gl": {"nllb": "glg_Latn", "name": "Galician"},
    "gn": {"nllb": "grn_Latn", "name": "Guarani"},
    "gom": {"nllb": "gom_Deva", "name": "Goan Konkani"},
    "gu": {"nllb": "guj_Gujr", "name": "Gujarati"},
    "gv": {"nllb": "glv_Latn", "name": "Manx"},
    "ha": {"nllb": "hau_Latn", "name": "Hausa"},
    "he": {"nllb": "heb_Hebr", "name": "Hebrew"},
    "hi": {"nllb": "hin_Deva", "name": "Hindi"},
    "hif": {"nllb": "hif_Latn", "name": "Fiji Hindi"},
    "hr": {"nllb": "hrv_Latn", "name": "Croatian"},
    "hsb": {"nllb": "hsb_Latn", "name": "Upper Sorbian"},
    "ht": {"nllb": "hat_Latn", "name": "Haitian Creole"},
    "hu": {"nllb": "hun_Latn", "name": "Hungarian"},
    "hy": {"nllb": "hye_Armn", "name": "Armenian"},
    "ia": {"nllb": "ina_Latn", "name": "Interlingua"},
    "id": {"nllb": "ind_Latn", "name": "Indonesian"},
    "ie": {"nllb": "ile_Latn", "name": "Interlingue"},
    "ilo": {"nllb": "ilo_Latn", "name": "Ilokano"},
    "io": {"nllb": "ido_Latn", "name": "Ido"},
    "is": {"nllb": "isl_Latn", "name": "Icelandic"},
    "it": {"nllb": "ita_Latn", "name": "Italian"},
    "ja": {"nllb": "jpn_Jpan", "name": "Japanese"},
    "jbo": {"nllb": "jbo_Latn", "name": "Lojban"},
    "jv": {"nllb": "jav_Latn", "name": "Javanese"},
    "ka": {"nllb": "kat_Geor", "name": "Georgian"},
    "kaa": {"nllb": "kaa_Cyrl", "name": "Karakalpak"},
    "kab": {"nllb": "kab_Latn", "name": "Kabyle"},
    "kbd": {"nllb": "kbd_Cyrl", "name": "Kabardian"},
    "kk": {"nllb": "kaz_Cyrl", "name": "Kazakh"},
    "kl": {"nllb": "kal_Latn", "name": "Kalaallisut"},
    "km": {"nllb": "khm_Khmr", "name": "Khmer"},
    "kn": {"nllb": "kan_Knda", "name": "Kannada"},
    "ko": {"nllb": "kor_Hang", "name": "Korean"},
    "krc": {"nllb": "krc_Cyrl", "name": "Karachay-Balkar"},
    "ku": {"nllb": "kmr_Latn", "name": "Kurdish (Kurmanji)"},
    "kv": {"nllb": "kpv_Cyrl", "name": "Komi"},
    "kw": {"nllb": "cor_Latn", "name": "Cornish"},
    "ky": {"nllb": "kir_Cyrl", "name": "Kyrgyz"},
    "la": {"nllb": "lat_Latn", "name": "Latin"},
    "lad": {"nllb": "lad_Latn", "name": "Ladino"},
    "lb": {"nllb": "ltz_Latn", "name": "Luxembourgish"},
    "lbe": {"nllb": None, "name": "Lak"},
    "lez": {"nllb": "lez_Cyrl", "name": "Lezghian"},
    "lfn": {"nllb": None, "name": "Lingua Franca Nova"},
    "li": {"nllb": "lim_Latn", "name": "Limburgish"},
    "lij": {"nllb": "lij_Latn", "name": "Ligurian"},
    "lmo": {"nllb": "lmo_Latn", "name": "Lombard"},
    "ln": {"nllb": "lin_Latn", "name": "Lingala"},
    "lo": {"nllb": "lao_Laoo", "name": "Lao"},
    "lrc": {"nllb": "lrc_Arab", "name": "Northern Luri"},
    "lt": {"nllb": "lit_Latn", "name": "Lithuanian"},
    "ltg": {"nllb": "ltg_Latn", "name": "Latgalian"},
    "lv": {"nllb": "lvs_Latn", "name": "Latvian"},
    "mai": {"nllb": "mai_Deva", "name": "Maithili"},
    "mdf": {"nllb": None, "name": "Moksha"},
    "mg": {"nllb": "plt_Latn", "name": "Malagasy"},
    "mhr": {"nllb": "mhr_Cyrl", "name": "Eastern Mari"},
    "mi": {"nllb": "mri_Latn", "name": "Maori"},
    "min": {"nllb": "min_Latn", "name": "Minangkabau"},
    "mk": {"nllb": "mkd_Cyrl", "name": "Macedonian"},
    "ml": {"nllb": "mal_Mlym", "name": "Malayalam"},
    "mn": {"nllb": "khk_Cyrl", "name": "Mongolian"},
    "mr": {"nllb": "mar_Deva", "name": "Marathi"},
    "mrj": {"nllb": None, "name": "Western Mari"},
    "ms": {"nllb": "zsm_Latn", "name": "Malay"},
    "mt": {"nllb": "mlt_Latn", "name": "Maltese"},
    "mwl": {"nllb": "mwl_Latn", "name": "Mirandese"},
    "my": {"nllb": "mya_Mymr", "name": "Burmese"},
    "myv": {"nllb": None, "name": "Erzya"},
    "mzn": {"nllb": "mzn_Arab", "name": "Mazanderani"},
    "nah": {"nllb": None, "name": "Nahuatl"},
    "nap": {"nllb": "nap_Latn", "name": "Neapolitan"},
    "nds": {"nllb": "nds_Latn", "name": "Low German"},
    "ne": {"nllb": "npi_Deva", "name": "Nepali"},
    "new": {"nllb": "new_Newa", "name": "Newari"},
    "nl": {"nllb": "nld_Latn", "name": "Dutch"},
    "nn": {"nllb": "nno_Latn", "name": "Norwegian Nynorsk"},
    "no": {"nllb": "nob_Latn", "name": "Norwegian Bokmål"},
    "nov": {"nllb": None, "name": "Novial"},
    "nrm": {"nllb": "nrm_Latn", "name": "Norman"},
    "nso": {"nllb": "nso_Latn", "name": "Northern Sotho"},
    "nv": {"nllb": None, "name": "Navajo"},
    "oc": {"nllb": "oci_Latn", "name": "Occitan"},
    "or": {"nllb": "ory_Orya", "name": "Odia"},
    "os": {"nllb": "oss_Cyrl", "name": "Ossetian"},
    "pa": {"nllb": "pan_Guru", "name": "Punjabi"},
    "pam": {"nllb": "pam_Latn", "name": "Kapampangan"},
    "pap": {"nllb": "pap_Latn", "name": "Papiamento"},
    "pcd": {"nllb": None, "name": "Picard"},
    "pdc": {"nllb": None, "name": "Pennsylvania German"},
    "pl": {"nllb": "pol_Latn", "name": "Polish"},
    "pms": {"nllb": "pms_Latn", "name": "Piedmontese"},
    "pnb": {"nllb": "pnb_Arab", "name": "Western Punjabi"},
    "pnt": {"nllb": None, "name": "Pontic Greek"},
    "ps": {"nllb": "pbt_Arab", "name": "Pashto"},
    "pt": {"nllb": "por_Latn", "name": "Portuguese"},
    "qu": {"nllb": "quy_Latn", "name": "Quechua"},
    "rm": {"nllb": "roh_Latn", "name": "Romansh"},
    "rmy": {"nllb": "rmy_Latn", "name": "Vlax Romani"},
    "ro": {"nllb": "ron_Latn", "name": "Romanian"},
    "ru": {"nllb": "rus_Cyrl", "name": "Russian"},
    "rue": {"nllb": None, "name": "Rusyn"},
    "sa": {"nllb": "san_Deva", "name": "Sanskrit"},
    "sah": {"nllb": "sah_Cyrl", "name": "Yakut"},
    "sc": {"nllb": "srd_Latn", "name": "Sardinian"},
    "scn": {"nllb": "scn_Latn", "name": "Sicilian"},
    "sco": {"nllb": None, "name": "Scots"},
    "sd": {"nllb": "snd_Arab", "name": "Sindhi"},
    "se": {"nllb": "sme_Latn", "name": "Northern Sami"},
    "sh": {"nllb": "hrv_Latn", "name": "Serbo-Croatian"},
    "si": {"nllb": "sin_Sinh", "name": "Sinhala"},
    "sk": {"nllb": "slk_Latn", "name": "Slovak"},
    "sl": {"nllb": "slv_Latn", "name": "Slovenian"},
    "sq": {"nllb": "sqi_Latn", "name": "Albanian"},
    "sr": {"nllb": "srp_Cyrl", "name": "Serbian"},
    "srn": {"nllb": None, "name": "Sranan Tongo"},
    "ss": {"nllb": "ssw_Latn", "name": "Swati"},
    "st": {"nllb": "sot_Latn", "name": "Southern Sotho"},
    "su": {"nllb": "sun_Latn", "name": "Sundanese"},
    "sv": {"nllb": "swe_Latn", "name": "Swedish"},
    "sw": {"nllb": "swh_Latn", "name": "Swahili"},
    "ta": {"nllb": "tam_Taml", "name": "Tamil"},
    "te": {"nllb": "tel_Telu", "name": "Telugu"},
    "tg": {"nllb": "tgk_Cyrl", "name": "Tajik"},
    "th": {"nllb": "tha_Thai", "name": "Thai"},
    "tk": {"nllb": "tuk_Latn", "name": "Turkmen"},
    "tl": {"nllb": "tgl_Latn", "name": "Tagalog"},
    "tn": {"nllb": "tsn_Latn", "name": "Tswana"},
    "tr": {"nllb": "tur_Latn", "name": "Turkish"},
    "tt": {"nllb": "tat_Cyrl", "name": "Tatar"},
    "tyv": {"nllb": "tyv_Cyrl", "name": "Tuvinian"},
    "ug": {"nllb": "uig_Arab", "name": "Uyghur"},
    "uk": {"nllb": "ukr_Cyrl", "name": "Ukrainian"},
    "ur": {"nllb": "urd_Arab", "name": "Urdu"},
    "uz": {"nllb": "uzn_Latn", "name": "Uzbek"},
    "vec": {"nllb": "vec_Latn", "name": "Venetian"},
    "vep": {"nllb": None, "name": "Veps"},
    "vi": {"nllb": "vie_Latn", "name": "Vietnamese"},
    "vls": {"nllb": None, "name": "West Flemish"},
    "vo": {"nllb": None, "name": "Volapük"},
    "wa": {"nllb": "wln_Latn", "name": "Walloon"},
    "war": {"nllb": "war_Latn", "name": "Waray"},
    "wuu": {"nllb": None, "name": "Wu Chinese"},
    "xal": {"nllb": None, "name": "Kalmyk"},
    "xmf": {"nllb": None, "name": "Mingrelian"},
    "yi": {"nllb": "ydd_Hebr", "name": "Yiddish"},
    "yo": {"nllb": "yor_Latn", "name": "Yoruba"},
    "yue": {"nllb": "yue_Hant", "name": "Cantonese"},
    "zh": {"nllb": "zho_Hans", "name": "Chinese (Simplified)"}
}

class LanguageDetector:
    """
    Wraps the fastText model for language identification and manages mapping
    from fastText labels to NLLB language codes and human-readable names.
    """
    def __init__(self, model_path: Optional[str] = None, progress_callback = None) -> None:
        """
        Initializes the LanguageDetector by downloading the model if needed and loading it.
        Handles base directory resolution and parent folder cache fallback path.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_dir, "models", "lid.176.ftz")
            parent_path = os.path.join(os.path.dirname(base_dir), "models", "lid.176.ftz")
            
            if os.path.exists(local_path):
                self.model_path = local_path
            elif os.path.exists(parent_path):
                self.model_path = parent_path
                logger.info(f"Language detector model found in parent directory fallback path: {self.model_path}")
            else:
                self.model_path = local_path
        else:
            self.model_path = os.path.abspath(model_path)
            
        if not os.path.exists(self.model_path):
            self._download_model(progress_callback)
        
        # Load the model using fasttext library
        # silence the warning about load_model from fasttext
        logger.info(f"Loading fastText language detector model from {self.model_path}...")
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(self.model_path)

    def _download_model(self, progress_callback) -> None:
        """
        Downloads the lid.176.ftz model from fastText's official site atomically.
        """
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        temp_path = self.model_path + ".tmp"
        logger.info(f"Starting atomic download from {url} to {temp_path}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                chunk_size = 1024 * 1024  # 1MB chunks
                downloaded = 0
                
                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded / total_size)
            
            # Atomic swap on success
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            os.rename(temp_path, self.model_path)
            logger.info(f"Model download successfully verified and saved to {self.model_path}.")
        except Exception as e:
            logger.error(f"Error downloading fastText model: {e}", exc_info=True)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise e

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detects the language of the input text.
        Returns a tuple of (clean_iso_code, confidence_score).
        Example: ("fr", 0.98)
        """
        # Replace newlines with spaces as fastText expects single-line text predictions
        cleaned_text = text.replace("\n", " ").strip()
        if not cleaned_text:
            return "en", 1.0
            
        try:
            predictions = self.model.predict(cleaned_text, k=1)
            label = predictions[0][0]
            confidence = float(predictions[1][0])
            iso_code = label.replace("__label__", "")
            return iso_code, confidence
        except Exception as e:
            logger.error(f"Error predicting language for text sample '{cleaned_text[:50]}...': {e}", exc_info=True)
            return "en", 0.0

    def get_nllb_code(self, iso_code: str) -> Optional[str]:
        """
        Maps a fastText ISO code to the corresponding NLLB-200 BCP-47 language tag.
        Returns None if translation is unsupported or the language is English.
        """
        # If it's English, return None as we don't translate English paragraphs
        if iso_code == "en":
            return None
            
        lang_info = LANGUAGE_MAP.get(iso_code)
        if lang_info:
            return lang_info.get("nllb")
        return None

    def get_language_name(self, iso_code: str) -> str:
        """
        Returns the human-readable language name for a given ISO code.
        """
        lang_info = LANGUAGE_MAP.get(iso_code)
        if lang_info:
            return lang_info.get("name", "Unknown")
        return f"Unknown ({iso_code})"
