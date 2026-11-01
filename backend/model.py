"""
Inference, Explainability, and Psychological Analysis Engine for Emotion Recognition.
Created by: Sumit Kumar (https://github.com/Sumit2003-dev)

Provides:
- Real-time multiclass emotion probabilities across 8 emotions
- Dominant emotion, emoji, color coding
- Sentiment polarity & compound score (-1.0 to +1.0)
- Emotional intensity (Subtle, Moderate, Intense)
- Token-level explainability (keyword attribution & highlight weights)
- Actionable AI empathy tips and communication guidelines
"""

import os
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "emotion_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ur": "Urdu",
    "pa": "Punjabi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "hinglish": "Hinglish (Hindi-English)",
}

# Visual & psychological attributes per emotion
EMOTION_META = {
    "joy": {
        "emoji": "😊",
        "icon": "sparkles",
        "color": "#F59E0B",       # Amber / Warm Gold
        "bg_gradient": "linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(217, 119, 6, 0.1))",
        "glow": "rgba(245, 158, 11, 0.4)",
        "label": "Joy & Elation",
        "sentiment": "Positive",
        "sentiment_val": 0.9,
        "psychology": "Signals satisfaction, accomplishment, vitality, and social connection.",
        "empathy_tip": "Amplify and validate their celebration. Share enthusiasm and acknowledge their win.",
        "suggested_reply": "That is incredible news! Huge congratulations on this wonderful milestone!",
    },
    "sadness": {
        "emoji": "😢",
        "icon": "cloud-rain",
        "color": "#3B82F6",       # Blue / Azure
        "bg_gradient": "linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(29, 78, 216, 0.1))",
        "glow": "rgba(59, 130, 246, 0.4)",
        "label": "Sadness & Grief",
        "sentiment": "Negative",
        "sentiment_val": -0.85,
        "psychology": "Reflects perceived loss, disappointment, fatigue, or emotional vulnerability.",
        "empathy_tip": "Provide a safe, patient presence without immediately rushing to fix everything.",
        "suggested_reply": "I am so deeply sorry you are going through this. Take all the time you need, I am here for you.",
    },
    "anger": {
        "emoji": "😡",
        "icon": "flame",
        "color": "#EF4444",       # Crimson / Red
        "bg_gradient": "linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.1))",
        "glow": "rgba(239, 68, 68, 0.4)",
        "label": "Anger & Frustration",
        "sentiment": "Negative",
        "sentiment_val": -0.8,
        "psychology": "Arises from boundary violations, perceived injustice, blocking of goals, or disrespect.",
        "empathy_tip": "De-escalate immediately. Acknowledge frustration plainly without becoming defensive.",
        "suggested_reply": "I hear your frustration loud and clear, and you have every right to be upset. Let's resolve this directly.",
    },
    "fear": {
        "emoji": "😨",
        "icon": "shield-alert",
        "color": "#8B5CF6",       # Violet / Purple
        "bg_gradient": "linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(109, 40, 217, 0.1))",
        "glow": "rgba(139, 92, 246, 0.4)",
        "label": "Fear & Anxiety",
        "sentiment": "Negative",
        "sentiment_val": -0.7,
        "psychology": "Triggers threat-detection, apprehension of future harm, uncertainty, or loss of safety.",
        "empathy_tip": "Ground them with clear facts, predictability, reassuring presence, and small safe steps.",
        "suggested_reply": "Take a slow deep breath. You are safe right now, and we will take this step by step together.",
    },
    "surprise": {
        "emoji": "😲",
        "icon": "zap",
        "color": "#10B981",       # Emerald / Teal
        "bg_gradient": "linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(5, 150, 105, 0.1))",
        "glow": "rgba(16, 185, 129, 0.4)",
        "label": "Surprise & Astonishment",
        "sentiment": "Neutral / Dynamic",
        "sentiment_val": 0.3,
        "psychology": "A fast orienting reflex responding to unexpected events or schema disruption.",
        "empathy_tip": "Help assimilate the unexpected information and give space to process the novelty.",
        "suggested_reply": "Wow, what a totally unexpected turn of events! How are you processing this?",
    },
    "love": {
        "emoji": "❤️",
        "icon": "heart",
        "color": "#EC4899",       # Rose / Pink
        "bg_gradient": "linear-gradient(135deg, rgba(236, 72, 153, 0.25), rgba(190, 24, 93, 0.1))",
        "glow": "rgba(236, 72, 153, 0.4)",
        "label": "Love & Affection",
        "sentiment": "Positive",
        "sentiment_val": 0.95,
        "psychology": "Fosters attachment, deep trust, warmth, reciprocal intimacy, and emotional safety.",
        "empathy_tip": "Express warmth in return, reciprocate appreciation, and honor the relational bond.",
        "suggested_reply": "Your words mean the world to me. Thank you for your kindness, love, and constant care.",
    },
    "disgust": {
        "emoji": "🤢",
        "icon": "biohazard",
        "color": "#84CC16",       # Lime / Olive
        "bg_gradient": "linear-gradient(135deg, rgba(132, 204, 22, 0.25), rgba(101, 163, 13, 0.1))",
        "glow": "rgba(132, 204, 22, 0.4)",
        "label": "Disgust & Aversion",
        "sentiment": "Negative",
        "sentiment_val": -0.85,
        "psychology": "Protective recoil from physical contaminants or moral degradation and hypocrisy.",
        "empathy_tip": "Validate boundary setting and support physical or emotional distance from the trigger.",
        "suggested_reply": "I completely understand your revulsion. Setting clear boundaries is 100% the right call.",
    },
    "neutral": {
        "emoji": "😐",
        "icon": "minus-circle",
        "color": "#94A3B8",       # Slate / Steel Gray
        "bg_gradient": "linear-gradient(135deg, rgba(148, 163, 184, 0.2), rgba(100, 116, 139, 0.1))",
        "glow": "rgba(148, 163, 184, 0.3)",
        "label": "Neutral & Objective",
        "sentiment": "Neutral",
        "sentiment_val": 0.0,
        "psychology": "Informational, pragmatic, task-oriented state with baseline physiological arousal.",
        "empathy_tip": "Maintain clear, structured, professional, and efficient communication.",
        "suggested_reply": "Thank you for the update. The information has been noted and logged accordingly.",
    },
}

# Emotion keyword cues for explainability / token highlight
KEYWORD_CUES = {
    "joy": ["thrilled", "ecstatic", "happy", "happiness", "joy", "bliss", "wonderful", "celebrating", "celebration", "delight", "delighted", "delightful", "cheers", "proud", "smile", "smiling", "awesome", "sunshine", "win", "won", "winning", "great", "fantastic", "achieved", "soaring", "overjoyed", "blessed", "liberated", "flying", "refreshed", "excited", "ecstasy", "superb", "triumph", "pleased", "grateful", "breakthrough"],
    "sadness": ["lonely", "empty", "sad", "sadness", "broken", "heartbroken", "grief", "grieving", "crying", "tears", "loss", "sorrow", "depressed", "depression", "devastated", "down", "bleak", "hopeless", "abandoned", "hurts", "painful", "melancholy", "drained", "void", "miserable", "mourning", "sorrowful", "alone", "gloomy"],
    "anger": ["furious", "angry", "rage", "livid", "infuriating", "outraged", "hate", "scam", "disrespect", "toxic", "boiling", "sick", "tired", "negligence", "unacceptable", "idiocy", "coward", "audacity", "cheated", "lies", "interrupted", "disgusted", "annoyed", "irritated", "fuming", "hostile", "mad"],
    "fear": ["terrified", "panic", "fear", "scared", "dread", "anxiety", "anxious", "trembling", "phobia", "fright", "paralyzed", "sinister", "nightmare", "chilling", "complications", "screaming", "menacing", "danger", "petrified", "spooked", "nervous", "horrified", "panicked"],
    "surprise": ["shock", "shocked", "shocking", "astonished", "astonishing", "unbelievable", "wow", "whoa", "unexpected", "jaw", "gasp", "gasped", "disbelief", "stunned", "speechless", "startled", "coincidence", "sudden", "unforeseen", "dumbfounded", "amazed", "astounded"],
    "love": ["love", "soulmate", "adoration", "adore", "cherish", "darling", "sweetheart", "kiss", "hugs", "romance", "romantic", "affection", "tender", "tenderness", "beloved", "cuddle", "partner", "forever", "heart", "warmth", "devotion", "sweet", "precious", "loving"],
    "disgust": ["disgust", "disgusted", "disgusting", "nauseating", "repulsive", "repulsed", "revolting", "foul", "filthy", "vomit", "slimy", "maggots", "vile", "putrid", "abhorrent", "contempt", "stomach", "stench", "curdled", "festering", "rot", "rotten", "gross", "sickening", "nasty"],
    "neutral": ["meeting", "schedule", "report", "document", "standard", "operating", "temperature", "verified", "system", "process", "hours", "status", "data", "information", "guidelines", "invoice", "transit", "department", "regular", "confirmed", "update"],
}


class EmotionClassifier:
    """Production Emotion Classifier with inference, token attribution, and psychological tips."""

    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_FILE
        self.metadata_path = metadata_path or METADATA_FILE
        self.pipeline = None
        self.transformer_pipeline = None
        self.metadata = {}
        self.classes = list(EMOTION_META.keys())
        self._translation_cache: Dict[str, Dict[str, Any]] = {}
        self._load_model()
        self._load_transformer()

    def _load_model(self):
        """Loads serialized model and metadata."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                if hasattr(self.pipeline, "classes_"):
                    self.classes = sorted(list(self.pipeline.classes_))
            except Exception as e:
                print(f"Error loading model pipeline from {self.model_path}: {e}")

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata from {self.metadata_path}: {e}")

    def _load_transformer(self):
        """Loads pre-trained DistilRoBERTa model if torch and transformers are available."""
        if os.environ.get("DISABLE_TRANSFORMER", "0").lower() in ("1", "true", "yes"):
            print("Notice: Transformer engine disabled via environment variable.")
            self.transformer_pipeline = None
            return

        try:
            from transformers import pipeline
            # Try offline / cached model first to eliminate network latency and timeout hangs
            try:
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1,
                    local_files_only=True,
                )
            except Exception:
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1,
                )
            print("Neural Transformer Emotion Engine (DistilRoBERTa) loaded successfully.")
        except Exception as e:
            print(f"Notice: Transformer engine not active (falling back to Calibrated LinearSVC): {e}")
            self.transformer_pipeline = None

    def _detect_and_translate(self, text: str) -> Dict[str, Any]:
        """
        Cross-lingual intelligence engine:
        Detects language of input text (Hindi, Spanish, French, German, Italian, Portuguese, Arabic, Hinglish, etc.).
        If non-English, dynamically provides cross-lingual translation into English so that the neural transformer
        and calibrated SVM classifiers achieve state-of-the-art accuracy across 100+ languages.
        """
        clean = (text or "").strip()
        if not clean:
            return {
                "working_text": clean,
                "detected_language": "en",
                "language_name": "English",
                "is_multilingual": False,
                "translated_text": None,
            }

        if clean in self._translation_cache:
            return self._translation_cache[clean]

        hinglish_words = {"bohot", "bahut", "mujhe", "pyaar", "pyar", "khush", "gussa", "dukh", "darr", "nahi", "accha", "bura", "bakwas", "kya", "shanti", "pareshan"}
        tokens_lower = set(clean.lower().split())
        has_hinglish = bool(tokens_lower.intersection(hinglish_words))

        allow_network = getattr(self, "_online_translate_enabled", True) and not os.environ.get("OFFLINE_MODE")
        foreign_tokens = {"estoy", "avec", "nicht", "dans", "nous", "vous", "pero", "como", "triste", "heureux", "muy", "tres"}
        is_foreign = bool(tokens_lower.intersection(foreign_tokens))
        has_non_ascii = not clean.isascii()

        # Instant path: if plain ASCII and not foreign/hinglish, or network disabled, return immediately without network overhead
        if not allow_network or (not has_non_ascii and not has_hinglish and not is_foreign):
            lang_name = "Hinglish (Hindi-English)" if has_hinglish else "English"
            lang_code = "hinglish" if has_hinglish else "en"
            result = {
                "working_text": clean,
                "detected_language": lang_code,
                "language_name": lang_name,
                "is_multilingual": has_hinglish,
                "translated_text": None,
            }
            self._translation_cache[clean] = result
            return result

        try:
            url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(clean)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                translated = "".join([part[0] for part in data[0] if part[0]]).strip()
                lang_code = data[2] if len(data) > 2 and data[2] else "en"

                is_multi = (lang_code != "en") or has_hinglish
                if has_hinglish and lang_code in ["en", "hi", "und"]:
                    lang_name = "Hinglish (Hindi-English)"
                    lang_code = "hinglish"
                else:
                    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())

                result = {
                    "working_text": translated if is_multi and translated else clean,
                    "detected_language": lang_code,
                    "language_name": lang_name,
                    "is_multilingual": is_multi,
                    "translated_text": translated if is_multi and translated.lower() != clean.lower() else None,
                }
                if len(self._translation_cache) > 1000:
                    self._translation_cache.clear()
                self._translation_cache[clean] = result
                return result
        except Exception:
            # Circuit breaker: disable online translate for current session to prevent cascading latency
            self._online_translate_enabled = False
            lang_name = "Hinglish (Hindi-English)" if has_hinglish else "English"
            lang_code = "hinglish" if has_hinglish else "en"
            result = {
                "working_text": clean,
                "detected_language": lang_code,
                "language_name": lang_name,
                "is_multilingual": has_hinglish,
                "translated_text": None,
            }
            self._translation_cache[clean] = result
            return result

    def is_ready(self) -> bool:
        return (self.pipeline is not None) or (self.transformer_pipeline is not None)

    @property
    def active_engine(self) -> str:
        if self.transformer_pipeline is not None and self.pipeline is not None:
            return "Hybrid Multilingual Neural Transformer (DistilRoBERTa + Calibrated LinearSVC)"
        if self.transformer_pipeline is not None:
            return "Neural Transformer (DistilRoBERTa)"
        if self.pipeline is not None:
            return "Calibrated LinearSVC (Platt Scaling)"
        return "Rule Heuristics Baseline"

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs emotion inference, token attribution, sentiment analysis, and psychological guidance."""
        raw_text = (text or "").strip()
        if not raw_text:
            return self._empty_response()

        # 0. Multilingual Detection & Cross-Lingual Alignment
        lang_info = self._detect_and_translate(raw_text)
        working_text = lang_info["working_text"]

        # 1. Feature probabilities from Scikit-Learn pipeline
        probabilities: Dict[str, float] = {}
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "predict_proba"):
                    probs = self.pipeline.predict_proba([working_text])[0]
                    classes = self.pipeline.classes_
                    for cls_name, prob in zip(classes, probs):
                        probabilities[cls_name] = float(prob)
                elif hasattr(self.pipeline, "decision_function"):
                    df = self.pipeline.decision_function([working_text])[0]
                    exp_df = np.exp(df - np.max(df))
                    probs = exp_df / np.sum(exp_df)
                    for cls_name, prob in zip(self.pipeline.classes_, probs):
                        probabilities[cls_name] = float(prob)
            except Exception as e:
                print(f"Scikit-learn inference notice: {e}")

        # Ensure all standard 8 classes exist in baseline
        for cls_name in EMOTION_META.keys():
            if cls_name not in probabilities:
                probabilities[cls_name] = 0.001

        # 2. If Neural Transformer is active, compute deep contextual probabilities & ensemble blend
        if self.transformer_pipeline is not None:
            try:
                tr_results = self.transformer_pipeline(working_text[:512])[0]
                tr_probs = {item["label"]: float(item["score"]) for item in tr_results}

                hybrid_probs = {}
                for c in self.classes:
                    p_sk = probabilities.get(c, 0.0)
                    p_tr = tr_probs.get(c, 0.0)
                    if c == "love":
                        is_love_term = bool(re.search(r"\b(love|loved|soulmate|marry|adore|cherish|darling|cuddle|heart|beloved)\b", working_text.lower())) or bool(re.search(r"\b(pyaar|pyar|prem|ishq|mohabbat)\b", raw_text.lower()))
                        hybrid_probs[c] = 0.65 * p_sk + (0.35 * tr_probs.get("joy", 0.0) if is_love_term else 0.0)
                    elif c == "joy":
                        hybrid_probs[c] = 0.40 * p_sk + 0.60 * p_tr
                    else:
                        hybrid_probs[c] = 0.35 * p_sk + 0.65 * p_tr

                tot = sum(hybrid_probs.values())
                if tot > 0:
                    probabilities = {k: round(v / tot, 4) for k, v in hybrid_probs.items()}
            except Exception as e:
                print(f"Notice: Transformer inference notice: {e}")

        # Normalize probabilities to sum to 1.0
        total_p = sum(probabilities.values())
        if total_p > 0:
            probabilities = {k: round(v / total_p, 4) for k, v in probabilities.items()}

        # 2. Rule & Cue Enhancement for High-Impact Emotive Tokens
        probabilities = self._apply_lexical_adjustments(working_text, probabilities)
        if lang_info["is_multilingual"]:
            probabilities = self._apply_lexical_adjustments(raw_text, probabilities)

        # 3. Dominant emotion selection
        sorted_emotions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        dominant_emotion, confidence = sorted_emotions[0]

        # 4. Intensity Assessment
        intensity_level, intensity_score = self._compute_intensity(raw_text, confidence)

        # 5. Sentiment Polarity & Compound Score
        sentiment_label, compound_score = self._compute_sentiment(probabilities, dominant_emotion)

        # 6. Token Attribution (Explainability)
        token_attributions = self._extract_token_attributions(
            working_text if lang_info["is_multilingual"] else raw_text,
            dominant_emotion
        )

        # 7. Metadata & Empathy Guidance
        meta = EMOTION_META.get(dominant_emotion, EMOTION_META["neutral"])

        return {
            "text": raw_text,
            "detected_language": lang_info["detected_language"],
            "language_name": lang_info["language_name"],
            "is_multilingual": lang_info["is_multilingual"],
            "translated_text": lang_info["translated_text"],
            "dominant_emotion": dominant_emotion,
            "dominant_label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "bg_gradient": meta["bg_gradient"],
            "confidence": round(confidence * 100, 1),
            "confidence_ratio": confidence,
            "sentiment": sentiment_label,
            "sentiment_score": compound_score,
            "intensity": {
                "level": intensity_level,
                "score": intensity_score,
            },
            "probabilities": [
                {
                    "emotion": k,
                    "label": EMOTION_META[k]["label"],
                    "emoji": EMOTION_META[k]["emoji"],
                    "color": EMOTION_META[k]["color"],
                    "probability": round(v * 100, 1),
                    "ratio": v,
                }
                for k, v in sorted_emotions
            ],
            "dimensions": self._compute_dimensions(probabilities),
            "tokens": token_attributions,
            "guidance": self._compute_guidance(dominant_emotion, lang_info["detected_language"]),
        }

    def _compute_guidance(self, emotion: str, lang_code: str = "en") -> Dict[str, str]:
        """Provides empathetic psychology context and suggested replies in the input language."""
        meta = EMOTION_META.get(emotion, EMOTION_META["neutral"])
        lang = (lang_code or "en").lower()

        if lang == "hi":
            hindi_guidance = {
                "joy": {"psychology": "संतुष्टि, आभार और खुशी को दर्शाता है।", "empathy_tip": "उनकी खुशी में शामिल हों और सराहना करें।", "suggested_reply": "यह तो बहुत ही बढ़िया खबर है! आपको ढेर सारी शुभकामनाएं! 🎉"},
                "sadness": {"psychology": "दुख, अकेलापन या निराशा की भावना।", "empathy_tip": "बिना किसी निर्णय के उनका साथ दें।", "suggested_reply": "यह सुनकर बहुत दुख हुआ। अगर किसी भी चीज़ की ज़रूरत हो तो मैं हमेशा आपके साथ हूँ।"},
                "anger": {"psychology": "अन्याय या उम्मीद टूटने से उत्पन्न रोष।", "empathy_tip": "माहौल को तुरंत शांत करें और बात समझें।", "suggested_reply": "मैं समझ सकता हूँ कि आप परेशान क्यों हैं। चलिए इसे तुरंत सुलझाते हैं।"},
                "fear": {"psychology": "भविष्य की अनिश्चितता या चिंता की आशंका।", "empathy_tip": "धैर्य और आश्वासन दें।", "suggested_reply": "घबराइए मत, गहरी सांस लीजिए। हम मिलकर इसे संभाल लेंगे।"},
                "surprise": {"psychology": "अप्रत्याशित घटना पर विस्मय की प्रतिक्रिया।", "empathy_tip": "बात को स्पष्ट करने में मदद करें।", "suggested_reply": "अरे वाह, यह तो वाकई अप्रत्याशित था! आप इसे कैसे संभाल रहे हैं?"},
                "love": {"psychology": "गहरा भावनात्मक जुड़ाव और स्नेह।", "empathy_tip": "सद्भावना और गर्मजोशी से उत्तर दें।", "suggested_reply": "आपके इस स्नेह के लिए बहुत-बहुत धन्यवाद, यह दिल को छू गया। ❤️"},
                "disgust": {"psychology": "अप्रिय या असहज स्थिति से दूरी बनाने की इच्छा।", "empathy_tip": "उनकी सीमा और भावनाओं का सम्मान करें।", "suggested_reply": "मैं आपकी इस परेशानी को समझता हूँ, आपकी सीमा का पूरा सम्मान है।"},
                "neutral": {"psychology": "तथ्यात्मक और स्पष्ट बातचीत।", "empathy_tip": "बातचीत को सीधा और उपयोगी रखें।", "suggested_reply": "जानकारी के लिए धन्यवाद, नोट कर लिया गया है।"},
            }
            if emotion in hindi_guidance:
                return hindi_guidance[emotion]

        if lang == "hinglish":
            hinglish_guidance = {
                "joy": {"psychology": "Khushi aur celebration ka mahaul hai.", "empathy_tip": "Unki khushi ko acknowledge karo aur celebrate karo.", "suggested_reply": "Aree waah, bohot badiya news hai! Party kab de rahe ho? Congratulations! 🎉"},
                "sadness": {"psychology": "Disappointment ya sadness ka signal hai.", "empathy_tip": "Supportive raho bina judge kiye.", "suggested_reply": "Sunke bura laga bro. Tension mat le, sab theek ho jayega, main hu na yahan."},
                "anger": {"psychology": "Frustration ya expectation break hone ka gussa.", "empathy_tip": "Calmly suno aur sort karne me help karo.", "suggested_reply": "Sahi me bohot irritating hai yeh. Gussa mat ho bro, shanti se baat karke sort karte hain."},
                "fear": {"psychology": "Anxiety aur tension ka feeling.", "empathy_tip": "Calmness aur safety ka assurance do.", "suggested_reply": "Chill kar bhai, itna darr mat. Hum milke sambhal lenge, tension mat le."},
                "surprise": {"psychology": "Unexpected news ya shocking revelation.", "empathy_tip": "Absorb karne ka time do.", "suggested_reply": "Arrey baap re! Yeh to bilkul unexpected tha! Ab kya plan hai?"},
                "love": {"psychology": "Emotional warmth aur care.", "empathy_tip": "Warmth reciprocate karo.", "suggested_reply": "Dil se shukriya bhai, aapki baat bohot achhi lagi ❤️"},
                "disgust": {"psychology": "Strong dislike ya avoidance.", "empathy_tip": "Boundaries respect karo.", "suggested_reply": "Totally understandable hai. Apni boundaries maintain karna zaroori hai."},
                "neutral": {"psychology": "Simple informational message.", "empathy_tip": "Direct aur clear raho.", "suggested_reply": "Theek hai bhai, samajh gaya. Update ke liye shukriya."},
            }
            if emotion in hinglish_guidance:
                return hinglish_guidance[emotion]

        if lang == "es":
            spanish_guidance = {
                "joy": {"psychology": "Refleja satisfacción y alegría.", "empathy_tip": "Celebra su logro genuinamente.", "suggested_reply": "¡Qué fantástica noticia! ¡Muchas felicidades por este gran logro! 🎉"},
                "sadness": {"psychology": "Indica tristeza o decepción.", "empathy_tip": "Ofrece compañía sin juzgar.", "suggested_reply": "Siento mucho por lo que estás pasando. Estoy aquí para lo que necesites."},
                "anger": {"psychology": "Surge por frustración o injusticia.", "empathy_tip": "Desescala la tensión de inmediato.", "suggested_reply": "Entiendo perfectamente tu frustración. Vamos a resolver esto de inmediato."},
                "fear": {"psychology": "Anticipación de incertidumbre o peligro.", "empathy_tip": "Brinda tranquilidad y apoyo.", "suggested_reply": "Respira hondo, todo saldrá bien. Vamos paso a paso, no estás solo."},
                "surprise": {"psychology": "Asombro ante lo inesperado.", "empathy_tip": "Ayuda a asimilar la novedad.", "suggested_reply": "¡Guau, eso no me lo esperaba para nada! ¿Cómo lo estás asimilando?"},
                "love": {"psychology": "Afecto y cariño recíproco.", "empathy_tip": "Responde con calidez.", "suggested_reply": "Tus palabras significan mucho para mí. ¡Muchas gracias por tu cariño! ❤️"},
                "disgust": {"psychology": "Rechazo hacia una situación.", "empathy_tip": "Respeta sus límites.", "suggested_reply": "Comprendo completamente tu rechazo. Establecer límites es lo correcto."},
                "neutral": {"psychology": "Mensaje puramente informativo.", "empathy_tip": "Mantén la claridad.", "suggested_reply": "Entendido y anotado. Muchas gracias por la información."},
            }
            if emotion in spanish_guidance:
                return spanish_guidance[emotion]

        return {
            "psychology": meta["psychology"],
            "empathy_tip": meta["empathy_tip"],
            "suggested_reply": meta["suggested_reply"],
        }

    def _apply_lexical_adjustments(self, text: str, probs: Dict[str, float]) -> Dict[str, float]:
        """Adjusts probabilities for clear cues, punctuation cues, and negations."""
        lowered = text.lower()
        adjusted = dict(probs)

        # 1. Contrast & Concession Handling (e.g., 'itna acha kaam kiya fir bhi dant padi', 'worked hard but failed')
        contrast_match = re.search(
            r"\b(but|however|although|though|yet|still|even though|nevertheless|despite|in spite of|lekin|magar|par|kintu|parantu|fir bhi|phir bhi|tab bhi|tabhi|bavajood|bawajood)\b",
            lowered
        )
        if contrast_match:
            clause_before = lowered[:contrast_match.start()]
            clause_after = lowered[contrast_match.end():]
            before_pos = bool(re.search(r"\b(accha|acha|achha|badhiya|good|great|well|hard work|mehnat|effort|love|happy|kaam kiya)\b", clause_before))
            after_neg = bool(re.search(r"\b(dant|daant|scold|fail|criticiz|hurt|ruin|terrible|bad|kharab|bekar|bura|upset|loss|lost)\b", clause_after))
            if before_pos and after_neg:
                if "joy" in adjusted:
                    adjusted["joy"] *= 0.05
                if "love" in adjusted:
                    adjusted["love"] *= 0.05
                adjusted["sadness"] = max(adjusted.get("sadness", 0) * 3.5, 0.52)
                adjusted["anger"] = max(adjusted.get("anger", 0) * 2.8, 0.36)
                if "neutral" in adjusted:
                    adjusted["neutral"] *= 0.15

        # Scolding / Rebuke / Unappreciated Hinglish idioms
        if re.search(r"\b(dant|daant)\s*(padi|pad\s*gayi|khayi|padti|padi\s*hai)\b|\b(fatkaar|fatkar)\b", lowered):
            if "joy" in adjusted:
                adjusted["joy"] *= 0.05
            adjusted["sadness"] = max(adjusted.get("sadness", 0) * 2.5, 0.48)
            adjusted["anger"] = max(adjusted.get("anger", 0) * 2.5, 0.42)
            if "neutral" in adjusted:
                adjusted["neutral"] *= 0.2

        # Betrayal / Heartbreak in Hinglish
        if re.search(r"\b(koi\s+(kadar|qadar)\s+na?hi|dhokha\s+(mila|diya)|dil\s+toot\s*(gaya|gya)|popat\s+ho\s*(gaya|gya))\b", lowered):
            if "joy" in adjusted:
                adjusted["joy"] *= 0.05
            adjusted["sadness"] = max(adjusted.get("sadness", 0) * 3.0, 0.55)
            adjusted["anger"] = max(adjusted.get("anger", 0) * 2.2, 0.35)
            if "neutral" in adjusted:
                adjusted["neutral"] *= 0.2

        # Check for phrase-level negations of positive traits (e.g., 'not that great', 'you are not that good', 'not impressive')
        has_negated_positive = bool(re.search(
            r"\b(not|nit|never|hardly|barely|rarely|ain'?t|aint|isn'?t|isnt|aren'?t|arent|wasn'?t|wasnt|weren'?t|werent|don'?t|dont|didn'?t|didnt)\s+(that\s+|so\s+|very\s+|really\s+|too\s+|all\s+that\s+|at\s+all\s+|quite\s+|much\s+)?(great|good|happy|glad|pleased|satisfied|fine|delighted|thrilled|excited|blessed|awesome|wonderful|special|cool|smart|impressive|amazing|fantastic|super|brilliant|fond|keen|accha|achha|badhiya|badiya|shandaar)\b|\b(khush\s+na?hi|na?hi\s+khush|accha\s+na?hi|na?hi\s+accha)\b",
            lowered
        ))
        is_personal_critique = bool(re.search(r"\b(you|u|he|she|they|it)\s+(are|r|is|ain'?t|aint|aren'?t|was)\s+(not|nit)\b|\b(huh|pfft|meh|whatever|ugh|bleh|nah)\b", lowered))

        if has_negated_positive:
            if "joy" in adjusted:
                adjusted["joy"] *= 0.05
            if "love" in adjusted:
                adjusted["love"] *= 0.05
            if "surprise" in adjusted:
                adjusted["surprise"] = min(adjusted["surprise"], 0.15)
            if is_personal_critique:
                adjusted["anger"] = max(adjusted.get("anger", 0) * 3.5, 0.60)
                adjusted["disgust"] = max(adjusted.get("disgust", 0) * 2.0, 0.35)
            else:
                adjusted["sadness"] = max(adjusted.get("sadness", 0) * 3.0, 0.55)
                adjusted["neutral"] = max(adjusted.get("neutral", 0) * 1.5, 0.35)

        # Check for dislike / aversion negations like 'dont like', "don't like", 'do not like', 'dont love'
        if re.search(r"\b(don'?t|dont|do not|didn'?t|didnt|did not|never|can'?t|cannot)\s+(like|love|adore|care for|enjoy|stand)\b|\b(pasand\s+na?hi|na?hi\s+pasand|nafrat)\b|dislike", lowered):
            if "joy" in adjusted:
                adjusted["joy"] *= 0.05
            if "love" in adjusted:
                adjusted["love"] *= 0.05
            adjusted["anger"] = max(adjusted.get("anger", 0) * 2.5, 0.50)
            adjusted["disgust"] = max(adjusted.get("disgust", 0) * 2.0, 0.35)

        # Sarcasm & Irony Detection (e.g. "Oh fantastic, my flight was delayed for 5 hours, what a joy")
        sarcastic_clues = bool(re.search(
            r"\b(oh\s+(great|fantastic|wonderful|brilliant|perfect|lovely|super|awesome|joy)|what a joy|just what i (needed|wanted)|thanks for (nothing|ruining)|love it when (my|the|it)|couldn'?t be better\b.*(broken|ruined|delayed|lost|crashed|failed)|pure joy\b.*(waiting|stuck|traffic|delay)|kya baat h\b.*(kharab|gussa|der|delay))\b",
            lowered
        ))
        has_positive_framing = bool(re.search(
            r"\b(oh\s+(great|fantastic|wonderful|brilliant|perfect|lovely|joy)|fantastic|wonderful|what a joy|brilliant|perfect|yay|so glad|love it)\b",
            lowered
        ))
        has_negative_situation = bool(re.search(
            r"\b(delay\w*|cancel\w*|broken|broke\w*|ruin\w*|fail\w*|stolen|crash\w*|headache|traffic|nightmare|disaster|stuck|waste of time|never again|worst|mess|lost|painful|sick|horrible|terrible)\b",
            lowered
        ))
        is_sarcastic = sarcastic_clues or (has_positive_framing and has_negative_situation)
        if is_sarcastic:
            if "joy" in adjusted:
                adjusted["joy"] *= 0.05
            if "love" in adjusted:
                adjusted["love"] *= 0.05
            if "surprise" in adjusted:
                adjusted["surprise"] = min(adjusted["surprise"], 0.15)
            adjusted["anger"] = max(adjusted.get("anger", 0) * 3.5, 0.65)
            adjusted["disgust"] = max(adjusted.get("disgust", 0) * 2.0, 0.35)

        # Litotes / Understatements: 'not bad', 'not that bad', 'not half bad', 'not terrible'
        if re.search(r"\b(not\s+(that\s+|too\s+|half\s+)?(bad|terrible|awful|horrible|worst)|bura\s+na?hi)\b", lowered):
            if "disgust" in adjusted:
                adjusted["disgust"] *= 0.05
            if "sadness" in adjusted:
                adjusted["sadness"] *= 0.05
            if "anger" in adjusted:
                adjusted["anger"] *= 0.05
            adjusted["neutral"] = max(adjusted.get("neutral", 0) * 2.0, 0.50)
            adjusted["joy"] = max(adjusted.get("joy", 0) * 1.5, 0.35)

        # Quick emotional slang boosts
        if re.search(r"\b(omg|wtf|holy cow|no way|unbelievable)\b", lowered):
            adjusted["surprise"] = max(adjusted.get("surprise", 0) * 1.8, 0.35)
        if re.search(r"\b(livid|furious|pissed|shut up|hate you|scam|angery|never flying)\b", lowered):
            adjusted["anger"] = max(adjusted.get("anger", 0) * 2.2, 0.55)
        if re.search(r"\b(i love you|my soulmate|marry me|adore you)\b", lowered):
            adjusted["love"] = max(adjusted.get("love", 0) * 2.0, 0.45)
        if re.search(r"\b(gross|disgusting|vomit|puke|nauseating|repulsive)\b", lowered):
            adjusted["disgust"] = max(adjusted.get("disgust", 0) * 2.0, 0.45)
        if re.search(r"\b(terrified|panic attack|frightened|petrified)\b", lowered):
            adjusted["fear"] = max(adjusted.get("fear", 0) * 2.0, 0.45)
        if any(emoji in text for emoji in ["🎉", "🥳", "🎊", "🥂"]) or re.search(r"\b(acceptance|accepted|screamed with joy|screamed in (?:the )?library|we won)\b", lowered):
            adjusted["joy"] = max(adjusted.get("joy", 0) * 2.5, 0.65)
            if "anger" in adjusted:
                adjusted["anger"] *= 0.15

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in adjusted.items()}
        return probs

    def _compute_intensity(self, text: str, confidence: float) -> Tuple[str, int]:
        """Calculates emotional intensity from exclamation marks, capital letters, and confidence."""
        score = int(confidence * 65)

        # Boost for exclamation marks
        exclamations = text.count("!")
        if exclamations > 0:
            score += min(exclamations * 8, 20)

        # Boost for capital words
        words = text.split()
        caps = [w for w in words if len(w) > 1 and w.isupper() and w.isalpha()]
        if caps:
            score += min(len(caps) * 6, 18)

        score = max(5, min(100, score))

        if score >= 75:
            level = "Intense / High"
        elif score >= 45:
            level = "Moderate"
        else:
            level = "Subtle / Low"

        return level, score

    def _compute_sentiment(self, probs: Dict[str, float], dominant: str) -> Tuple[str, float]:
        """Computes continuous sentiment polarity score from -1.0 (strongly negative) to +1.0 (strongly positive)."""
        compound = 0.0
        for emotion, prob in probs.items():
            meta = EMOTION_META.get(emotion)
            if meta:
                compound += meta["sentiment_val"] * prob

        compound = round(max(-1.0, min(1.0, compound)), 3)

        if dominant in ["joy", "love"]:
            label = "Positive"
        elif dominant in ["sadness", "anger", "fear", "disgust"]:
            label = "Negative"
        elif dominant == "surprise":
            label = "Positive" if compound >= 0.0 else "Negative"
        else:
            label = "Neutral"

        return label, compound

    def _extract_token_attributions(self, text: str, dominant: str) -> List[Dict[str, Any]]:
        """Breaks down text into tokens with emotional contribution weights for visual highlighting."""
        # Split tokens preserving punctuation
        tokens_raw = re.findall(r"\b[\w'-]+\b|[^\w\s]", text)
        result = []

        # Gather target keywords
        cues_for_dominant = set(KEYWORD_CUES.get(dominant, []))
        all_cues = {}
        for emo, words in KEYWORD_CUES.items():
            for w in words:
                all_cues[w] = emo

        is_sarcastic_text = dominant in ("anger", "disgust") and bool(re.search(
            r"\b(oh\s+(great|fantastic|wonderful|brilliant|perfect|lovely|super|awesome|joy)|what a joy|just what i (needed|wanted)|thanks for (nothing|ruining)|love it when (my|the|it)|fantastic|wonderful)\b",
            text.lower()
        ))
        sarcastic_words = {"fantastic", "joy", "great", "wonderful", "perfect", "lovely", "awesome", "super", "brilliant"}

        has_negated_positive = bool(re.search(
            r"\b(not|nit|never|hardly|barely|rarely|ain'?t|aint|isn'?t|isnt|aren'?t|arent|wasn'?t|wasnt|weren'?t|werent|don'?t|dont|didn'?t|didnt)\s+(that\s+|so\s+|very\s+|really\s+|too\s+|all\s+that\s+|at\s+all\s+|quite\s+|much\s+)?(great|good|happy|glad|pleased|satisfied|fine|delighted|thrilled|excited|blessed|awesome|wonderful|special|cool|smart|impressive|amazing|fantastic|super|brilliant|fond|keen)\b",
            text.lower()
        ))
        negated_positive_words = {"great", "good", "happy", "glad", "pleased", "satisfied", "fine", "delighted", "thrilled", "excited", "blessed", "awesome", "wonderful", "special", "cool", "smart", "impressive", "amazing", "fantastic", "super", "brilliant", "fond", "keen", "not", "nit", "that", "huh"}

        for token in tokens_raw:
            cleaned = token.lower().strip()
            is_cue = False
            associated_emo = dominant
            weight = 0.0

            if (is_sarcastic_text and cleaned in sarcastic_words) or (has_negated_positive and cleaned in negated_positive_words):
                is_cue = True
                associated_emo = dominant
                weight = 0.85
            elif cleaned in cues_for_dominant:
                is_cue = True
                associated_emo = dominant
                weight = 0.85
            elif cleaned in all_cues:
                is_cue = True
                associated_emo = all_cues[cleaned]
                weight = 0.65
            elif len(cleaned) > 4 and any(cleaned.startswith(c[:4]) for c in cues_for_dominant if len(c) > 4):
                is_cue = True
                associated_emo = dominant
                weight = 0.55

            token_data = {
                "word": token,
                "is_emotional": is_cue,
                "emotion": associated_emo if is_cue else "neutral",
                "color": EMOTION_META.get(associated_emo, {}).get("color", "#94A3B8") if is_cue else None,
                "weight": weight,
            }
            result.append(token_data)

        return result

    def _compute_dimensions(self, probs: Dict[str, float]) -> Dict[str, Any]:
        """Calculates Valence, Arousal, Dominance (VAD) and Subjectivity dimensions based on affective computing standards."""
        vad_table = {
            "joy": (0.85, 0.72, 0.75),
            "love": (0.92, 0.65, 0.68),
            "surprise": (0.20, 0.82, 0.45),
            "neutral": (0.00, 0.15, 0.50),
            "sadness": (-0.80, 0.25, 0.20),
            "fear": (-0.82, 0.85, 0.15),
            "anger": (-0.75, 0.88, 0.70),
            "disgust": (-0.80, 0.60, 0.55),
        }
        val = sum(probs.get(e, 0.0) * vad_table[e][0] for e in vad_table)
        aro = sum(probs.get(e, 0.0) * vad_table[e][1] for e in vad_table)
        dom = sum(probs.get(e, 0.0) * vad_table[e][2] for e in vad_table)
        
        subj = 1.0 - probs.get("neutral", 0.0)
        
        val_clamped = round(max(-1.0, min(1.0, val)), 3)
        aro_pct = int(round(max(0.0, min(1.0, aro)) * 100))
        dom_pct = int(round(max(0.0, min(1.0, dom)) * 100))
        subj_pct = int(round(max(0.0, min(1.0, subj)) * 100))
        
        if aro_pct >= 70:
            aro_label = "High Surge"
        elif aro_pct >= 40:
            aro_label = "Moderate Activation"
        else:
            aro_label = "Calm / Low"

        if dom_pct >= 65:
            dom_label = "Assertive / Empowered"
        elif dom_pct >= 40:
            dom_label = "Balanced Agency"
        else:
            dom_label = "Submissive / Helpless"

        return {
            "valence": val_clamped,
            "arousal": aro_pct,
            "arousal_label": aro_label,
            "dominance": dom_pct,
            "dominance_label": dom_label,
            "subjectivity": subj_pct,
            "subjectivity_label": "High Emotion" if subj_pct >= 65 else ("Balanced" if subj_pct >= 35 else "Objective Fact"),
        }

    def _empty_response(self) -> Dict[str, Any]:
        meta = EMOTION_META["neutral"]
        return {
            "text": "",
            "detected_language": "en",
            "language_name": "English",
            "is_multilingual": False,
            "translated_text": None,
            "dominant_emotion": "neutral",
            "dominant_label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "bg_gradient": meta["bg_gradient"],
            "confidence": 0.0,
            "confidence_ratio": 0.0,
            "sentiment": "Neutral",
            "sentiment_score": 0.0,
            "intensity": {"level": "Subtle / Low", "score": 0},
            "probabilities": [
                {
                    "emotion": k,
                    "label": v["label"],
                    "emoji": v["emoji"],
                    "color": v["color"],
                    "probability": 12.5,
                    "ratio": 0.125,
                }
                for k, v in EMOTION_META.items()
            ],
            "dimensions": {
                "valence": 0.0,
                "arousal": 15,
                "arousal_label": "Calm / Low",
                "dominance": 50,
                "dominance_label": "Balanced Agency",
                "subjectivity": 0,
                "subjectivity_label": "Objective Fact",
            },
            "tokens": [],
            "guidance": self._compute_guidance("neutral", "en"),
        }

    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Runs batch inference and aggregates emotion distribution metrics."""
        results = [self.predict(t) for t in texts if t and t.strip()]
        if not results:
            return {"results": [], "summary": {}}

        # Aggregate counts & distributions
        emotion_counts: Dict[str, int] = {k: 0 for k in EMOTION_META.keys()}
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        total_intensity = 0

        for res in results:
            emo = res["dominant_emotion"]
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            sentiment_counts[res["sentiment"]] = sentiment_counts.get(res["sentiment"], 0) + 1
            total_intensity += res["intensity"]["score"]

        total_samples = len(results)
        distribution = [
            {
                "emotion": k,
                "label": EMOTION_META[k]["label"],
                "emoji": EMOTION_META[k]["emoji"],
                "color": EMOTION_META[k]["color"],
                "count": count,
                "percentage": round((count / total_samples) * 100, 1),
            }
            for k, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        top_emotion = distribution[0]["emotion"] if distribution else "neutral"

        summary = {
            "total_analyzed": total_samples,
            "top_emotion": top_emotion,
            "top_emotion_label": EMOTION_META[top_emotion]["label"],
            "top_emotion_emoji": EMOTION_META[top_emotion]["emoji"],
            "average_intensity": round(total_intensity / total_samples, 1),
            "sentiment_ratio": {
                k: round((v / total_samples) * 100, 1) for k, v in sentiment_counts.items()
            },
            "distribution": distribution,
        }

        return {
            "results": results,
            "summary": summary,
        }


# Quick test
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    classifier = EmotionClassifier()
    test_phrases = [
        "I just won the gold medal and couldn't be happier!",
        "My heart hurts and I feel so hopeless and alone.",
        "How dare you lie to my face after everything I did for you?!",
        "I was trembling with fear as the siren went off in the pitch black.",
        "Holy cow, I never expected to find a treasure chest in the attic!",
        "I love you more than life itself, you are my everything.",
        "The food was covered in green slime and smelled putrid, absolutely repulsive.",
        "The conference call is scheduled for 3:00 PM on Tuesday afternoon.",
    ]
    for phrase in test_phrases:
        res = classifier.predict(phrase)
        print(f"[{res['emoji']} {res['dominant_emotion'].upper():<8}] ({res['confidence']}%) {res['text'][:65]}...")
