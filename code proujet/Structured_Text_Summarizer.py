from transformers import pipeline, AutoTokenizer
from langdetect import detect, DetectorFactory
import logging
from typing import Optional, List, Dict
import torch
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure language detection
DetectorFactory.seed = 0

# Try to import Arabic preprocessor if available
try:
    from arabert.preprocess import ArabertPreprocessor
    arabic_preprocessor = ArabertPreprocessor(model_name="aubmindlab/aragpt2-base")
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    logging.warning("ArabertPreprocessor not available. Arabic preprocessing will be limited.")


class StructuredTextSummarizer:
    def __init__(self):
        """Initialize with reliable public models."""
        self.models = {
            'en': {'model': 'facebook/bart-large-cnn', 'pipeline': None},
            'fr': {'model': 'moussaKam/barthez-orangesum-abstract', 'pipeline': None},
            'ar': {'model': 'csebuetnlp/mT5_multilingual_XLSum', 'pipeline': None}
        }
        self.tokenizers = {}
        self.max_input_length = 512  # Reduced for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self._initialize_models()
        
        # Translation for structured parts labels
        self.section_labels = {
            'en': {
                'intro': 'Introduction',
                'key_points': 'Key Points',
                'conclusion': 'Conclusion'
            },
            'fr': {
                'intro': 'Introduction',
                'key_points': 'Points Clés',
                'conclusion': 'Conclusion'
            },
            'ar': {
                'intro': 'مقدمة',
                'key_points': 'النقاط الرئيسية',
                'conclusion': 'خاتمة'
            }
        }

    def _initialize_models(self):
        """Load only English model by default to save memory."""
        try:
            # Only load English initially
            lang = 'en'
            config = self.models[lang]
            config['pipeline'] = pipeline(
                "summarization",
                model=config['model'],
                tokenizer=config['model'],
                device=self.device
            )
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(config['model'])
            logging.info(f"Model {lang} loaded successfully")
        except Exception as e:
            logging.error(f"Loading error: {str(e)}")
            raise

    def _preprocess_arabic(self, text: str) -> str:
        """Special cleaning for Arabic."""
        if ARABIC_SUPPORT:
            return arabic_preprocessor.preprocess(text)
        else:
            # Basic Arabic preprocessing if ArabertPreprocessor is not available
            return text.strip()

    def detect_language(self, text: str) -> str:
        """Simplified language detection."""
        try:
            lang = detect(text)
            return lang if lang in ['fr', 'en', 'ar'] else 'en'
        except:
            return 'en'

    def load_language_model(self, lang: str):
        """Load an additional model on demand with robust error handling."""
        if lang not in self.models or self.models[lang]['pipeline'] is not None:
            return

        try:
            config = self.models[lang]
            config['pipeline'] = pipeline(
                "summarization",
                model=config['model'],
                tokenizer=config['model'],
                device=self.device
            )
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(config['model'])
            logging.info(f"Model {lang} loaded successfully")
        except Exception as e:
            logging.error(f"Unable to load model {lang}: {str(e)}")
            # If Arabic model fails, try fallback to another Arabic model
            if lang == 'ar':
                try:
                    fallback_model = "google/mt5-small"  # Verified working model
                    logging.info(f"Trying fallback Arabic model: {fallback_model}")
                    self.models[lang]['model'] = fallback_model
                    self.models[lang]['pipeline'] = pipeline(
                        "summarization",
                        model=fallback_model,
                        tokenizer=fallback_model,
                        device=self.device
                    )
                    self.tokenizers[lang] = AutoTokenizer.from_pretrained(fallback_model)
                    logging.info(f"Fallback model for {lang} loaded successfully")
                except Exception as e2:
                    logging.error(f"Fallback model also failed: {str(e2)}")
                    self.models[lang] = None
            else:
                self.models[lang] = None

    def _extract_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract potential sections from text."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Find natural sections if possible
        section_dict = {
            'intro': [],
            'body': [],
            'conclusion': []
        }
        
        # Simple heuristics for section identification
        if len(paragraphs) >= 3:
            section_dict['intro'] = [paragraphs[0]]
            section_dict['conclusion'] = [paragraphs[-1]]
            section_dict['body'] = paragraphs[1:-1]
        elif len(paragraphs) == 2:
            section_dict['intro'] = [paragraphs[0]]
            section_dict['body'] = []
            section_dict['conclusion'] = [paragraphs[-1]]
        else:
            # If only one paragraph, estimate intro and conclusion
            text_sentences = re.split(r'[.!?]', text)
            text_sentences = [s.strip() for s in text_sentences if s.strip()]
            
            if len(text_sentences) >= 3:
                section_dict['intro'] = [text_sentences[0] + '.']
                section_dict['conclusion'] = [text_sentences[-1] + '.']
                middle_sentences = ' '.join(text_sentences[1:-1]) + '.'
                section_dict['body'] = [middle_sentences]
            elif len(text_sentences) == 2:
                section_dict['intro'] = [text_sentences[0] + '.']
                section_dict['conclusion'] = [text_sentences[1] + '.']
                section_dict['body'] = []
            else:
                section_dict['intro'] = [text]
                section_dict['body'] = []
                section_dict['conclusion'] = []
                
        return section_dict

    def _extract_key_points(self, body_text: str, lang: str, max_points: int = 3) -> List[str]:
        """Extract key points from the body text."""
        if not body_text.strip():
            return []
            
        # Simple extraction of key sentences
        sentences = [s.strip() + '.' for s in re.split(r'[.!?]', body_text) if s.strip()]
        
        # If very few sentences, return all of them
        if len(sentences) <= max_points:
            return sentences
            
        # Try to identify important sentences (simplified TF-IDF approach)
        # Count word frequency 
        all_text = ' '.join(sentences).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Score sentences based on important word frequency
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sent_words = re.findall(r'\b\w+\b', sentence.lower())
            for word in sent_words:
                if word in word_freq:
                    score += word_freq[word]
            # Normalize by sentence length to avoid favoring long sentences
            normalized_score = score / (len(sent_words) + 1)  
            scored_sentences.append((normalized_score, sentence))
            
        # Get top sentences
        scored_sentences.sort(reverse=True)
        key_points = [s[1] for s in scored_sentences[:max_points]]
        
        # Sort key points back to original order
        original_order = {}
        for i, sentence in enumerate(sentences):
            original_order[sentence] = i
        
        key_points.sort(key=lambda s: original_order.get(s, 0))
        
        return key_points

    def summarize_structured(self, text: str, summary_length: str = "medium", lang: Optional[str] = None) -> Dict[str, str]:
        """Generate a structured summary with introduction, key points, and conclusion."""
        if not text.strip():
            return {
                "intro": "",
                "key_points": [],
                "conclusion": "",
                "full_text": ""
            }

        # Determine language if not specified
        detected_lang = self.detect_language(text)
        lang = lang or detected_lang
        logging.info(f"Text language detected as: {detected_lang}, requested: {lang}")

        # If language not supported, fallback to English
        if lang not in self.models:
            logging.warning(f"Language {lang} not supported, falling back to English")
            lang = 'en'

        # Preprocessing based on language
        if lang == 'ar':
            text = self._preprocess_arabic(text)

        # First get a conventional summary to work with
        standard_summary = self.summarize_text(text, summary_length, lang)
        
        # Extract structured sections from the summary
        sections = self._extract_sections(standard_summary)
        
        # Extract key points from the body content
        body_text = ' '.join(sections['body'])
        key_points = self._extract_key_points(body_text, lang)
        
        # If we couldn't extract meaningful key points from the summary,
        # try to extract them directly from the original text
        if not key_points and len(text.split()) > 50:
            key_points = self._extract_key_points(text, lang)
        
        # Create structured result with localized section headings
        labels = self.section_labels.get(lang, self.section_labels['en'])
        
        return {
            "intro": ' '.join(sections['intro']),
            "key_points": key_points,
            "conclusion": ' '.join(sections['conclusion']),
            "full_text": standard_summary
        }

    def summarize_text(self, text: str, summary_length: str = "medium", lang: Optional[str] = None) -> str:
        """Robust version with complete error handling."""
        if not text.strip():
            return ""

        # Determine language if not specified
        detected_lang = self.detect_language(text)
        lang = lang or detected_lang

        # Preprocessing based on language
        if lang == 'ar':
            text = self._preprocess_arabic(text)

        # Try to load the appropriate language model
        self.load_language_model(lang)

        # Fall back to English if the requested language model is unavailable
        if lang not in self.models or self.models[lang] is None:
            logging.warning(f"Model for {lang} unavailable, falling back to English")
            lang = 'en'
            # Ensure English model is loaded
            self.load_language_model(lang)

        if self.models[lang]['pipeline'] is None:
            logging.error(f"Failed to load any model for {lang}, returning extract")
            # Basic fallback when all models fail
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')

        # Adaptive parameters
        word_count = len(text.split())
        length_settings = {
            "short": {
                "min_length": max(10, word_count//20),  # At least 10 words
                "max_length": max(30, word_count//10)   # At least 30 words
            },
            "medium": {
                "min_length": max(20, word_count//15),
                "max_length": max(60, word_count//7)
            },
            "long": {
                "min_length": max(30, word_count//10),
                "max_length": max(100, word_count//5)
            }
        }
        params = length_settings.get(summary_length, length_settings["medium"])

        # Dynamic adjustment of max_length if text is too short
        tokenizer = self.tokenizers[lang]
        input_tokens = tokenizer.encode(text, truncation=True, max_length=self.max_input_length)
        input_length = len(input_tokens)

        if input_length < params['max_length']:
            params['max_length'] = max(params['min_length'] + 10, input_length - 1)

        try:
            # Direct summarization without chunking to simplify
            result = self.models[lang]['pipeline'](
                text,
                **params,
                do_sample=False
            )
            summary = result[0]['summary_text']
            return self._postprocess_summary(summary)

        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            # Basic fallback
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')

    def _postprocess_summary(self, summary: str) -> str:
        """Clean the final summary."""
        summary = ' '.join(summary.split()).strip()
        if summary and summary[-1] not in {'.', '!', '?'}:
            summary += '.'
        return summary.capitalize()

    def format_structured_summary(self, structured_summary: Dict[str, str], lang: str = 'en') -> str:
        """Format the structured summary for display."""
        labels = self.section_labels.get(lang, self.section_labels['en'])
        
        result = []
        
        # Add introduction
        if structured_summary["intro"]:
            result.append(f"**{labels['intro']}**")
            result.append(structured_summary["intro"])
            result.append("")
        
        # Add key points
        if structured_summary["key_points"]:
            result.append(f"**{labels['key_points']}**")
            for i, point in enumerate(structured_summary["key_points"], 1):
                result.append(f"{i}. {point}")
            result.append("")
        
        # Add conclusion
        if structured_summary["conclusion"]:
            result.append(f"**{labels['conclusion']}**")
            result.append(structured_summary["conclusion"])
        
        return "\n".join(result)
