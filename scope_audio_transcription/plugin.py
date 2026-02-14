"""
Audio Transcription Plugin for Daydream Scope

Captures audio, transcribes with Whisper, extracts keywords,
and maps them to Scope generation parameters.
"""

import asyncio
import queue
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import whisper

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
except ImportError:
    raise ImportError("pip install nltk")

try:
    import spacy
except ImportError:
    raise ImportError("pip install spacy && python -m spacy download en_core_web_sm")


class AudioTranscriptionPlugin:
    """
    Real-time audio transcription plugin for Scope.
    
    Transcribes audio input and extracts:
    - Full phrases (for prompt injection)
    - Concrete nouns (for keyword-based control)
    - Sentiment/emotion (for parameter modulation)
    """
    
    def __init__(self, model_size: str = "base.en"):
        """
        Initialize the transcription plugin.
        
        Args:
            model_size: Whisper model size ("tiny.en", "base.en", "small.en", "medium.en")
        """
        self.model_size = model_size
        self.model = None
        self.nlp = None
        self.stop_words = set()
        
        # Audio buffer settings
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # seconds
        self.audio_queue = queue.LifoQueue(maxsize=5)
        
        # Transcription outputs
        self.last_phrase = ""
        self.last_keyword = ""
        self.confidence = 0.0
        
        # NLP weights for concrete noun extraction
        self.lex_weights = {
            "noun.artifact": 1.4,
            "noun.object": 1.4,
            "noun.natural_object": 1.2,
            "noun.food": 1.2,
            "noun.animal": 1.2,
            "noun.plant": 1.0,
            "noun.body": 1.0,
            "noun.substance": 1.0,
            "noun.location": 0.9,
            "noun.person": 0.6,
        }
        
    def setup(self):
        """Initialize models and resources."""
        print("[Scope Audio Transcription] Loading Whisper model...")
        self.model = whisper.load_model(self.model_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            
        print("[Scope Audio Transcription] Loading NLP models...")
        self._download_nltk_resources()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words("english"))
        
        print("[Scope Audio Transcription] Ready!")
        
    def _download_nltk_resources(self):
        """Download required NLTK data."""
        for pkg in ["wordnet", "omw-1.4", "stopwords"]:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
                
    def _wn_concreteness(self, lemma: str) -> float:
        """Calculate concreteness score using WordNet."""
        best = 0.0
        for syn in wn.synsets(lemma, pos=wn.NOUN):
            w = self.lex_weights.get(syn.lexname(), 0.0)
            if w > best:
                best = w
        return best
        
    def _extract_concrete_noun(self, text: str) -> str:
        """
        Extract the most concrete noun phrase from text.
        
        Prioritizes physical objects, animals, food over abstract concepts.
        """
        doc = self.nlp(text)
        candidates = []
        
        for chunk in doc.noun_chunks:
            tokens = [t for t in chunk if t.pos_ in {"NOUN", "PROPN", "ADJ"}]
            if not tokens:
                continue
                
            phrase = " ".join(t.text for t in tokens).strip()
            if not phrase:
                continue
                
            # Score based on concreteness
            nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"}]
            score = sum(self._wn_concreteness(t.lemma_.lower()) for t in nouns)
            candidates.append((score, phrase))
            
        if not candidates:
            return ""
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
        
    def process_audio(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process an audio chunk and return transcription results.
        
        Args:
            audio_chunk: Audio data as numpy array (16kHz mono)
            
        Returns:
            Dictionary with:
                - full_phrase: Complete transcription
                - keyword: Extracted concrete noun
                - confidence: Transcription confidence
                - word_count: Number of words
        """
        # Add to queue (LIFO keeps most recent)
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            # Queue full, drop oldest
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_chunk)
            except:
                pass
                
        # Get most recent chunk
        try:
            audio = self.audio_queue.get_nowait()
        except queue.Empty:
            return {
                "full_phrase": self.last_phrase,
                "keyword": self.last_keyword,
                "confidence": self.confidence,
                "word_count": len(self.last_phrase.split())
            }
            
        # Transcribe
        audio_float = audio.astype(np.float32) / 32768.0
        result = self.model.transcribe(
            audio_float,
            language="en",
            fp16=torch.cuda.is_available()
        )
        
        text = result["text"].strip()
        if not text:
            return {
                "full_phrase": self.last_phrase,
                "keyword": self.last_keyword,
                "confidence": self.confidence,
                "word_count": 0
            }
            
        # Extract keyword
        keyword = self._extract_concrete_noun(text)
        
        # Calculate confidence (average of segment confidences)
        segments = result.get("segments", [])
        if segments:
            confidences = [seg.get("no_speech_prob", 1.0) for seg in segments]
            confidence = 1.0 - (sum(confidences) / len(confidences))
        else:
            confidence = 0.5
            
        # Store results
        self.last_phrase = text
        self.last_keyword = keyword if keyword else self.last_keyword
        self.confidence = confidence
        
        return {
            "full_phrase": text,
            "keyword": keyword,
            "confidence": confidence,
            "word_count": len(text.split())
        }
        
    def get_scope_parameters(self, transcription: Dict[str, Any]) -> Dict[str, float]:
        """
        Map transcription results to Scope generation parameters.
        
        Args:
            transcription: Output from process_audio()
            
        Returns:
            Dictionary of parameter name -> value mappings
        """
        params = {}
        
        # Word count affects generation intensity
        word_count = transcription["word_count"]
        if word_count > 0:
            # More words = more intense generation
            params["cfg_scale"] = min(15.0, 7.0 + (word_count * 0.5))
            
        # Confidence affects strength
        confidence = transcription["confidence"]
        params["strength"] = 0.3 + (confidence * 0.4)  # 0.3 to 0.7
        
        # Keyword presence affects variation
        if transcription["keyword"]:
            params["seed_variation"] = 0.8  # High variation with keywords
        else:
            params["seed_variation"] = 0.2  # Low variation without
            
        return params
        
    def get_prompt_injection(self, transcription: Dict[str, Any]) -> Optional[str]:
        """
        Get prompt text to inject into generation.
        
        Args:
            transcription: Output from process_audio()
            
        Returns:
            Prompt string or None
        """
        keyword = transcription["keyword"]
        if keyword and len(keyword) > 2:
            return f"{keyword}, cinematic, dramatic lighting"
        return None
        
    def cleanup(self):
        """Clean up resources."""
        if self.model:
            del self.model
        if self.nlp:
            del self.nlp
        torch.cuda.empty_cache()


# Scope Plugin Interface
def create_plugin(config: Dict[str, Any]) -> AudioTranscriptionPlugin:
    """
    Factory function for Scope to instantiate the plugin.
    
    Args:
        config: Plugin configuration from Scope
        
    Returns:
        AudioTranscriptionPlugin instance
    """
    model_size = config.get("model_size", "base.en")
    plugin = AudioTranscriptionPlugin(model_size=model_size)
    plugin.setup()
    return plugin
