"""Audio Transcription pipeline for Daydream Scope.

Captures microphone audio, transcribes with Whisper AI, extracts
concrete nouns using spaCy + WordNet, and maps speech features to
Scope generation parameters.

In text mode: generates a debug visualization showing transcription state.
In video/preprocessor mode: passes through video while injecting prompts.

Author: Krista Faist
Based on The Mirror's Echo interactive installation.
"""

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import AudioTranscriptionConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class AudioTranscriptionPipeline(Pipeline):
    """Voice-controlled pipeline that transcribes speech and maps it to parameters.

    Uses Whisper AI for transcription, spaCy + WordNet for keyword extraction,
    and maps voice features to Scope generation parameters.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioTranscriptionConfig

    def __init__(
        self,
        model_size: str = "base.en",
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.model_size = model_size
        self.height = height
        self.width = width
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Transcription state
        self.last_phrase = ""
        self.last_keyword = ""
        self.confidence = 0.0

        # Audio capture
        self.sample_rate = 16000
        self._audio_queue: queue.LifoQueue = queue.LifoQueue(maxsize=5)
        self._stream = None
        self._audio_thread = None
        self._running = False
        self._audio_buffer = None
        self._buffer_index = 0

        # Models (loaded lazily)
        self._whisper_model = None
        self._nlp = None
        self._stop_words: set = set()

        # NLP weights for concrete noun extraction
        self._lex_weights = {
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

        # Load models
        self._load_models()

        # Start audio capture
        self._start_audio_capture()

    # ---- Model loading ----

    def _load_models(self):
        """Load Whisper, spaCy, and NLTK resources."""
        import whisper

        logger.info(f"Loading Whisper model: {self.model_size}")
        self._whisper_model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper model loaded")

        try:
            import nltk
            for pkg in ("wordnet", "omw-1.4", "stopwords"):
                try:
                    nltk.data.find(f"corpora/{pkg}")
                except LookupError:
                    nltk.download(pkg, quiet=True)
            from nltk.corpus import stopwords
            self._stop_words = set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"NLTK setup issue (non-fatal): {e}")

        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            self._nlp = None

    # ---- Audio capture ----

    def _start_audio_capture(self):
        """Start background microphone capture thread."""
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            logger.warning(f"sounddevice not available: {e}. Audio capture disabled.")
            return

        chunk_samples = int(self.sample_rate * 3.0)
        self._audio_buffer = np.zeros(chunk_samples, dtype=np.int16)
        self._buffer_index = 0
        self._running = True

        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")

            audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
            remaining = chunk_samples - self._buffer_index

            if len(audio_chunk) <= remaining:
                self._audio_buffer[self._buffer_index:self._buffer_index + len(audio_chunk)] = audio_chunk
                self._buffer_index += len(audio_chunk)
            else:
                self._audio_buffer[self._buffer_index:] = audio_chunk[:remaining]
                # Buffer full — push to queue
                try:
                    self._audio_queue.put_nowait(self._audio_buffer.copy())
                except queue.Full:
                    try:
                        self._audio_queue.get_nowait()
                        self._audio_queue.put_nowait(self._audio_buffer.copy())
                    except queue.Empty:
                        pass
                self._buffer_index = 0
                self._audio_buffer[:] = 0

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=_audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            logger.info("Audio capture started")
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self._stream = None

    # ---- NLP / Keyword extraction ----

    def _wn_concreteness(self, lemma: str) -> float:
        """Calculate concreteness score using WordNet."""
        from nltk.corpus import wordnet as wn

        best = 0.0
        for syn in wn.synsets(lemma, pos=wn.NOUN):
            w = self._lex_weights.get(syn.lexname(), 0.0)
            if w > best:
                best = w
        return best

    def _extract_concrete_noun(self, text: str) -> str:
        """Extract the most concrete noun phrase from text.

        Prioritizes physical objects, animals, food over abstract concepts.
        """
        if self._nlp is None:
            # Fallback: return first capitalized word or longest word
            words = text.split()
            return max(words, key=len) if words else ""

        doc = self._nlp(text)
        candidates = []

        for chunk in doc.noun_chunks:
            tokens = [t for t in chunk if t.pos_ in {"NOUN", "PROPN", "ADJ"}]
            if not tokens:
                continue
            phrase = " ".join(t.text for t in tokens).strip()
            if not phrase:
                continue
            nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"}]
            score = sum(self._wn_concreteness(t.lemma_.lower()) for t in nouns)
            candidates.append((score, phrase))

        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ---- Transcription ----

    def _process_audio_queue(self, confidence_threshold: float) -> dict:
        """Process the latest audio chunk from the queue.

        Returns dict with full_phrase, keyword, confidence, word_count.
        """
        try:
            audio = self._audio_queue.get_nowait()
        except queue.Empty:
            return {
                "full_phrase": self.last_phrase,
                "keyword": self.last_keyword,
                "confidence": self.confidence,
                "word_count": len(self.last_phrase.split()) if self.last_phrase else 0,
            }

        # Transcribe with Whisper
        audio_float = audio.astype(np.float32) / 32768.0
        result = self._whisper_model.transcribe(
            audio_float,
            language="en",
            fp16=(self.device.type == "cuda"),
        )

        text = result["text"].strip()
        if not text:
            return {
                "full_phrase": self.last_phrase,
                "keyword": self.last_keyword,
                "confidence": self.confidence,
                "word_count": 0,
            }

        # Calculate confidence
        segments = result.get("segments", [])
        if segments:
            probs = [seg.get("no_speech_prob", 1.0) for seg in segments]
            confidence = 1.0 - (sum(probs) / len(probs))
        else:
            confidence = 0.5

        # Apply threshold
        if confidence < confidence_threshold:
            return {
                "full_phrase": self.last_phrase,
                "keyword": self.last_keyword,
                "confidence": confidence,
                "word_count": 0,
            }

        # Extract keyword
        keyword = self._extract_concrete_noun(text)

        # Update state
        self.last_phrase = text
        if keyword:
            self.last_keyword = keyword
        self.confidence = confidence

        return {
            "full_phrase": text,
            "keyword": keyword or self.last_keyword,
            "confidence": confidence,
            "word_count": len(text.split()),
        }

    # ---- Parameter mapping ----

    def _map_to_scope_params(self, transcription: dict, keyword_weight: float) -> dict:
        """Map transcription results to Scope generation parameters."""
        params = {}

        word_count = transcription["word_count"]
        if word_count > 0:
            params["cfg_scale"] = min(15.0, 7.0 + (word_count * 0.5))

        confidence = transcription["confidence"]
        params["strength"] = 0.3 + (confidence * 0.4)

        if transcription["keyword"]:
            params["seed_variation"] = keyword_weight
        else:
            params["seed_variation"] = 0.2

        return params

    def _build_prompt(self, transcription: dict, prompt_style: str) -> str | None:
        """Build prompt text from transcription keywords."""
        keyword = transcription["keyword"]
        if keyword and len(keyword) > 2:
            return f"{keyword}, {prompt_style}"
        return None

    # ---- Visualization ----

    def _render_debug_frame(self, transcription: dict, params: dict, prompt: str | None) -> torch.Tensor:
        """Render a simple debug visualization frame showing transcription state.

        Returns tensor of shape (1, H, W, 3) in [0, 1] range.
        """
        frame = torch.full(
            (1, self.height, self.width, 3),
            0.08,
            dtype=torch.float32,
            device=self.device,
        )

        # Confidence bar on left edge
        if transcription["confidence"] > 0:
            bar_height = int(self.height * transcription["confidence"])
            bar_top = self.height - bar_height
            # Green bar
            frame[0, bar_top:, 0:8, 1] = 0.7

        # Keyword indicator — bright block in top-right when keyword detected
        if transcription["keyword"]:
            block_size = 40
            # Cyan block
            frame[0, 10:10 + block_size, self.width - 10 - block_size:self.width - 10, 1] = 0.8
            frame[0, 10:10 + block_size, self.width - 10 - block_size:self.width - 10, 2] = 0.8

        # Word count visualizer — horizontal bars
        word_count = transcription["word_count"]
        if word_count > 0:
            bar_width = min(self.width - 40, word_count * 30)
            y_pos = self.height // 2
            # Warm orange bars
            frame[0, y_pos:y_pos + 12, 20:20 + bar_width, 0] = 0.9
            frame[0, y_pos:y_pos + 12, 20:20 + bar_width, 1] = 0.5

        # Active listening indicator — pulsing dot bottom-center
        dot_x = self.width // 2
        dot_y = self.height - 30
        dot_r = 8
        y1, y2 = max(0, dot_y - dot_r), min(self.height, dot_y + dot_r)
        x1, x2 = max(0, dot_x - dot_r), min(self.width, dot_x + dot_r)
        # Red listening dot
        frame[0, y1:y2, x1:x2, 0] = 0.9
        frame[0, y1:y2, x1:x2, 1] = 0.15
        frame[0, y1:y2, x1:x2, 2] = 0.15

        return frame.clamp(0, 1)

    # ---- Pipeline interface ----

    def prepare(self, **kwargs) -> Requirements:
        """Declare input requirements.

        In text mode we don't need video input.
        In video/preprocessor mode we need 1 frame.
        """
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process one pipeline tick.

        Reads audio from microphone queue, transcribes, extracts keywords,
        maps to Scope parameters, and returns output.

        In text mode: returns a debug visualization frame.
        In video mode: passes through the input video (for preprocessor use).

        The prompt and parameter modifications are logged and can be
        picked up by Scope's parameter system.

        Args:
            **kwargs: Pipeline parameters including:
                - video: Input video frames (list of tensors) or None
                - confidence_threshold: Min confidence to accept (float)
                - keyword_weight: How strongly keywords affect variation (float)
                - prompt_style: Style suffix for prompts (str)

        Returns:
            Dict with "video" key containing output tensor in THWC [0,1] format.
        """
        # Get runtime parameters
        confidence_threshold = kwargs.get("confidence_threshold", 0.3)
        keyword_weight = kwargs.get("keyword_weight", 0.8)
        prompt_style = kwargs.get("prompt_style", "cinematic")
        video_input = kwargs.get("video")

        # Process audio
        if self._whisper_model is not None:
            transcription = self._process_audio_queue(confidence_threshold)
        else:
            transcription = {
                "full_phrase": "",
                "keyword": "",
                "confidence": 0.0,
                "word_count": 0,
            }

        # Map to parameters
        scope_params = self._map_to_scope_params(transcription, keyword_weight)
        prompt = self._build_prompt(transcription, prompt_style)

        # Log state for debugging
        if transcription["full_phrase"]:
            logger.info(f"Transcription: {transcription['full_phrase']}")
        if transcription["keyword"]:
            logger.info(f"Keyword: {transcription['keyword']}")
        if prompt:
            logger.info(f"Prompt: {prompt}")
        if scope_params:
            logger.debug(f"Params: {scope_params}")

        # Generate output
        if video_input is not None:
            # Preprocessor / video mode: pass through input
            from scope.core.pipelines.process import normalize_frame_sizes

            frames = normalize_frame_sizes(video_input)
            stacked = torch.stack([f.squeeze(0) for f in frames], dim=0)
            result = stacked.to(device=self.device, dtype=torch.float32) / 255.0
            return {"video": result.clamp(0, 1)}
        else:
            # Text mode: render debug visualization
            return {"video": self._render_debug_frame(transcription, scope_params, prompt)}

    def __del__(self):
        """Clean up audio stream and models."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._whisper_model is not None:
            del self._whisper_model
        if self._nlp is not None:
            del self._nlp
        torch.cuda.empty_cache()
