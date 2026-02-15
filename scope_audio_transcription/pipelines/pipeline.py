"""Audio Transcription pipeline for Daydream Scope.

Captures microphone audio, transcribes with Whisper AI,
extracts concrete nouns via NLP, and injects them as prompts.

Voice prompts persist until replaced by new voice nouns
or until user submits a new prompt from the UI prompt box.
Cache resets on every prompt change for clean transitions.

Author: Krista Faist
"""

import logging
import queue
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from .schema import AudioTranscriptionConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _find_mic_device():
    """Auto-detect best microphone device. Prefers Intel Smart Sound, then
    NVIDIA Broadcast, then system default. Returns (device_id, sample_rate)."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        preferred = [
            ("Intel", 48000),
            ("NVIDIA Broadcast", 48000),
            ("Realtek", 44100),
        ]
        for keyword, rate in preferred:
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and keyword.lower() in d['name'].lower():
                    # Find the WASAPI version (48kHz) if available
                    if abs(d['default_samplerate'] - rate) < 100:
                        logger.info(f"AUDIO-PLUGIN: auto-detected mic [{i}] {d['name']} @ {rate}Hz")
                        return i, rate
            # Second pass: accept any sample rate for this keyword
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and keyword.lower() in d['name'].lower():
                    r = int(d['default_samplerate'])
                    logger.info(f"AUDIO-PLUGIN: auto-detected mic [{i}] {d['name']} @ {r}Hz")
                    return i, r
        # Fallback to system default
        default_id = sd.default.device[0]
        if default_id is not None and default_id >= 0:
            r = int(devices[default_id]['default_samplerate'])
            logger.info(f"AUDIO-PLUGIN: using default mic [{default_id}] {devices[default_id]['name']} @ {r}Hz")
            return default_id, r
    except Exception as e:
        logger.error(f"AUDIO-PLUGIN: mic detection failed: {e}")
    return None, 48000


class AudioTranscriptionPipeline(Pipeline):

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioTranscriptionConfig

    def __init__(self, model_size="small.en", height=512, width=512, device=None, **kwargs):
        self.model_size = model_size
        self.height = height
        self.width = width
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Audio
        self.sample_rate = 16000
        self._audio_queue = queue.LifoQueue(maxsize=3)
        self._stream = None
        self._running = False
        self._audio_buffer = None
        self._buffer_index = 0

        # Transcription
        self._last_text = ""
        self._last_injected = ""
        self._last_inject_time = 0.0
        self._last_ui_prompt = ""
        self._ui_prompt_initialized = False

        # Timing - process audio every 5 seconds
        self._last_process_time = 0.0
        self._process_interval = 5.0

        # Track whether we just flushed (skip video output for 1 frame after flush)
        self._flush_pending = False
        # Double-flush: count down frames to keep sending reset_cache
        self._flush_frames_remaining = 0

        self._whisper_model = None
        self._nlp = None
        logger.info("AUDIO-PLUGIN: about to load whisper")
        try:
            self._load_whisper()
        except Exception as e:
            logger.error(f"AUDIO-PLUGIN: whisper load FAILED: {e}")
        logger.info("AUDIO-PLUGIN: about to load spaCy")
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("AUDIO-PLUGIN: spaCy loaded")
        except Exception as e:
            logger.error(f"AUDIO-PLUGIN: spaCy load FAILED: {e}")
        logger.info("AUDIO-PLUGIN: about to start audio")
        try:
            self._start_audio()
        except Exception as e:
            logger.error(f"AUDIO-PLUGIN: audio start FAILED: {e}")
        logger.info("AUDIO-PLUGIN: init complete")

    def _load_whisper(self):
        from faster_whisper import WhisperModel
        logger.info("AUDIO-PLUGIN: loading faster-whisper")
        self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        logger.info("AUDIO-PLUGIN: whisper loaded")

    def _start_audio(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            logger.error(f"AUDIO-PLUGIN: sounddevice error: {e}")
            return

        # Auto-detect mic device
        mic_device, self._native_rate = _find_mic_device()
        if mic_device is None:
            logger.error("AUDIO-PLUGIN: no microphone found")
            return

        chunk_samples = int(self._native_rate * 3.0)
        self._audio_buffer = np.zeros(chunk_samples, dtype=np.int16)
        self._buffer_index = 0
        self._running = True

        def _cb(indata, frames, time_info, status):
            if status:
                logger.debug(f"AUDIO-PLUGIN: mic status: {status}")
            audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
            remaining = chunk_samples - self._buffer_index
            if len(audio_chunk) <= remaining:
                self._audio_buffer[self._buffer_index:self._buffer_index + len(audio_chunk)] = audio_chunk
                self._buffer_index += len(audio_chunk)
            else:
                self._audio_buffer[self._buffer_index:] = audio_chunk[:remaining]
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
            logger.info(f"AUDIO-PLUGIN: opening mic device {mic_device} at {self._native_rate}Hz")
            self._stream = sd.InputStream(
                samplerate=self._native_rate, channels=1, callback=_cb,
                blocksize=1024, device=mic_device,
            )
            self._stream.start()
            logger.info("AUDIO-PLUGIN: mic capture started")
        except Exception as e:
            logger.error(f"AUDIO-PLUGIN: mic start failed: {e}")

    def prepare(self, **kwargs) -> Requirements:
        if kwargs.get("video") is not None:
            return Requirements(input_size=1)
        return None

    def __call__(self, **kwargs) -> dict:
        video_input = kwargs.get("video")
        output = {}

        # Pass video through (preprocessor must output video)
        # BUT skip video on flush frames so downstream doesn't queue stale frames
        if video_input is not None and not self._flush_pending:
            frame = video_input[0] if isinstance(video_input, list) else video_input
            if frame.dim() == 4:
                frame = frame.squeeze(0)
            frames = frame.unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255.0
            output["video"] = frames.clamp(0, 1)
        elif video_input is not None and self._flush_pending:
            # Still need to output video (preprocessor requirement) but mark for flush
            frame = video_input[0] if isinstance(video_input, list) else video_input
            if frame.dim() == 4:
                frame = frame.squeeze(0)
            frames = frame.unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255.0
            output["video"] = frames.clamp(0, 1)
            self._flush_pending = False

        # Tell downstream to use text-only mode
        output["input_mode"] = "text"

        # Throttle preprocessor to ~1fps to match downstream StreamDiffusion actual throughput
        # StreamDiffusion runs at 0.5-1fps — anything faster just floods the output queue
        time.sleep(1.0)

        # Double-flush: keep sending reset_cache for a few frames after prompt change
        # to catch any stale frames that snuck into the queue
        if self._flush_frames_remaining > 0:
            self._flush_frames_remaining -= 1
            output["reset_cache"] = True

        # --- UI PROMPT BOX: check if user typed a new prompt ---
        ui_prompts = kwargs.get("prompts")
        if ui_prompts and isinstance(ui_prompts, list) and len(ui_prompts) > 0:
            ui_text = ui_prompts[0].get("text", "") if isinstance(ui_prompts[0], dict) else str(ui_prompts[0])
            if ui_text:
                if not self._ui_prompt_initialized:
                    self._last_ui_prompt = ui_text
                    self._ui_prompt_initialized = True
                    logger.info(f"AUDIO-PLUGIN: initial UI prompt recorded: '{ui_text}'")
                elif ui_text != self._last_ui_prompt:
                    # User hit enter with new text — flush everything, inject immediately
                    logger.info(f"AUDIO-PLUGIN: UI prompt changed to '{ui_text}', clearing voice")
                    self._last_ui_prompt = ui_text
                    self._last_injected = ""
                    self._flush_pending = True
                    self._flush_frames_remaining = 3
                    output["prompts"] = [{"text": ui_text, "weight": 100.0}]
                    output["reset_cache"] = True
                    logger.info(f"AUDIO-PLUGIN: >>> FLUSH + INJECT (text box): '{ui_text}'")
                    return output

        # Keep injecting last voice prompt every frame until replaced
        if self._last_injected:
            output["prompts"] = [{"text": self._last_injected, "weight": 100.0}]

        # --- VOICE: only process audio every N seconds ---
        now = time.monotonic()
        if now - self._last_process_time < self._process_interval:
            return output
        self._last_process_time = now

        # Check for audio
        if self._audio_queue.empty():
            return output

        # Transcribe
        try:
            audio = self._audio_queue.get_nowait()
        except queue.Empty:
            return output

        logger.info("AUDIO-PLUGIN: transcribing...")
        audio_float = audio.astype(np.float32) / 32768.0
        # Resample from native rate to 16kHz for Whisper
        if self._native_rate != self.sample_rate:
            ratio = self.sample_rate / self._native_rate
            new_len = int(len(audio_float) * ratio)
            indices = np.arange(new_len) / ratio
            indices_floor = np.clip(indices.astype(int), 0, len(audio_float) - 1)
            audio_float = audio_float[indices_floor]
        max_amp = np.max(np.abs(audio_float))
        logger.info(f"AUDIO-PLUGIN: audio amplitude={max_amp:.4f}")
        if max_amp < 0.008:
            return output
        segments, info = self._whisper_model.transcribe(
            audio_float, language="en", beam_size=1, vad_filter=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"AUDIO-PLUGIN: result='{text}'")

        if not text or len(text) < 3:
            return output

        # Extract concrete/compound nouns — skip filler speech
        nouns = []
        if self._nlp:
            doc = self._nlp(text)
            for chunk in doc.noun_chunks:
                has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in chunk)
                if has_noun:
                    nouns.append(chunk.text.strip())
            if not nouns:
                nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
            logger.info(f"AUDIO-PLUGIN: nouns extracted: {nouns}")
        else:
            nouns = [text]

        if not nouns:
            logger.info(f"AUDIO-PLUGIN: no nouns found in '{text}', skipping")
            return output

        noun_prompt = ", ".join(nouns)

        # Skip duplicates
        if noun_prompt == self._last_injected:
            return output

        # New voice nouns — flush everything and inject immediately
        self._last_injected = noun_prompt
        self._last_inject_time = time.monotonic()
        self._flush_pending = True
        self._flush_frames_remaining = 3
        output["prompts"] = [{"text": noun_prompt, "weight": 100.0}]
        output["reset_cache"] = True
        logger.info(f"AUDIO-PLUGIN: >>> FLUSH + INJECT (voice): '{noun_prompt}' (from: '{text}')")
        return output

    def __del__(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
