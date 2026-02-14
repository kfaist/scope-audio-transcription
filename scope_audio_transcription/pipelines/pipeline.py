"""Audio Transcription pipeline for Daydream Scope.

Captures microphone audio, transcribes with Whisper AI,
and injects transcriptions directly as prompts.

IMPORTANT: As a preprocessor, we must pass video through but NOT flood
the output queue. We return video on every call (Scope needs it) but
the framework's throttler + output queue handle backpressure.

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

        # Timing - process audio every 5 seconds
        self._last_process_time = 0.0
        self._process_interval = 5.0

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

        # Record at native 48000Hz (Intel Smart Sound), resample to 16kHz for Whisper
        self._native_rate = 48000
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
            # Use Intel Smart Sound mic (device 27 @ 48000Hz) - louder pickup
            mic_device = 27
            logger.info(f"AUDIO-PLUGIN: opening mic device {mic_device} at {self._native_rate}Hz")
            self._stream = sd.InputStream(samplerate=self._native_rate, channels=1, callback=_cb, blocksize=1024, device=mic_device)
            self._stream.start()
            logger.info("AUDIO-PLUGIN: mic capture started on Intel Smart Sound")
        except Exception as e:
            logger.error(f"AUDIO-PLUGIN: mic start failed: {e}")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video_input = kwargs.get("video")
        output = {}

        # Pass video through for our own output queue (preprocessor must output video)
        if video_input is not None:
            frame = video_input[0] if isinstance(video_input, list) else video_input
            if frame.dim() == 4:
                frame = frame.squeeze(0)
            frames = frame.unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255.0
            output["video"] = frames.clamp(0, 1)

        # Tell downstream StreamDiffusion to ignore video and use text-only mode
        # pipeline_processor sets _video_mode=False, so video won't be passed to SD
        output["input_mode"] = "text"

        # Slow down to match downstream consumption (~1fps)
        time.sleep(0.9)

        # Re-inject last voice prompt if we have one and it's recent (10s timeout)
        # Otherwise the UI typed prompt passes through untouched
        if self._last_injected and (time.monotonic() - self._last_inject_time < 10.0):
            output["prompts"] = [{"text": self._last_injected, "weight": 100.0}]
        elif self._last_injected and (time.monotonic() - self._last_inject_time >= 10.0):
            # Voice prompt expired — clear it and reset cache so UI prompt takes over
            logger.info("AUDIO-PLUGIN: voice prompt expired, reverting to UI prompt")
            self._last_injected = ""
            output["reset_cache"] = True

        # Only process audio every N seconds
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
        # Resample from native rate (44100) to 16000 for Whisper
        if self._native_rate != self.sample_rate:
            ratio = self.sample_rate / self._native_rate
            new_len = int(len(audio_float) * ratio)
            indices = np.arange(new_len) / ratio
            indices_floor = np.clip(indices.astype(int), 0, len(audio_float) - 1)
            audio_float = audio_float[indices_floor]
        max_amp = np.max(np.abs(audio_float))
        logger.info(f"AUDIO-PLUGIN: audio amplitude={max_amp:.4f}")
        if max_amp < 0.02:
            return output
        segments, info = self._whisper_model.transcribe(
            audio_float, language="en", beam_size=1, vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"AUDIO-PLUGIN: result='{text}'")

        if not text or len(text) < 3:
            return output

        # Extract concrete/compound nouns only — skip filler speech
        nouns = []
        if self._nlp:
            doc = self._nlp(text)
            # Get compound nouns (adjective+noun phrases) and standalone nouns
            for chunk in doc.noun_chunks:
                # Filter out pronouns and determiners-only chunks
                has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in chunk)
                if has_noun:
                    nouns.append(chunk.text.strip())
            # Fallback: grab individual nouns if no chunks found
            if not nouns:
                nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
            logger.info(f"AUDIO-PLUGIN: nouns extracted: {nouns}")
        else:
            # No spaCy — pass through raw text
            nouns = [text]

        if not nouns:
            logger.info(f"AUDIO-PLUGIN: no nouns found in '{text}', falling back to UI prompt")
            return output

        # Build prompt from nouns only
        noun_prompt = ", ".join(nouns)

        # Skip duplicates
        if noun_prompt == self._last_injected:
            return output

        # Update persistent prompt and reset cache so SD regenerates from new prompt
        self._last_injected = noun_prompt
        self._last_inject_time = time.monotonic()
        output["prompts"] = [{"text": noun_prompt, "weight": 100.0}]
        output["reset_cache"] = True
        logger.info(f"AUDIO-PLUGIN: >>> NEW PROMPT: '{noun_prompt}' (from: '{text}')")
        return output

    def __del__(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
