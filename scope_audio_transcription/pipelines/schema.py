"""Configuration schema for Audio Transcription pipeline."""

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class AudioTranscriptionConfig(BasePipelineConfig):
    """Configuration for the Audio Transcription pipeline.

    Captures microphone audio, transcribes with Whisper AI, extracts
    concrete nouns via NLP, and maps speech to Scope generation parameters.
    Can run standalone (text mode) or as a preprocessor that injects
    prompts and modifies parameters for another pipeline.
    """

    pipeline_id = "audio-transcription"
    pipeline_name = "Audio Transcription"
    pipeline_description = (
        "Voice-controlled AI video generation. Transcribes speech with Whisper, "
        "extracts keywords, and maps voice to generation parameters. "
        "Use as preprocessor to inject prompts into other pipelines."
    )
    pipeline_version = "0.2.0"

    # No prompts from the user — we generate them from speech
    supports_prompts = False

    # Text mode (no video input required) — generates output from audio
    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(input_size=1),
    }

    # Can also be used as a preprocessor for other pipelines
    usage = [UsageType.PREPROCESSOR]

    # No heavy model downloads required by Scope (Whisper downloads on first use)
    requires_models = False
    estimated_vram_gb = 1.0

    # --- Load-time parameters (set before streaming starts) ---

    model_size: str = Field(
        default="base.en",
        description="Whisper model size. Larger = more accurate but slower.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Whisper Model",
            is_load_param=True,
        ),
    )

    # --- Runtime parameters (adjustable while streaming) ---

    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum transcription confidence to accept results.",
        json_schema_extra=ui_field_config(
            order=2,
            label="Confidence Threshold",
        ),
    )

    keyword_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How strongly keywords influence seed variation.",
        json_schema_extra=ui_field_config(
            order=3,
            label="Keyword Weight",
        ),
    )

    prompt_style: str = Field(
        default="cinematic",
        description="Style suffix appended to keyword prompts (e.g. cinematic, dreamy, abstract).",
        json_schema_extra=ui_field_config(
            order=4,
            label="Prompt Style",
        ),
    )

    audio_chunk_seconds: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Duration of audio chunks to process (seconds).",
        json_schema_extra=ui_field_config(
            order=5,
            label="Audio Chunk Duration",
        ),
    )
