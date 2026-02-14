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
    concrete nouns via spaCy NLP, and injects them as prompts into
    a downstream StreamDiffusion pipeline.

    Runs as a preprocessor: captures voice, extracts nouns, overrides
    the downstream pipeline's prompt and input_mode so spoken words
    drive the generated imagery.
    """

    pipeline_id = "audio-transcription"
    pipeline_name = "Audio Transcription"
    pipeline_description = (
        "Voice-controlled AI video generation. Transcribes speech with "
        "faster-whisper, extracts concrete nouns with spaCy, and injects "
        "them as prompts into StreamDiffusion. Speak and watch the imagery change."
    )
    pipeline_version = "0.3.0"

    # Prompts come from voice — but UI prompt box still works as fallback
    supports_prompts = True

    # Video passthrough mode (preprocessor receives video, passes it along)
    modes = {
        "video": ModeDefaults(input_size=1, default=True),
        "text": ModeDefaults(),
    }

    # Preprocessor: sits in front of StreamDiffusion
    usage = [UsageType.PREPROCESSOR]

    # Whisper model downloads on first use (~150MB for small.en)
    requires_models = False
    estimated_vram_gb = 0.5  # Whisper runs on CPU (int8)

    # --- Load-time parameters ---

    model_size: str = Field(
        default="small.en",
        description="Whisper model size. Options: tiny.en, base.en, small.en, medium.en. Larger = more accurate but slower.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Whisper Model",
            is_load_param=True,
        ),
    )
