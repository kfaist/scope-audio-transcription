"""
Scope Audio Transcription Plugin

Real-time speech-to-text that controls Scope video generation parameters.
Uses Whisper AI for transcription, spaCy + WordNet for keyword extraction,
and maps voice input to generation parameters.

Author: Krista Faist
Based on The Mirror's Echo interactive installation.
"""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register the audio transcription pipeline with Scope."""
    from .pipelines.pipeline import AudioTranscriptionPipeline

    register(AudioTranscriptionPipeline)
