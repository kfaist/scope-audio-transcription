from setuptools import setup, find_packages

setup(
    name="scope-audio-transcription",
    version="0.1.0",
    author="Krista Faist",
    author_email="kristabluedoor@gmail.com",
    description="Real-time audio transcription plugin for Daydream Scope",
    url="https://github.com/kfaist/scope-audio-transcription",
    packages=["scope_audio_transcription"],
    python_requires=">=3.9",
    install_requires=[
        "openai-whisper",
        "nltk",
        "spacy",
        "soundfile",
    ],
    entry_points={
        "scope": [
            "audio-transcription = scope_audio_transcription.plugin:create_plugin",
        ],
    },
)
