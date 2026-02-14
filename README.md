# 🎤 Scope Audio Transcription Plugin

**Voice-controlled AI video generation for [Daydream Scope](https://daydream.live)**

Speak into your microphone and watch AI-generated imagery transform in real time. Say "butterfly" and a butterfly appears. Say "ocean sunset" and the scene shifts. Your voice becomes the paintbrush.

Built for live performance and interactive installation. Based on [The Mirror's Echo](https://kristafaist.com) by Krista Faist.

## How It Works

```
Microphone → faster-whisper → spaCy NLP → StreamDiffusion
    ↓              ↓              ↓              ↓
 48kHz audio   "that's my    [Freddy,      AI-generated
 capture       little guy    little guy]    imagery of
               Freddy"       (nouns only)   Freddy
```

The plugin runs as a **preprocessor** in front of StreamDiffusion:

1. Captures microphone audio in 3-second chunks at 48kHz
2. Resamples to 16kHz and transcribes with faster-whisper (CPU, int8 quantized)
3. Extracts concrete nouns and noun phrases using spaCy NLP
4. Injects extracted nouns as the generation prompt with cache reset
5. Filters filler speech — "um okay so like" produces no prompt change
6. Falls back to the UI prompt box when no voice nouns are detected

## Features

- **Real-time voice-to-image** — speak and see results in ~2 seconds
- **Noun extraction** — only concrete nouns drive the image, not filler words
- **UI prompt fallback** — text box prompt stays active when you're not speaking
- **Whisper on CPU** — faster-whisper int8 keeps your GPU free for StreamDiffusion
- **Cache reset on change** — clean transitions between prompts, no ghosting
- **LIFO audio queue** — always transcribes the most recent speech, not a backlog
- **Prompt monitor** — included tkinter overlay shows what's driving the video

## Installation

### Prerequisites

- [Daydream Scope](https://daydream.live) installed
- Python 3.10+ (Scope's bundled Python works)
- A microphone

### Install the plugin

```bash
# From Scope's virtual environment
pip install -e .
python -m spacy download en_core_web_sm
```

### Microphone setup

The plugin defaults to device 27 (Intel Smart Sound) at 48kHz. To find your mic device number:

```python
import sounddevice as sd
print(sd.query_devices())
```

Then edit `pipeline.py` line with `mic_device = 27` to match your device index.

## Usage

1. Open Daydream Scope
2. Select **Audio Transcription** as the first pipeline (preprocessor)
3. Select **StreamDiffusion** as the second pipeline
4. Set Input Mode to **Video** (the plugin overrides this to text-only internally)
5. Type a base prompt in the text box (this is your fallback prompt)
6. Click Play — speak into your mic and watch the imagery respond

### What you'll see in the logs

```
AUDIO-PLUGIN: transcribing...
AUDIO-PLUGIN: audio amplitude=0.0352
AUDIO-PLUGIN: result='That's my little guy Freddy.'
AUDIO-PLUGIN: nouns extracted: ['my little guy', 'Freddy']
AUDIO-PLUGIN: >>> NEW PROMPT: 'my little guy, Freddy' (from: 'That's my little guy Freddy.')
```

### Prompt priority

| Source | Behavior |
|--------|----------|
| Voice nouns | Immediately override the active prompt with cache reset |
| UI text box | Accepted after the user types a new value; clears voice prompt |
| No speech | Voice prompt persists until UI text box changes |

## Prompt Monitor

An always-on-top tkinter overlay that shows what's driving the video output in real time.

```bash
# Launch the monitor
python tools/scope-prompt-monitor.pyw
```

Shows:
- 🎤 **VOICE** (green) — voice noun prompt active
- 📝 **UI PROMPT** (yellow) — text box prompt active
- 🔶 **FALLBACK** (orange) — voice timed out, reverted to text box
- Amplitude bars, extracted nouns, raw transcription, skipped filler

## Architecture

```
scope_audio_transcription/
├── __init__.py              # Package version
├── plugin.py                # @hookimpl registration for Scope
└── pipelines/
    ├── __init__.py           # Pipeline exports
    ├── pipeline.py           # Main pipeline (voice capture + NLP + prompt injection)
    └── schema.py             # Scope UI configuration schema
tools/
└── scope-prompt-monitor.pyw  # Real-time prompt overlay (tkinter)
```

### Pipeline processor integration

The plugin requires three edits to Scope's `pipeline_processor.py` to ensure prompt overrides from the preprocessor always reach StreamDiffusion:

1. **Queue bypass** — preprocessor parameters merge directly into the next processor's state instead of going through the parameter queue (which can fill up and drop overrides)
2. **Larger parameter queue** — `maxsize=64` instead of 8
3. **Larger output queue** — `maxsize=64` instead of 8

See the [installation guide](https://github.com/kfaist/scope-audio-transcription/wiki) for exact edit locations.

## Whisper Model Options

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| `tiny.en` | 39MB | Fastest | Basic | ~0.5GB |
| `base.en` | 74MB | Fast | Good | ~0.5GB |
| `small.en` | 244MB | **Default** | **Great** | ~0.5GB |
| `medium.en` | 769MB | Slower | Best | ~1GB |

All models run on CPU with int8 quantization via faster-whisper, keeping GPU memory free for StreamDiffusion.

## Based On

This plugin is the technical core of **The Mirror's Echo**, an interactive AI projection installation by Krista Faist. The installation transforms spoken words into evolving visual landscapes using Whisper AI, spaCy NLP, TouchDesigner, and StreamDiffusion.

- **Artist:** [Krista Faist](https://kristafaist.com)
- **Gallery:** Chaos Contemporary Craft, Sarasota FL
- **Fuse Factory** Artist-in-Residence 2024, Columbus OH

## License

MIT — Copyright (c) 2025 Krista Faist
