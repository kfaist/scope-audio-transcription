# 🎤 Scope Audio Transcription Plugin

**Real-time speech-to-text control for Daydream Scope video generation**

Transform spoken words into dynamic visual parameters for interactive AI video experiences.

## ✨ Features

- 🎙️ **Real-time Whisper transcription** - Fast, accurate speech-to-text
- 🎯 **Intelligent keyword extraction** - Prioritizes concrete nouns (objects, animals, food)
- 🎛️ **Automatic parameter mapping** - Speech controls CFG scale, strength, variation
- ✨ **Dynamic prompt injection** - Keywords enhance generation prompts
- ⚡ **GPU accelerated** - 10x faster with CUDA
- 🔄 **LIFO queue management** - Always uses most recent audio

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run demo
python example.py
```

## 💡 Use Cases

- **Live Performances** - Voice-reactive visuals
- **Interactive Installations** - Like Mirror's Echo!
- **Storytelling** - Narration drives visual evolution
- **Accessibility** - Hands-free video generation

## 📊 How It Works

```
Microphone → Whisper AI → NLP Extraction → Scope Parameters
                ↓              ↓                   ↓
            "butterfly"   Physical noun    CFG: 8.5, Strength: 0.6
```

## 🎯 Parameter Mapping

| Speech Feature | Scope Parameter | Effect |
|----------------|-----------------|--------|
| Word count | `cfg_scale` | More words = higher guidance |
| Confidence | `strength` | Higher confidence = stronger |
| Keywords | `seed_variation` | Keywords = more variation |

## 📖 API Example

```python
from scope_audio_transcription import AudioTranscriptionPlugin

# Initialize
plugin = AudioTranscriptionPlugin(model_size="base.en")
plugin.setup()

# Process audio
result = plugin.process_audio(audio_chunk)
print(result["keyword"])  # "butterfly"

# Get Scope parameters
params = plugin.get_scope_parameters(result)
# {"cfg_scale": 8.5, "strength": 0.6, "seed_variation": 0.8}

# Get prompt injection
prompt = plugin.get_prompt_injection(result)
# "butterfly, cinematic, dramatic lighting"
```

## ⚙️ Configuration

Edit `plugin.json`:

```json
{
  "model_size": "base.en"
}
```

**Model Options:**
- `tiny.en` - Fastest (1GB VRAM)
- `base.en` - **Recommended** (1GB VRAM)  
- `small.en` - Better accuracy (2GB VRAM)
- `medium.en` - Best accuracy (5GB VRAM)

## 🎬 Based On

This plugin is based on **The Mirror's Echo** interactive installation by Krista Faist, which has been successfully deployed in live gallery settings with proven real-time transcription performance.

## 📝 License

AGPL-3.0 WITH DUAL LICENSE FOR COMMERCIAL - Copyright (c) 2026 Krista Faist

## 🔗 Links

- **Author:** Krista Faist (kristabluedoor@gmail.com)
- **Gallery:** Chaos Contemporary Craft, Sarasota FL
- **Daydream:** Interactive AI Video Program 2026

---

**🎤 Transform speech into stunning AI video!**
