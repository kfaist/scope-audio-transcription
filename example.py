"""
Voice-Controlled Scope Demo
Speak into your microphone to control video generation!
"""

import sounddevice as sd
import numpy as np
from scope_audio_transcription import AudioTranscriptionPlugin

print("🎤 Scope Audio Transcription Demo")
print("=" * 60)

# Initialize
print("Loading Whisper model...")
plugin = AudioTranscriptionPlugin(model_size="base.en")
plugin.setup()

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Buffer
audio_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
buffer_index = 0

def audio_callback(indata, frames, time, status):
    global buffer_index
    
    if status:
        print(f"⚠️  {status}")
        
    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
    
    remaining = CHUNK_SAMPLES - buffer_index
    if len(audio_chunk) <= remaining:
        audio_buffer[buffer_index:buffer_index + len(audio_chunk)] = audio_chunk
        buffer_index += len(audio_chunk)
    else:
        audio_buffer[buffer_index:] = audio_chunk[:remaining]
        
        # Process
        result = plugin.process_audio(audio_buffer.copy())
        
        if result["full_phrase"]:
            print(f"\n💬 {result['full_phrase']}")
            
        if result["keyword"]:
            print(f"   🎯 Keyword: {result['keyword']}")
            
        print(f"   📊 Confidence: {result['confidence']:.2f}")
        
        # Scope parameters
        params = plugin.get_scope_parameters(result)
        print(f"   🎛️  Scope:")
        for key, val in params.items():
            print(f"      {key}: {val:.2f}")
            
        # Prompt
        prompt = plugin.get_prompt_injection(result)
        if prompt:
            print(f"   ✨ Prompt: {prompt}")
            
        print("=" * 60)
        
        buffer_index = 0
        audio_buffer[:] = 0

print(f"\n🎙️  Listening (16kHz)...")
print("🗣️  Speak into your microphone")
print("⌨️  Press Ctrl+C to stop\n")
print("=" * 60)

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=1024
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
        
plugin.cleanup()
print("✅ Done!")
