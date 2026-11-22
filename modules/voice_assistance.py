"""
Voice Assistance Module for BlindSpot
Provides spoken acoustic feedback for visually impaired pedestrians:
- Offline pyttsx3 or gTTS speech generation
- In-memory MP3 audio caching for instant replay in Streamlit
- Graceful fallbacks: app never crashes if audio device is missing or offline
"""

import io
import os
import hashlib
from typing import Optional, Tuple
from pathlib import Path

# In-memory audio cache to prevent redundant synthesis
_AUDIO_CACHE = {}

def generate_speech_audio(text: str, lang: str = 'en') -> Optional[bytes]:
    """
    Generates MP3 audio bytes using gTTS (or falls back safely).
    Returns raw audio bytes suitable for Streamlit st.audio().
    """
    if not text or not text.strip():
        return None
        
    text_clean = text.strip()
    cache_key = hashlib.md5(text_clean.encode('utf-8')).hexdigest()
    
    if cache_key in _AUDIO_CACHE:
        return _AUDIO_CACHE[cache_key]
        
    try:
        from gtts import gTTS
        fp = io.BytesIO()
        tts = gTTS(text=text_clean, lang=lang, slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        _AUDIO_CACHE[cache_key] = audio_bytes
        return audio_bytes
    except Exception as e:
        # Fallback if offline or gTTS network fails
        print(f"[VoiceAssistance] Audio synthesis notice: {e}. Falling back to visual text alerts.")
        return None

def speak_offline_pyttsx3(text: str) -> bool:
    """
    Attempts direct system voice playback using pyttsx3.
    Safe: does not crash if sound driver is uninitialized or in headless environment.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[VoiceAssistance] pyttsx3 engine notice: {e}")
        return False
