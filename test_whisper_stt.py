#!/usr/bin/env python3
"""
Test script for Whisper STT implementation
"""

import asyncio
import numpy as np
from whisper_stt import WhisperSTT
from livekit.agents.utils import AudioBuffer


async def test_whisper_stt():
    """Test the Whisper STT implementation."""
    print("Testing Whisper STT implementation...")
    
    # Create Whisper STT instance
    stt = WhisperSTT(model_name="base", language=None)
    
    # Create a simple test audio buffer (silence)
    # In a real scenario, this would contain actual audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Create AudioBuffer
    audio_buffer = AudioBuffer(
        samples=samples,
        sample_rate=sample_rate,
        num_channels=1
    )
    
    print(f"Audio buffer created: {len(samples)} samples at {sample_rate}Hz")
    
    # Test transcription
    try:
        result = await stt._recognize_impl(audio_buffer)
        print(f"Transcription result: {result}")
        if result.alternatives:
            speech_data = result.alternatives[0]
            print(f"Text: '{speech_data.text}'")
            print(f"Language: {speech_data.language}")
            print(f"Confidence: {speech_data.confidence}")
        print(f"Event type: {result.type}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
    
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_whisper_stt()) 