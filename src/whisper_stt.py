import asyncio
import logging
import whisper
import numpy as np
from typing import Optional, AsyncGenerator
from livekit.agents.stt import STT, SpeechEvent, SpeechData, SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)


class WhisperSTT(STT):
    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        """
        Initialize Whisper STT with specified model.
        
        Args:
            model_name: Whisper model to use ("tiny", "base", "small", "medium", "large")
            language: Language code (e.g., "en", "es", "fr") or None for auto-detection
        """
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self.model_name = model_name
        self.language = language
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str] = None,
        conn_options=None,
    ) -> SpeechEvent:
        """
        Recognize speech from audio buffer using Whisper.
        
        Args:
            buffer: AudioBuffer containing audio data
            language: Language code or None for auto-detection
            
        Returns:
            SpeechEvent with transcription data
        """
        try:
            # Convert AudioBuffer to numpy array
            if hasattr(buffer, 'samples'):
                audio_data = np.array(buffer.samples, dtype=np.float32)
            elif hasattr(buffer, 'data'):
                # Handle AudioFrame data (int16 format)
                raw_data = np.frombuffer(buffer.data, dtype=np.int16)
                # Convert int16 to float32 and normalize
                audio_data = raw_data.astype(np.float32) / 32768.0
            else:
                # Handle other audio formats
                audio_data = np.array(buffer, dtype=np.float32)
            
            # Prepare transcription options
            options = {}
            if language or self.language:
                options["language"] = language or self.language
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                **options
            )
            
            # Extract text and language
            text = result.get("text", "").strip()
            detected_language = result.get("language", language or self.language or "unknown")
            
            logger.debug(f"Whisper transcription: {text[:100]}...")
            
            # Create SpeechData
            speech_data = SpeechData(
                language=detected_language,
                text=text,
                confidence=1.0,  # Whisper doesn't provide confidence scores
            )
            
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[speech_data]
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(
                    language=language or self.language or "unknown",
                    text="",
                    confidence=0.0
                )]
            )
    



# Convenience function to create Whisper STT instance
def create_whisper_stt(model: str = "base", language: Optional[str] = None) -> WhisperSTT:
    """
    Create a Whisper STT instance.
    
    Args:
        model: Whisper model name ("tiny", "base", "small", "medium", "large")
        language: Language code or None for auto-detection
        
    Returns:
        WhisperSTT instance
    """
    return WhisperSTT(model_name=model, language=language) 