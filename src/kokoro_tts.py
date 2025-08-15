import asyncio
import logging
import os
import numpy as np
import torch
from typing import Optional
from livekit.agents.tts import TTS, TTSCapabilities
from livekit import rtc
from typing import Optional, AsyncGenerator  # <-- ADD THIS IMPORT
from typing import Optional, AsyncGenerator # This one is important
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)
class _AudioDataWrapper:
    def __init__(self, frame: rtc.AudioFrame):
        self.frame = frame


class KokoroTTS(TTS):
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        device: str = "auto",
        voice: str = "af_heart",
        lang_code: str = "a"
    ):
        """
        Initialize Kokoro TTS with lazy loading to avoid timeout issues.
        
        Args:
            model_path: Path to Kokoro model directory
            device: Device to run inference on ("auto", "cpu" or "cuda")
            voice: Default voice to use
            lang_code: Language code ('a'=American English, 'b'=British English)
        """
        # Initialize with streaming capabilities
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),  # Start with non-streaming to avoid complexity
            sample_rate=24000,
            num_channels=1
        )
        
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.voice = voice
        self.lang_code = lang_code
        
        # Lazy loading to avoid initialization timeout
        self.model = None
        self._model_loaded = False
        self._sample_rate = 24000
        
        logger.info(f"Kokoro TTS initialized (lazy loading enabled, device: {self.device})")
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded (lazy loading to avoid startup timeout)"""
        if self._model_loaded:
            return
        
        try:
            logger.info("Loading Kokoro TTS model...")
            
            # Try to load kokoro package
            try:
                from kokoro import KPipeline
                self.model = KPipeline(lang_code=self.lang_code)
                self._synthesis_method = "standard"
                logger.info("Kokoro standard model loaded successfully")
                
            except ImportError:
                logger.error("Kokoro package not found. Install with: pip install kokoro")
                raise
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS model: {e}")
            raise

    @asynccontextmanager
    async def synthesize(self, text: str, **kwargs):
        """
        This method is a context manager that yields an async generator.
        This is the required structure for older livekit-agent versions.
        """
        self._ensure_model_loaded()
        if not self.model:
            raise RuntimeError("Kokoro model not loaded")

        async def generator():
            """This inner generator does the actual work and yields the wrapped audio."""
            try:
                logger.debug(f"Synthesizing: {text[:50]}...")
                
                loop = asyncio.get_event_loop()
                samples, sample_rate = await loop.run_in_executor(
                    None, 
                    self._synthesize_blocking, 
                    text
                )
                
                audio_frame = self._samples_to_audio_frame(samples, sample_rate)
                
                # THE KEY FIX: Wrap the AudioFrame before yielding it.
                yield _AudioDataWrapper(frame=audio_frame)
                
            except Exception as e:
                logger.error(f"Kokoro synthesis failed: {e}")
                # We don't yield anything on failure here to stop the stream.

        # The context manager yields the generator object itself.
        yield generator()
    # async def synthesize(self, text: str, **kwargs) -> AsyncGenerator[rtc.AudioFrame, None]:
    #     """
    #     Synthesize speech from text using Kokoro.
    #     This method is an async generator that yields a single AudioFrame.
        
    #     Args:
    #         text: Text to synthesize
            
    #     Yields:
    #         AudioFrame with synthesized audio
    #     """
    #     try:
    #         logger.debug(f"Synthesizing: {text[:50]}...")
            
    #         # Ensure model is loaded
    #         self._ensure_model_loaded()
            
    #         if not self.model:
    #             raise RuntimeError("Kokoro model not loaded")
            
    #         # Run synthesis in executor to avoid blocking
    #         loop = asyncio.get_event_loop()
    #         samples, sample_rate = await loop.run_in_executor(
    #             None, 
    #             self._synthesize_blocking, 
    #             text
    #         )
            
    #         # Convert samples to AudioFrame
    #         audio_frame = self._samples_to_audio_frame(samples, sample_rate)
            
    #         logger.debug(f"Synthesis completed: {len(samples)} samples")
    #         yield audio_frame  # <-- THIS IS THE KEY CHANGE
            
    #     except Exception as e:
    #         logger.error(f"Kokoro synthesis failed: {e}")
    #         # Yield silence on error to keep the agent running
    #         yield self._create_silence_frame()
    
    # def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, int]:
    #     """Blocking synthesis call"""
    #     try:
    #         samples, sample_rate = self.model.create(
    #             text=text,
    #             voice=self.voice,
    #             speed=1.0,
    #             remove_silence=True
    #         )
    #         return samples, sample_rate
            
    #     except Exception as e:
    #         logger.error(f"Blocking synthesis failed: {e}")
    #         # Return empty audio on error
    #         return np.zeros(1000, dtype=np.float32), self._sample_rate
    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, int]:
        """The actual blocking call to the Kokoro library."""
        if not self.model:
            raise RuntimeError("Kokoro model not loaded")

        # The KPipeline object is callable and returns a generator.
        # We call it directly, instead of using a ".create()" method.
        generator = self.model(
            text=text,
            voice=self.voice,
            speed=1.0,
            # remove_silence=True # This parameter is not supported in the direct call
        )

        # The generator may yield multiple audio chunks for longer text.
        # We must loop through it and combine all chunks.
        all_audio_chunks = []
        for _, _, audio_chunk in generator:
            all_audio_chunks.append(audio_chunk)

        # Concatenate all chunks into a single numpy array
        samples = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

        # Kokoro's sample rate is fixed at 24000
        sample_rate = 24000
        
        return samples, sample_rate
    
    def _samples_to_audio_frame(self, samples: np.ndarray, sample_rate: int) -> rtc.AudioFrame:
        """Convert audio samples to LiveKit AudioFrame"""
        # Ensure samples are float32 and in correct range
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        # Ensure samples are in [-1, 1] range
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert to int16 for AudioFrame
        samples_int16 = (samples * 32767).astype(np.int16)
        
        return rtc.AudioFrame(
            data=samples_int16.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(samples_int16)
        )
    
    def _create_silence_frame(self, duration: float = 0.1) -> rtc.AudioFrame:
        """Create a silent audio frame for error cases"""
        num_samples = int(self._sample_rate * duration)
        samples = np.zeros(num_samples, dtype=np.int16)
        
        return rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=self._sample_rate,
            num_channels=1,
            samples_per_channel=num_samples
        )
    
    def set_voice(self, voice: str) -> None:
        """Change the voice for synthesis"""
        available_voices = self.get_available_voices()
        if voice in available_voices:
            self.voice = voice
            logger.info(f"Voice changed to: {voice}")
        else:
            logger.warning(f"Voice '{voice}' not available. Available: {available_voices}")
    
    def get_available_voices(self) -> list[str]:
        """Get list of available voices"""
        return [
            "af_heart",      # American Female, warm
            "af_sarah",      # American Female, professional  
            "af_nicole",     # American Female, young
            "af_sky",        # American Female, bright
            "am_michael",    # American Male, professional
            "am_adam",       # American Male, deep
            "am_daniel",     # American Male, casual
            "bf_emma",       # British Female
            "bm_lewis",      # British Male
        ]
    
    async def aclose(self) -> None:
        """Clean up resources"""
        self.model = None
        self._model_loaded = False
        logger.info("Kokoro TTS resources cleaned up")


# Convenience function to create Kokoro TTS instance
def create_kokoro_tts(
    model_path: Optional[str] = None, 
    device: str = "auto",
    voice: str = "af_heart",
    lang_code: str = "a"
) -> KokoroTTS:
    """
    Create a Kokoro TTS instance with safe initialization.
    
    Args:
        model_path: Path to Kokoro model directory
        device: Device to run inference on ("auto", "cpu", or "cuda")
        voice: Default voice to use
        lang_code: Language code
        
    Returns:
        KokoroTTS instance
    """
    return KokoroTTS(
        model_path=model_path,
        device=device,
        voice=voice,
        lang_code=lang_code
    )


# Test function (run separately to verify installation)
async def test_kokoro_standalone():
    """Standalone test function - run this separately to test Kokoro"""
    print("🧪 Testing Kokoro TTS (standalone)...")
    
    try:
        # Test kokoro import
        from kokoro import KPipeline
        print("✅ Kokoro package imported successfully")
        
        # Test model loading
        print("Loading model...")
        pipeline = KPipeline(lang_code='a')
        print("✅ Model loaded successfully")
        
        # Test synthesis
        print("Testing synthesis...")
        samples, sample_rate = pipeline.create(
            text="Hello! This is a test of Kokoro TTS.",
            voice="af_heart",
            speed=1.0,
            remove_silence=True
        )
        print(f"✅ Synthesis successful! Generated {len(samples)} samples at {sample_rate}Hz")
        
        # Save test file
        try:
            import soundfile as sf
            sf.write("kokoro_standalone_test.wav", samples, sample_rate)
            print("✅ Test audio saved as 'kokoro_standalone_test.wav'")
        except ImportError:
            print("ℹ️ soundfile not available - couldn't save test file")
        
        return True
        
    except ImportError as e:
        print(f"❌ Kokoro import failed: {e}")
        print("Install with: pip install kokoro")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(test_kokoro_standalone())