# Migration from Deepgram to Whisper STT

This document explains the changes made to replace Deepgram STT with an offline Whisper implementation.

## Changes Made

### 1. New Files Created

- **`src/whisper_stt.py`**: Custom Whisper STT implementation that integrates with LiveKit Agents
- **`test_whisper_stt.py`**: Test script to verify Whisper STT functionality

### 2. Files Modified

- **`src/agent.py`**: 
  - Removed Deepgram import
  - Added WhisperSTT import
  - Replaced `deepgram.STT()` with `WhisperSTT()`
  - Updated comments

- **`pyproject.toml`**:
  - Removed `deepgram` from livekit-agents dependencies
  - Added `openai-whisper` and `numpy` dependencies

## Whisper STT Features

### Model Options
- **"tiny"**: 39MB - Fastest, basic transcription
- **"base"**: 74MB - Good balance (default)
- **"small"**: 244MB - Better accuracy
- **"medium"**: 769MB - High accuracy
- **"large"**: 1550MB - Best accuracy

### Language Support
- Set `language=None` for auto-detection (recommended)
- Or specify language code: `"en"`, `"es"`, `"fr"`, etc.

### Offline Usage
- Models are downloaded once and cached locally
- Works completely offline after initial download
- Models stored in `~/.cache/whisper/`

## Usage

```python
from whisper_stt import WhisperSTT

# Basic usage with auto language detection
stt = WhisperSTT(model_name="base", language=None)

# Or specify language
stt = WhisperSTT(model_name="small", language="en")
```

## Installation

1. Install dependencies:
```bash
pip install -e .
```

2. The first time you run the agent, Whisper will download the model automatically.

## Testing

Run the test script to verify everything works:
```bash
python test_whisper_stt.py
```

## Benefits of Whisper

1. **Offline Operation**: No internet required after model download
2. **Privacy**: All processing happens locally
3. **Cost**: No API costs or usage limits
4. **Multilingual**: Supports 99+ languages
5. **Customizable**: Different model sizes for speed vs accuracy trade-offs

## Migration Notes

- The Whisper implementation provides the same interface as Deepgram STT
- No changes needed to your agent logic
- Whisper may have slightly different latency characteristics
- Consider using "tiny" or "base" models for real-time applications 