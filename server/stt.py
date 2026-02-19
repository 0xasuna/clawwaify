"""
Fast Speech-to-Text using microphone with faster-whisper.
Uses the fastest models for low latency.
"""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import sys

# Model options (fastest to most accurate):
# "tiny.en"      - ~39M params, fastest, English only
# "base.en"      - ~74M params, fast, English only  
# "small.en"     - ~244M params, good balance
# "distil-small.en" - distilled, very fast
# "distil-medium.en" - distilled, fast + accurate

class SpeechToText:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        """
        Initialize the STT model.
        
        Args:
            model_size: Model to use (tiny.en is fastest)
            device: "cuda" for GPU, "cpu" for CPU
            compute_type: "float16" for GPU, "int8" for CPU
        """
        print(f"Loading model '{model_size}'...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000  # Whisper expects 16kHz
        print("Model loaded!")
    
    def record_audio(self, duration=5, silence_threshold=0.01, silence_duration=1.5):
        """
        Record audio from microphone.
        
        Args:
            duration: Maximum recording duration in seconds
            silence_threshold: Audio level below this is considered silence
            silence_duration: Stop after this many seconds of silence
        """
        print("Recording... (speak now)")
        
        audio_chunks = []
        silence_samples = 0
        silence_limit = int(silence_duration * self.sample_rate)
        
        def callback(indata, frames, time, status):
            nonlocal silence_samples
            if status:
                print(status, file=sys.stderr)
            
            audio_chunks.append(indata.copy())
            
            # Check for silence
            volume = np.abs(indata).mean()
            if volume < silence_threshold:
                silence_samples += frames
            else:
                silence_samples = 0
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                           dtype='float32', callback=callback):
            # Wait for max duration or silence
            import time
            start = time.time()
            while time.time() - start < duration:
                if silence_samples > silence_limit and len(audio_chunks) > 10:
                    break
                sd.sleep(100)
        
        print("Recording stopped.")
        
        if not audio_chunks:
            return np.array([], dtype=np.float32)
        
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        return audio
    
    def transcribe(self, audio):
        """
        Transcribe audio to text.
        
        Args:
            audio: numpy array of audio data (16kHz, float32)
        
        Returns:
            Transcribed text string
        """
        if len(audio) == 0:
            return ""
        
        segments, info = self.model.transcribe(
            audio,
            language="en",  # Skip language detection for speed
            beam_size=1,    # Greedy decoding (fastest)
            vad_filter=True,  # Filter silence
            without_timestamps=True  # Faster without timestamps
        )
        
        # Combine all segments
        text = " ".join(segment.text.strip() for segment in segments)
        return text
    
    def listen(self, duration=5):
        """
        Record from microphone and transcribe.
        
        Args:
            duration: Maximum recording duration
        
        Returns:
            Transcribed text
        """
        audio = self.record_audio(duration=duration)
        return self.transcribe(audio)


def main():
    """Test the STT system."""
    # Use CPU with int8 for fast inference without GPU
    # For GPU: device="cuda", compute_type="float16"
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    
    print("\n--- Speech to Text Test ---")
    print("Press Enter to start recording (or 'q' to quit)")
    
    while True:
        cmd = input("\n> ")
        if cmd.lower() == 'q':
            break
        
        text = stt.listen(duration=10)
        print(f"\nYou said: {text}")


if __name__ == "__main__":
    main()
