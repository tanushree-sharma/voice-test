"""
Audio recorder for capturing and saving conversation audio.

Handles sample rate mismatches between input (microphone) and output (TTS) audio
by automatically resampling to the highest detected rate.
"""
import wave
import numpy as np
from scipy import signal
from loguru import logger
from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioRecorder(FrameProcessor):
    """
    Custom frame processor that captures all audio (user input + TTS output) and saves to WAV.
    Handles sample rate mismatches by resampling to the highest detected rate.
    """

    def __init__(self, output_path: str, sample_rate: int = 24000, channels: int = 1):
        super().__init__()
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_frames = []  # (audio_data, frame_sample_rate) tuples
        self.detected_sample_rates = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Capture all audio frames
        if isinstance(frame, AudioRawFrame):
            # Pipecat uses 'sample_rate' attribute on AudioRawFrame
            frame_sample_rate = getattr(frame, 'sample_rate', None)
            if frame_sample_rate:
                self.detected_sample_rates.add(frame_sample_rate)
            self.audio_frames.append((frame.audio, frame_sample_rate))

        await self.push_frame(frame, direction)

    def save_recording(self):
        """Resamples and saves all captured audio to a WAV file."""
        if not self.audio_frames:
            logger.warning("No audio frames to save")
            return

        # Use the highest detected sample rate (typically TTS output at 24kHz)
        if self.detected_sample_rates:
            self.sample_rate = max(self.detected_sample_rates)

        try:
            with wave.open(self.output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)

                for audio_data, frame_sample_rate in self.audio_frames:
                    # Resample if needed
                    if frame_sample_rate and frame_sample_rate != self.sample_rate:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        num_samples = int(len(audio_array) * self.sample_rate / frame_sample_rate)
                        resampled = signal.resample(audio_array, num_samples)
                        audio_data = resampled.astype(np.int16).tobytes()

                    wav_file.writeframes(audio_data)

            logger.info(f"Recording saved to: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save recording to {self.output_path}: {e}")

