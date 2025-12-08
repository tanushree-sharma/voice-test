"""
Audio recorder for capturing and saving conversation audio.

Handles sample rate mismatches between input (microphone) and output (TTS) audio
by automatically resampling to the highest detected rate.

Uses a streaming approach with a bounded buffer to avoid memory leaks on long conversations.
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

    Uses a bounded buffer to prevent memory leaks - flushes to disk periodically.
    """

    # Maximum frames to buffer before flushing to disk (prevents memory leak)
    MAX_BUFFER_FRAMES = 500  # ~10 seconds at typical frame rates

    def __init__(self, output_path: str, sample_rate: int = 24000, channels: int = 1):
        super().__init__()
        self.output_path = output_path
        self._default_sample_rate = sample_rate  # Store default, never mutate
        self._target_sample_rate = None  # Will be set during initialization
        self.channels = channels
        self.audio_frames = []  # (audio_data, frame_sample_rate) tuples
        self.detected_sample_rates = set()
        self._wav_file = None
        self._is_initialized = False

    def _initialize_wav_file(self):
        """Initialize the WAV file with headers when first frame arrives."""
        if self._is_initialized:
            return

        try:
            # Determine target sample rate: use highest detected, or fall back to default
            # Store in separate variable to avoid mutating the default
            self._target_sample_rate = (
                max(self.detected_sample_rates)
                if self.detected_sample_rates
                else self._default_sample_rate
            )

            self._wav_file = wave.open(self.output_path, 'wb')
            self._wav_file.setnchannels(self.channels)
            self._wav_file.setsampwidth(2)  # 16-bit PCM
            self._wav_file.setframerate(self._target_sample_rate)
            self._is_initialized = True
            logger.debug(f"Initialized WAV file at {self.output_path} with sample rate {self._target_sample_rate}")
        except Exception as e:
            logger.error(f"Failed to initialize WAV file {self.output_path}: {e}")

    def _flush_buffer(self):
        """Flush buffered frames to disk and clear buffer."""
        if not self.audio_frames or not self._wav_file:
            return

        try:
            for audio_data, frame_sample_rate in self.audio_frames:
                # Resample if needed
                if frame_sample_rate and frame_sample_rate != self._target_sample_rate:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    num_samples = int(len(audio_array) * self._target_sample_rate / frame_sample_rate)
                    resampled = signal.resample(audio_array, num_samples)
                    audio_data = resampled.astype(np.int16).tobytes()

                self._wav_file.writeframes(audio_data)

            # Clear buffer after writing to disk
            self.audio_frames.clear()
            logger.debug(f"Flushed audio buffer to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to flush audio buffer to {self.output_path}: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Capture all audio frames
        if isinstance(frame, AudioRawFrame):
            # Pipecat uses 'sample_rate' attribute on AudioRawFrame
            frame_sample_rate = getattr(frame, 'sample_rate', None)
            if frame_sample_rate:
                self.detected_sample_rates.add(frame_sample_rate)

            # Initialize file on first frame
            if not self._is_initialized:
                self._initialize_wav_file()

            # Buffer the frame
            self.audio_frames.append((frame.audio, frame_sample_rate))

            # Flush to disk if buffer is getting too large (prevents memory leak)
            if len(self.audio_frames) >= self.MAX_BUFFER_FRAMES:
                self._flush_buffer()

        await self.push_frame(frame, direction)

    def save_recording(self):
        """Flush any remaining buffered audio and close the WAV file."""
        if not self._is_initialized:
            logger.warning("No audio frames to save - WAV file was never initialized")
            return

        try:
            # Flush any remaining buffered frames
            self._flush_buffer()

            # Close the WAV file
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None

            logger.info(f"Recording saved to: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save recording to {self.output_path}: {e}")
        finally:
            self._is_initialized = False

