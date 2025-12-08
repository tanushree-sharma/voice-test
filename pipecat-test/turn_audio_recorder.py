"""
Turn-aware audio recorder for capturing and saving per-turn audio snippets.

Records separate audio files for user speech and AI responses for each conversation turn,
enabling fine-grained audio analysis in LangSmith.
"""
import wave
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import signal
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnAudioRecorder(FrameProcessor):
    """
    Frame processor that captures user and AI audio separately per turn.

    Subscribes to TurnTrackingObserver events to detect turn boundaries and saves
    separate WAV files for user and AI audio when each turn ends.

    Uses bounded buffers to prevent memory issues on very long turns.
    """

    # Maximum frames to buffer per turn per source (prevents memory leak on long turns)
    MAX_BUFFER_FRAMES = 1000  # ~20 seconds at typical frame rates

    def __init__(
        self,
        span_processor,
        conversation_id: str,
        recordings_dir: Path,
        turn_tracker=None,
        user_sample_rate: int = 16000,
        ai_sample_rate: int = 24000,
        channels: int = 1,
    ):
        """
        Initialize turn audio recorder.

        Args:
            span_processor: LangSmithSTTSpanProcessor instance for registering recordings
            conversation_id: Unique conversation identifier
            recordings_dir: Directory to save audio files
            turn_tracker: TurnTrackingObserver instance (can be set later via connect_to_turn_tracker)
            user_sample_rate: Default sample rate for user audio (16kHz typical for mic)
            ai_sample_rate: Default sample rate for AI audio (24kHz typical for TTS)
            channels: Number of audio channels (1 for mono)
        """
        super().__init__()
        self._span_processor = span_processor
        self._conversation_id = conversation_id
        self._recordings_dir = Path(recordings_dir)
        self._turn_tracker = turn_tracker
        self._channels = channels

        # Current turn state
        self._current_turn_number = 0
        self._is_turn_active = False

        # Buffers for current turn: [(audio_data, sample_rate), ...]
        self._current_user_frames = []
        self._current_ai_frames = []

        # Track detected sample rates separately for user and AI
        self._user_detected_rates = set()
        self._ai_detected_rates = set()

        # Default sample rates
        self._default_user_rate = user_sample_rate
        self._default_ai_rate = ai_sample_rate

        # Map turn numbers to saved file paths
        self._turn_recordings = {}  # turn_number -> {user: path, ai: path}

        # Ensure recordings directory exists
        self._recordings_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TurnAudioRecorder initialized for conversation {conversation_id}"
        )

    def connect_to_turn_tracker(self, turn_tracker):
        """
        Connect to turn tracker and register event handlers.

        This method should be called after the PipelineTask is created,
        as the TurnTrackingObserver is created by the task.

        Args:
            turn_tracker: TurnTrackingObserver instance
        """
        self._turn_tracker = turn_tracker

        # Register event handlers using wrapper methods
        turn_tracker.add_event_handler("on_turn_started", self._on_turn_started_wrapper)
        turn_tracker.add_event_handler("on_turn_ended", self._on_turn_ended_wrapper)

        logger.info(
            f"TurnAudioRecorder connected to TurnTrackingObserver for conversation {self._conversation_id}"
        )

    async def _on_turn_started_wrapper(self, observer, turn_number: int):
        """
        Wrapper for turn started event handler.
        Event handlers receive (observer, *args) - we ignore observer and delegate to handler.

        Args:
            observer: TurnTrackingObserver instance (ignored)
            turn_number: The turn number that just started
        """
        await self._handle_turn_started(turn_number)

    async def _on_turn_ended_wrapper(self, observer, turn_number: int, duration: float, was_interrupted: bool):
        """
        Wrapper for turn ended event handler.
        Event handlers receive (observer, *args) - we ignore observer and delegate to handler.

        Args:
            observer: TurnTrackingObserver instance (ignored)
            turn_number: The turn number that just ended
            duration: Duration of the turn in seconds
            was_interrupted: Whether the turn was interrupted by user
        """
        await self._handle_turn_ended(turn_number, duration, was_interrupted)

    async def _handle_turn_started(self, turn_number: int):
        """
        Handle turn started event.

        Resets buffers and prepares for new turn audio capture.
        Buffer clearing here prevents unbounded memory growth across turns.

        Args:
            turn_number: The turn number that just started
        """
        self._current_turn_number = turn_number
        self._is_turn_active = True
        # Clear buffers from previous turn (prevents memory accumulation)
        self._current_user_frames = []
        self._current_ai_frames = []
        self._user_detected_rates.clear()
        self._ai_detected_rates.clear()

    async def _handle_turn_ended(
        self, turn_number: int, duration: float, was_interrupted: bool
    ):
        """
        Handle turn ended event.

        Args:
            turn_number: The turn number that just ended
            duration: Duration of the turn in seconds
            was_interrupted: Whether the turn was interrupted by user
        """
        self._is_turn_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process audio frames and buffer them by type (user vs AI).

        Args:
            frame: The frame to process
            direction: Direction of frame flow
        """
        await super().process_frame(frame, direction)

        # Only capture audio frames when turn is active
        if isinstance(frame, AudioRawFrame) and self._is_turn_active:
            # Pipecat uses 'sample_rate' attribute on AudioRawFrame
            frame_sample_rate = getattr(frame, 'sample_rate', None)

            # Distinguish user audio from AI audio
            if isinstance(frame, InputAudioRawFrame):
                # User audio from microphone
                if frame_sample_rate:
                    self._user_detected_rates.add(frame_sample_rate)

                # Enforce buffer limit to prevent memory leak on very long turns
                if len(self._current_user_frames) < self.MAX_BUFFER_FRAMES:
                    self._current_user_frames.append((frame.audio, frame_sample_rate))
                elif len(self._current_user_frames) == self.MAX_BUFFER_FRAMES:
                    logger.warning(
                        f"User audio buffer limit reached for turn {self._current_turn_number}. "
                        f"Dropping additional frames to prevent memory leak."
                    )

            elif isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
                # AI/TTS audio output
                if frame_sample_rate:
                    self._ai_detected_rates.add(frame_sample_rate)

                # Enforce buffer limit to prevent memory leak on very long turns
                if len(self._current_ai_frames) < self.MAX_BUFFER_FRAMES:
                    self._current_ai_frames.append((frame.audio, frame_sample_rate))
                elif len(self._current_ai_frames) == self.MAX_BUFFER_FRAMES:
                    logger.warning(
                        f"AI audio buffer limit reached for turn {self._current_turn_number}. "
                        f"Dropping additional frames to prevent memory leak."
                    )

        await self.push_frame(frame, direction)

    def save_turn_audio_sync(self, turn_number: int):
        """
        Synchronously save audio files for the given turn number.
        Called directly by the span processor when the turn span ends.

        Args:
            turn_number: The turn number to save

        Returns:
            Dict with 'user' and/or 'ai' keys pointing to saved file paths, or empty dict
        """
        saved_files = {}

        # Only save if this is the current turn
        if turn_number != self._current_turn_number:
            return saved_files

        try:
            # Save user audio if exists
            if self._current_user_frames:
                user_path = self._recordings_dir / f"turn_{turn_number}_user.wav"
                sample_rate = (
                    max(self._user_detected_rates)
                    if self._user_detected_rates
                    else self._default_user_rate
                )
                self._save_wav_file(user_path, self._current_user_frames, sample_rate)
                saved_files['user'] = str(user_path)
                logger.debug(f"Saved user audio for turn {turn_number} to {user_path}")

            # Save AI audio if exists
            if self._current_ai_frames:
                ai_path = self._recordings_dir / f"turn_{turn_number}_ai.wav"
                sample_rate = (
                    max(self._ai_detected_rates)
                    if self._ai_detected_rates
                    else self._default_ai_rate
                )
                self._save_wav_file(ai_path, self._current_ai_frames, sample_rate)
                saved_files['ai'] = str(ai_path)
                logger.debug(f"Saved AI audio for turn {turn_number} to {ai_path}")

        except Exception as e:
            logger.error(f"Failed to save turn {turn_number} audio in conversation {self._conversation_id}: {e}")

        return saved_files

    def _save_wav_file(
        self, output_path: Path, frames: list, target_sample_rate: int
    ):
        """
        Save audio frames to a WAV file with resampling if needed.

        Args:
            output_path: Path to save the WAV file
            frames: List of (audio_data, frame_sample_rate) tuples
            target_sample_rate: Target sample rate for the output file
        """
        if not frames:
            return

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(target_sample_rate)

            for audio_data, frame_sample_rate in frames:
                # Resample if needed
                if frame_sample_rate and frame_sample_rate != target_sample_rate:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    num_samples = int(
                        len(audio_array) * target_sample_rate / frame_sample_rate
                    )
                    resampled = signal.resample(audio_array, num_samples)
                    audio_data = resampled.astype(np.int16).tobytes()

                wav_file.writeframes(audio_data)
