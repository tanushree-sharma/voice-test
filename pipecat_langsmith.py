"""
Pipecat + LangSmith Demo: Voice Agent with Full Observability

This demo shows how to build a voice agent using Pipecat and send telemetry to LangSmith
for observability. The agent uses Whisper for STT, OpenAI for LLM/TTS, and records conversations.

Core flow: Audio input -> STT -> LLM -> TTS -> Audio output

Setup: Configure these environment variables in your .env file:
- OPENAI_API_KEY: Your OpenAI API key
- OTEL_EXPORTER_OTLP_ENDPOINT: LangSmith OTLP endpoint
- OTEL_EXPORTER_OTLP_HEADERS: LangSmith auth headers (x-api-key=your_key)
"""
import asyncio
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Load environment variables FIRST (before any imports that need them)
load_dotenv(override=True)

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='faster_whisper')
np.seterr(divide='ignore', invalid='ignore')

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# Import supporting modules (setup_tracing runs during import, so env vars must be loaded first)
from langsmith_processor import LangSmithSTTSpanProcessor  # noqa: F401 - registers processor
from audio_recorder import AudioRecorder

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    """
    Main demo function showing how to build a voice agent with Pipecat and LangSmith observability.
    Flow: Audio input -> STT -> LLM -> TTS -> Audio output
    """
    # Generate unique conversation ID (used for grouping spans in LangSmith)
    conversation_id = str(uuid.uuid4())
    logger.info(f"Starting conversation: {conversation_id}")

    # Setup recording directory
    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_path = recordings_dir / f"conversation_{timestamp}.wav"

    # Configure local audio transport with voice activity detection
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Initialize services (API keys read from .env)
    stt = WhisperSTTService()
    llm = OpenAILLMService(model="gpt-4o-mini")
    tts = OpenAITTSService(voice="alloy")

    # Define the system prompt and conversation context
    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": """You are an expert French tutor coach. Your role is to help students learn and practice French by having natural conversations in French with students at their level. Assume the user is a beginner and start simple. Keep your responses short, let the user guide the conversation. Assume the user is speaking in french and if you cant understand then ask them to speak louder and slower."""
            }
        ]
    )
    context_aggregator = llm.create_context_aggregator(context)
    audio_recorder = AudioRecorder(str(recording_path))

    # Build the pipeline: audio flows through each processor in order
    pipeline = Pipeline([
        transport.input(),           # Capture microphone input
        stt,                         # Transcribe audio to text
        context_aggregator.user(),   # Add user message to context
        llm,                         # Generate AI response
        tts,                         # Convert response to speech
        audio_recorder,              # Record all audio
        transport.output(),          # Play audio through speakers
        context_aggregator.assistant(),  # Add assistant message to context
    ])

    # Create task with tracing enabled for LangSmith observability
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,  # Required when tracing is enabled
        conversation_id=conversation_id,
    )

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    try:
        await runner.run(task)
    finally:
        audio_recorder.save_recording()
        logger.info(f"Recording saved to: {recording_path}")


if __name__ == "__main__":
    asyncio.run(main())
