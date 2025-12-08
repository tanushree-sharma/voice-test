import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from opentelemetry.sdk.trace import TracerProvider

from langsmith_processor import LangSmithSpanProcessor

# Try loading .env.local first, then .env, then check parent directory
script_dir = Path(__file__).parent
env_loaded = False
env_files_tried = []

for env_file in [".env.local", ".env"]:
    env_path = script_dir / env_file
    if env_path.exists():
        try:
            load_dotenv(env_path, override=True)
            env_loaded = True
            break
        except Exception as e:
            print(
                f"⚠️  Warning: Error loading {env_path}: {e}\n"
                f"   Continuing without loading this file...",
                file=__import__("sys").stderr
            )
            env_files_tried.append(str(env_path))

# Also try parent directory
if not env_loaded:
    parent_env = script_dir.parent / ".env.local"
    if parent_env.exists():
        try:
            load_dotenv(parent_env, override=True)
            env_loaded = True
        except Exception as e:
            print(
                f"⚠️  Warning: Error loading {parent_env}: {e}\n"
                f"   Continuing without loading this file...",
                file=__import__("sys").stderr
            )
            env_files_tried.append(str(parent_env))
    else:
        parent_env = script_dir.parent / ".env"
        if parent_env.exists():
            try:
                load_dotenv(parent_env, override=True)
                env_loaded = True
            except Exception as e:
                print(
                    f"⚠️  Warning: Error loading {parent_env}: {e}\n"
                    f"   Continuing without loading this file...",
                    file=__import__("sys").stderr
                )
                env_files_tried.append(str(parent_env))

# Fallback: try loading from current directory without specifying file
if not env_loaded:
    try:
        load_dotenv(override=True)
    except Exception as e:
        print(
            f"⚠️  Warning: Error loading .env file: {e}\n"
            f"   Continuing without environment file...",
            file=__import__("sys").stderr
        )


def setup_langsmith():
    """Setup OpenTelemetry tracing to export spans to LangSmith."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

    if not endpoint or not headers:
        print(
            "⚠️  Warning: OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS not set. "
            "LangSmith tracing will be disabled. To enable, set these environment variables:\n"
            "  - OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel\n"
            "  - OTEL_EXPORTER_OTLP_HEADERS=x-api-key=your_langsmith_api_key\n"
            "You can set them in a .env.local or .env file in this directory.",
            file=__import__("sys").stderr
        )
        return

    trace_provider = TracerProvider()
    # Register LangSmith processor (handles enrichment + forwarding to OTLP)
    trace_provider.add_span_processor(LangSmithSpanProcessor())
    set_tracer_provider(trace_provider)
    print("✅ LangSmith tracing enabled", file=__import__("sys").stderr)


# Setup LangSmith tracing before creating the server (optional if env vars not set)
setup_langsmith()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    # Run in console mode (local terminal demo)
    # Override sys.argv to always use console mode, ignoring any command-line args
    sys.argv = [sys.argv[0], "console"]
    agents.cli.run_app(server)