# Live Audio Tracing Demos

This repo hosts two LangSmith-integrated audio agents:

| Demo | Stack | Entry Script | Span Processor |
| --- | --- | --- | --- |
| `livekit-test/` | LiveKit Agents | `livekit_test/livekitagents_langsmith.py` | `livekit-test/langsmith_processor.py` |
| `pipecat-test/` | Pipecat | `pipecat-test/pipecat_langsmith.py` | `pipecat-test/langsmith_processor.py` |

Both pipelines stream STT → LLM → TTS and export OpenTelemetry spans enriched with prompts, completions, and conversation summaries so LangSmith shows a clean trace tree.

## Prerequisites

- Python 3.13
- Virtualenv or similar (recommended)
- Provider API keys (AssemblyAI, OpenAI, Cartesia, etc.)
- LiveKit RTC + API keys for the LiveKit demo
- LangSmith API key (used by both demos)

Create a `.env.local` file at the repo root with the following required environment variables:

```
OPENAI_API_KEY=<api-key>
# other API keys, depending on providers used
LANGSMITH_OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=<api-key>,Langsmith-Project=pipecat-voice
# if using LiveKit
LIVEKIT_API_KEY=<api-key>
LIVEKIT_API_SECRET=<secret>
LIVEKIT_URL=wss://demo-xxxxx.livekit.cloud
```

## Install Dependencies

The repo includes a fully pinned environment snapshot. After cloning:

```bash
git clone <repo-url>
cd audio

python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If you prefer to install from `pyproject.toml`, run `pip install -e .`, but the requirements file guarantees you match the exact environment used to build the demos.

## LiveKit Demo (`livekit-test/`)

```
source .venv/bin/activate
LANGSMITH_PROCESSOR_DEBUG=true python livekit-test/livekitagents_langsmith.py
```

- Forces console mode so you can talk via your microphone in the terminal.
- Uses AssemblyAI universal streaming STT, OpenAI `gpt-4.1-mini`, Cartesia Sonic TTS, Silero VAD, and the multilingual turn detector.
- The custom span processor aggregates:
  - Full prompts/completions on every `llm_node`, `tts_node`, etc.
  - Conversation summaries on both `agent_session` and `job_entrypoint` spans (the root is deferred until the flow completes).

Open your LangSmith workspace → latest trace to see the entire hierarchy. Set `LANGSMITH_PROCESSOR_DEBUG=false` if you don’t want console logs.

## Pipecat Demo (`pipecat-test/`)

```
source .venv/bin/activate
python pipecat-test/pipecat_langsmith.py
```

- Demonstrates the same LangSmith instrumentation pattern inside a Pipecat pipeline.
- The Pipecat span processor captures STT prompts, LLM calls, TTS synthesis, and per-turn summaries.
- Recordings saved under `pipecat-test/recordings/` are automatically attached to the root span when available.

## Troubleshooting

- **No spans in LangSmith** – check `OTEL_EXPORTER_OTLP_*` env vars; the scripts warn if tracing is disabled.
- **Console exits immediately** – LiveKit’s CLI needs a TTY; run inside Terminal/iTerm, not inside certain IDE consoles. Grant microphone access if prompted.
- **Missing dependencies** – regenerate the environment: `rm -rf .venv && python3.13 -m venv .venv && pip install -r requirements.txt`.
- **Different API vendors** – swap STT/LLM/TTS providers in the scripts; the span processors rely only on the OpenTelemetry attributes, so they’ll adapt as long as the spans include `lk.*` or the Pipecat equivalents.

## Project Structure

```
audio/
├── livekit-test/
│   ├── langsmith_processor.py
│   └── livekitagents_langsmith.py
├── pipecat-test/
│   ├── langsmith_processor.py
│   ├── pipecat_langsmith.py
│   └── recordings/
├── requirements.txt
├── pyproject.toml
└── README.md
```

Feel free to extend either processor (attachments, extra metadata, custom evaluators) and re-run the demos. Everything flows through LangSmith with consistent span semantics, so you can compare the two stacks side by side.

