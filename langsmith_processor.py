"""
LangSmith span processor for Pipecat.

Enriches OpenTelemetry spans from Pipecat with LangSmith-compatible attributes
for proper conversation tracking and visualization.
"""
import json
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan, TracerProvider
from opentelemetry import trace
from pipecat.utils.tracing.setup import setup_tracing


class LangSmithSTTSpanProcessor(SpanProcessor):
    """
    Custom OpenTelemetry span processor that enriches Pipecat spans with LangSmith-compatible attributes.
    This enables proper conversation tracking and message visualization in LangSmith's UI.
    """

    def __init__(self):
        super().__init__()
        # Track conversation messages across spans for proper LangSmith grouping
        self.conversation_messages = {}  # trace_id -> list of messages
        self.turn_messages = {}  # parent_span_id -> list of messages
        self.trace_to_conversation_id = {}  # trace_id -> conversation_id

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        pass
    
    def on_end(self, span: ReadableSpan) -> None:
        """
        Enriches spans with LangSmith-compatible attributes before they're exported.
        Maps Pipecat span types (stt, llm, tts, turn, conversation) to LangSmith's expected format.
        """
        # Track each conversation as a thread in LangSmith
        trace_id = format(span.context.trace_id, '032x')
        span._attributes["langsmith.metadata.thread_id"] = trace_id

        # Link all spans to their conversation for proper grouping in LangSmith
        if trace_id in self.trace_to_conversation_id:
            conversation_id = self.trace_to_conversation_id[trace_id]
            span._attributes["conversation.id"] = conversation_id
            span._attributes["langsmith.parent_span_id"] = "conversation"
        
        # STT span: audio input -> transcribed text
        if span.name == "stt":
            transcript = span.attributes.get("transcript", "")
            span._attributes["langsmith.span.kind"] = "llm"
            self._set_prompt_attributes(span, [{"role": "user", "content": "audio_segment"}])
            self._set_completion_attributes(span, [{"role": "assistant", "content": transcript}])
        
        # LLM span: conversation messages -> AI response
        elif span.name == "llm":
            input_data = span.attributes.get("input", "")
            output_data = span.attributes.get("output", "")
            span._attributes["langsmith.span.kind"] = "llm"

            # Parse and add input messages
            messages = []
            try:
                messages = json.loads(input_data)
                self._set_prompt_attributes(span, messages)
            except json.JSONDecodeError:
                pass

            # Add LLM output
            if output_data:
                self._set_completion_attributes(span, [{"role": "assistant", "content": output_data}])

            # Track messages for aggregating into turn and conversation spans
            parent_span_id = format(span.parent.span_id, '016x') if span.parent else None
            self._track_messages(self.conversation_messages, trace_id, messages, output_data)
            if parent_span_id:
                self._track_messages(self.turn_messages, parent_span_id, messages, output_data)
        
        # TTS span: text -> audio
        elif span.name == "tts":
            text = span.attributes.get("text", "")
            voice_id = span.attributes.get("voice_id", "")
            span._attributes["langsmith.span.kind"] = "llm"
            self._set_prompt_attributes(span, [
                {"role": "system", "content": f"Convert to speech with voice: {voice_id}"},
                {"role": "user", "content": text}
            ])
            self._set_completion_attributes(span, [{"role": "assistant", "content": f"Generated audio for: {text}"}])
        
        # Turn span: represents a single user-assistant interaction
        elif span.name == "turn":
            turn_number = span.attributes.get("turn.number", 0)
            was_interrupted = span.attributes.get("turn.was_interrupted", False)
            span._attributes["langsmith.span.kind"] = "chain"

            # Aggregate messages from this turn's child spans
            span_id = format(span.context.span_id, '016x')
            turn_msgs = self.turn_messages.get(span_id, [])
            user_msgs = self._get_messages_by_role(turn_msgs, "user")
            assistant_msgs = self._get_messages_by_role(turn_msgs, "assistant")

            # Add user input(s)
            if user_msgs:
                self._set_prompt_attributes(span, user_msgs)
            else:
                self._set_prompt_attributes(span, [{"role": "user", "content": f"Turn {turn_number}"}])

            # Add assistant response(s)
            if assistant_msgs:
                self._set_completion_attributes(span, assistant_msgs)
            else:
                status = "interrupted" if was_interrupted else "no response"
                self._set_completion_attributes(span, [{"role": "assistant", "content": status}])

            # Cleanup
            if span_id in self.turn_messages:
                del self.turn_messages[span_id]
        
        # Conversation span: represents the entire conversation session
        elif span.name == "conversation":
            conversation_id = span.attributes.get("conversation.id", "")
            conversation_type = span.attributes.get("conversation.type", "voice")

            # Store conversation_id for linking child spans
            self.trace_to_conversation_id[trace_id] = conversation_id

            span._attributes["langsmith.span.kind"] = "chain"
            span._attributes["langsmith.root_span"] = True

            # Aggregate all messages from the conversation
            conv_msgs = self.conversation_messages.get(trace_id, [])

            if conv_msgs:
                system_msg, first_user_msg, remaining_msgs = self._split_conversation_messages(conv_msgs)

                # Add input (system + first user message)
                prompt_msgs = []
                if system_msg:
                    prompt_msgs.append(system_msg)
                if first_user_msg:
                    prompt_msgs.append(first_user_msg)
                self._set_prompt_attributes(span, prompt_msgs)

                # Add output (remaining conversation)
                if remaining_msgs:
                    self._set_completion_attributes(span, remaining_msgs)
                else:
                    self._set_completion_attributes(span, [{"role": "assistant", "content": "No responses yet"}])
            else:
                self._set_prompt_attributes(span, [{"role": "system", "content": f"Starting {conversation_type} conversation"}])
                self._set_completion_attributes(span, [{"role": "assistant", "content": "No messages"}])

            # Cleanup
            if trace_id in self.conversation_messages:
                del self.conversation_messages[trace_id]
            if trace_id in self.trace_to_conversation_id:
                del self.trace_to_conversation_id[trace_id]
    
    def _set_prompt_attributes(self, span: ReadableSpan, messages: list, start_idx: int = 0):
        """Set gen_ai.prompt.* attributes from a list of messages."""
        for i, msg in enumerate(messages):
            idx = start_idx + i
            span._attributes[f"gen_ai.prompt.{idx}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.prompt.{idx}.content"] = msg.get("content", "")
    
    def _set_completion_attributes(self, span: ReadableSpan, messages: list, start_idx: int = 0):
        """Set gen_ai.completion.* attributes from a list of messages."""
        for i, msg in enumerate(messages):
            idx = start_idx + i
            span._attributes[f"gen_ai.completion.{idx}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.completion.{idx}.content"] = msg.get("content", "")
    
    def _get_messages_by_role(self, messages: list, role: str) -> list:
        """Filter messages by role."""
        return [msg for msg in messages if msg.get("role") == role]
    
    def _split_conversation_messages(self, messages: list) -> tuple:
        """
        Split conversation messages into system, first user, and remaining messages.
        Returns: (system_msg, first_user_msg, remaining_msgs)
        """
        system_msg = None
        first_user_msg = None
        remaining_msgs = []
        first_user_found = False

        for msg in messages:
            role = msg.get("role", "")
            if role == "system" and system_msg is None:
                system_msg = msg
            elif role == "user" and not first_user_found:
                first_user_msg = msg
                first_user_found = True
            elif first_user_found:
                remaining_msgs.append(msg)

        return (system_msg, first_user_msg, remaining_msgs)
    
    def _track_messages(self, target_dict: dict, key: str, messages: list, output_data: str):
        """
        Track messages in target_dict, avoiding duplicates.
        Preserves all deduplication logic: case-insensitive content comparison,
        system prompt handling, and duplicate detection.
        """
        if key not in target_dict:
            target_dict[key] = []
            # Add system prompt once at the start
            for msg in messages:
                if msg.get("role") == "system":
                    target_dict[key].append(msg)
                    break

        # Add the latest user message if it's new
        last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_msg:
            new_content = last_user_msg.get("content", "").strip().lower()
            existing_contents = [m.get("content", "").strip().lower()
                                for m in target_dict[key] if m.get("role") == "user"]
            if new_content and new_content not in existing_contents:
                target_dict[key].append(last_user_msg)

        # Add the assistant response if it's new
        if output_data:
            new_assistant_content = output_data.strip().lower()
            existing_assistant_contents = [m.get("content", "").strip().lower()
                                          for m in target_dict[key] if m.get("role") == "assistant"]
            if new_assistant_content not in existing_assistant_contents:
                target_dict[key].append({"role": "assistant", "content": output_data})

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# Setup OpenTelemetry tracing with LangSmith
# Configure OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS in your .env
setup_tracing(
    service_name="pipecat-langsmith-demo",
    exporter=OTLPSpanExporter(),
    console_export=True,
)

# Register our custom span processor to enrich Pipecat spans for LangSmith
tracer_provider = trace.get_tracer_provider()
if isinstance(tracer_provider, TracerProvider):
    tracer_provider.add_span_processor(LangSmithSTTSpanProcessor())

