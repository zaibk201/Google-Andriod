# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Vertex AI message normalization (fixes #156).

Vertex AI rejects turns where functionResponse parts are followed by user prompt
or observation text with HTTP 400 ("Requests ending with a model turn are not supported").
These tests verify that outbound messages for Vertex AI are normalized so that
human text/observations precede tool responses, without altering the underlying
conversation history ledger.
"""

import asyncio
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_vertexai.chat_models import _parse_chat_history_gemini

from artemis.llm.google import is_vertex_provider
from artemis.llm.router import ModelEndpoint, ModelProvider
from artemis.services.llm import (
    RobustChatModelWrapper,
    acomplete,
    normalize_vertex_messages,
)


def test_is_vertex_provider():
    assert is_vertex_provider("vertexai") is True
    assert is_vertex_provider("vertex") is True
    assert is_vertex_provider("VERTEX_AI") is True
    assert is_vertex_provider(ModelProvider.VERTEX_AI) is True
    assert is_vertex_provider("google") is False
    assert is_vertex_provider("gemini") is False
    assert is_vertex_provider("openai") is False
    assert is_vertex_provider(None) is False


def test_normalize_empty_or_non_tool_messages():
    assert normalize_vertex_messages([]) == []

    single = [HumanMessage(content="Hello")]
    assert normalize_vertex_messages(single) == single

    conv_no_tools = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user turn 1"),
        AIMessage(content="model turn 1"),
        HumanMessage(content="user turn 2"),
        AIMessage(content="model turn 2"),
    ]
    norm = normalize_vertex_messages(conv_no_tools)
    assert norm == conv_no_tools
    assert norm is not conv_no_tools


def test_normalize_human_observation_before_tool_message():
    history = [
        SystemMessage(content="system"),
        HumanMessage(content="open settings"),
        AIMessage(
            content="",
            tool_calls=[{"name": "click", "args": {"x": 100, "y": 200}, "id": "call_1"}],
        ),
        ToolMessage(content="clicked element", tool_call_id="call_1"),
        HumanMessage(content="Observation: settings app opened"),
    ]

    # Save copy of original to verify immutability
    original_types = [type(m).__name__ for m in history]

    norm = normalize_vertex_messages(history)

    # Underlying history must NOT be mutated
    assert [type(m).__name__ for m in history] == original_types
    assert history[3].content == "clicked element"
    assert history[4].content == "Observation: settings app opened"

    # In normalized outbound copy, HumanMessage observation must precede ToolMessage
    assert len(norm) == 5
    assert isinstance(norm[0], SystemMessage)
    assert isinstance(norm[1], HumanMessage)
    assert isinstance(norm[2], AIMessage)
    assert isinstance(norm[3], HumanMessage)
    assert norm[3].content == "Observation: settings app opened"
    assert isinstance(norm[4], ToolMessage)
    assert norm[4].content == "clicked element"
    assert norm[4].name == "click"


def test_normalize_multi_tool_and_multi_turn():
    history = [
        SystemMessage(content="system"),
        HumanMessage(content="start workflow"),
        # Turn 1: 2 tool calls
        AIMessage(
            content="executing burst",
            tool_calls=[
                {"name": "click", "args": {"x": 50, "y": 50}, "id": "c1"},
                {"name": "type_text", "args": {"text": "hello"}, "id": "c2"},
            ],
        ),
        ToolMessage(content="ok", tool_call_id="c1"),
        ToolMessage(content="typed", tool_call_id="c2"),
        HumanMessage(content="Observation 1"),
        # Turn 2: 1 tool call
        AIMessage(
            content="",
            tool_calls=[{"name": "press_home", "args": {}, "id": "c3"}],
        ),
        ToolMessage(content="home pressed", tool_call_id="c3"),
        HumanMessage(content="Observation 2"),
    ]

    norm = normalize_vertex_messages(history)

    assert len(norm) == len(history)

    # In Turn 1, Observation 1 should precede tool messages c1 and c2
    assert isinstance(norm[3], HumanMessage)
    assert norm[3].content == "Observation 1"
    assert isinstance(norm[4], ToolMessage)
    assert norm[4].tool_call_id == "c1"
    assert norm[4].name == "click"
    assert isinstance(norm[5], ToolMessage)
    assert norm[5].tool_call_id == "c2"
    assert norm[5].name == "type_text"

    # In Turn 2, Observation 2 should precede tool message c3
    assert isinstance(norm[7], HumanMessage)
    assert norm[7].content == "Observation 2"
    assert isinstance(norm[8], ToolMessage)
    assert norm[8].tool_call_id == "c3"
    assert norm[8].name == "press_home"


def test_vertex_part_ordering_with_langchain_serializer():
    """Verify that normalized messages produce [text, function_response] in Vertex Content."""
    history = [
        HumanMessage(content="Open battery settings and report percentage"),
        AIMessage(
            content="",
            tool_calls=[{"name": "click_settings", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="settings opened", tool_call_id="call_1"),
        HumanMessage(content="Current screen: battery page at 84%"),
    ]

    # Without normalization: LangChain Vertex produces [function_response, text],
    # which Vertex Gemini 3 rejects with HTTP 400.
    _, raw_contents = _parse_chat_history_gemini(history, None)
    turn_raw = raw_contents[-1]
    assert turn_raw.role == "user"
    raw_part_types = [p._pb.WhichOneof("data") for p in turn_raw.parts]
    assert raw_part_types == ["function_response", "text"]

    # With normalization: outbound copy is reordered
    norm = normalize_vertex_messages(history)
    _, norm_contents = _parse_chat_history_gemini(norm, None)
    turn_norm = norm_contents[-1]
    assert turn_norm.role == "user"
    norm_part_types = [p._pb.WhichOneof("data") for p in turn_norm.parts]
    # Vertex requires text before function_response so function_response closes the turn
    assert norm_part_types == ["text", "function_response"]


@pytest.mark.asyncio
async def test_robust_chat_model_wrapper_normalizes_vertex_endpoint():
    class MockVertexModel:
        def __init__(self):
            self.last_messages = None

        async def astream(self, messages, *args, **kwargs):
            self.last_messages = messages
            yield AIMessageChunk(content="done")

        def invoke(self, messages, *args, **kwargs):
            self.last_messages = messages
            return AIMessage(content="done")

    endpoint = ModelEndpoint(
        provider=ModelProvider.VERTEX_AI,
        model_name="gemini-3.8-flash",
    )
    base = MockVertexModel()
    wrapper = RobustChatModelWrapper(base, endpoint=endpoint)

    history = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "click", "args": {}, "id": "c1"}]),
        ToolMessage(content="ok", tool_call_id="c1"),
        HumanMessage(content="observation"),
    ]

    # Test complete()
    await wrapper.complete(history)
    assert base.last_messages is not None
    assert isinstance(base.last_messages[2], HumanMessage)
    assert base.last_messages[2].content == "observation"
    assert isinstance(base.last_messages[3], ToolMessage)

    # Original history passed by caller was NOT modified
    assert isinstance(history[2], ToolMessage)
    assert isinstance(history[3], HumanMessage)

    # Test invoke()
    await asyncio.to_thread(wrapper.invoke, history)
    assert isinstance(base.last_messages[2], HumanMessage)
    assert isinstance(base.last_messages[3], ToolMessage)


@pytest.mark.asyncio
async def test_non_vertex_model_preserves_order():
    class MockOpenAIModel:
        def __init__(self):
            self.last_messages = None

        async def astream(self, messages, *args, **kwargs):
            self.last_messages = messages
            yield AIMessageChunk(content="done")

    endpoint = ModelEndpoint(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
    )
    base = MockOpenAIModel()
    wrapper = RobustChatModelWrapper(base, endpoint=endpoint)

    history = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "click", "args": {}, "id": "c1"}]),
        ToolMessage(content="ok", tool_call_id="c1"),
        HumanMessage(content="observation"),
    ]

    await wrapper.complete(history)
    # Non-vertex models must keep the original order: ToolMessage then HumanMessage
    assert isinstance(base.last_messages[2], ToolMessage)
    assert isinstance(base.last_messages[3], HumanMessage)


def test_normalize_orphan_tool_message_backfills_fallback_name():
    """Verify that orphan ToolMessages (no preceding AIMessage) have safe fallback names."""
    history = [
        HumanMessage(content="previous context"),
        ToolMessage(content="orphan result", tool_call_id="call_orphan_123"),
    ]

    norm = normalize_vertex_messages(history)

    # Underlying history remains untouched
    assert history[1].name is None

    # Outbound normalized copy gets tool_call_id as fallback name
    assert len(norm) == 2
    assert isinstance(norm[1], ToolMessage)
    assert norm[1].name == "call_orphan_123"

    # Serializer must produce valid function_response without crashing
    _, norm_contents = _parse_chat_history_gemini(norm, None)
    assert len(norm_contents) == 1
    assert norm_contents[0].parts[1].function_response.name == "call_orphan_123"


def test_normalize_unmatched_tool_call_id_in_turn():
    """Verify that a ToolMessage with an unmatched ID in a tool turn gets a safe fallback name."""
    history = [
        HumanMessage(content="run action"),
        AIMessage(
            content="",
            tool_calls=[{"name": "click", "args": {}, "id": "matched_id"}],
        ),
        ToolMessage(content="unmatched result", tool_call_id="unknown_tool_call"),
    ]

    norm = normalize_vertex_messages(history)

    # Underlying history remains untouched
    assert history[2].name is None

    # Falls back to tool_call_id safely rather than leaving name as None
    assert norm[2].name == "unknown_tool_call"


def test_normalize_tool_message_retains_existing_name():
    """Verify that a ToolMessage with an explicit name is preserved."""
    history = [
        HumanMessage(content="run action"),
        ToolMessage(content="result", tool_call_id="c1", name="custom_tool"),
    ]

    norm = normalize_vertex_messages(history)
    assert norm[1].name == "custom_tool"
