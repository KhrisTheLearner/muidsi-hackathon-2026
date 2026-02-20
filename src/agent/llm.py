"""Archia-backed LLM client.

Archia uses the OpenAI Responses API (/v1/responses), NOT the Chat
Completions API (/v1/chat/completions). This module wraps the Responses
API as a LangChain-compatible ChatModel so it works with LangGraph.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import OpenAI

load_dotenv()

ARCHIA_TOKEN = os.getenv("ARCHIA_TOKEN", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "priv-claude-sonnet-4-5-20250929")
ARCHIA_BASE_URL = "https://registry.archia.app/v1"

# Model routing: specialized agents use smaller tool sets = fewer tokens.
# Haiku 3.5 for simple fetch/format tasks, Sonnet for complex reasoning.
FAST_MODEL = os.getenv("FAST_MODEL", "priv-claude-haiku-4-5-20251001")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", FAST_MODEL)
DATA_MODEL = os.getenv("DATA_MODEL", FAST_MODEL)
VIZ_MODEL = os.getenv("VIZ_MODEL", FAST_MODEL)
LOGISTICS_MODEL = os.getenv("LOGISTICS_MODEL", FAST_MODEL)
ML_MODEL = os.getenv("ML_MODEL", DEFAULT_MODEL)
SQL_MODEL = os.getenv("SQL_MODEL", os.getenv("ANALYST_MODEL", DEFAULT_MODEL))
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", DEFAULT_MODEL)


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client pointed at Archia."""
    return OpenAI(
        base_url=ARCHIA_BASE_URL,
        api_key="not-used",
        default_headers={"Authorization": f"Bearer {ARCHIA_TOKEN}"},
        timeout=60.0,  # 60s max — prevents hanging on Archia 504s
    )


class ArchiaChatModel(BaseChatModel):
    """LangChain ChatModel that uses Archia's Responses API."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    tools: list[dict] | None = None

    @property
    def _llm_type(self) -> str:
        return "archia"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = _get_openai_client()

        # Convert LangChain messages to Responses API input
        input_items = _messages_to_input(messages)

        # Build request kwargs
        req: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "temperature": self.temperature,
        }

        # Add tools if bound
        tools = kwargs.get("tools") or self.tools
        if tools:
            req["tools"] = tools

        try:
            response = client.responses.create(**req)
        except Exception as e:
            # Return error as AI message so agent can recover gracefully
            error_msg = f"Archia API error: {type(e).__name__}: {e}"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=error_msg))]
            )

        # Parse response output into AIMessage
        ai_message = _parse_response(response)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(self, tools: list, **kwargs) -> ArchiaChatModel:
        """Bind tools in OpenAI function-calling format."""
        formatted = []
        for tool in tools:
            schema = tool.args_schema.model_json_schema() if hasattr(tool, "args_schema") and tool.args_schema else {}
            # Clean up schema for OpenAI format
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # Remove 'title' fields that langchain adds
            clean_props = {}
            for k, v in properties.items():
                clean_props[k] = {pk: pv for pk, pv in v.items() if pk != "title"}

            formatted.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": clean_props,
                    "required": required,
                },
            })

        # Return a new instance with tools set
        return ArchiaChatModel(
            model=self.model,
            temperature=self.temperature,
            tools=formatted,
        )


def _messages_to_input(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to Archia Responses API input format.

    NOTE: The Archia API hangs when the "developer" role is used. We work around
    this by injecting system message content as a prefix in the first user message.
    """
    # Extract system message content first
    system_content = ""
    non_system = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_content = msg.content
        else:
            non_system.append(msg)

    items = []
    first_user_done = False
    for msg in non_system:
        if isinstance(msg, HumanMessage):
            if not first_user_done and system_content:
                # Prepend system context into the first user message
                content = f"[System instructions]\n{system_content}\n\n[User query]\n{msg.content}"
                first_user_done = True
            else:
                content = msg.content
            items.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Reconstruct function call outputs
                for tc in msg.tool_calls:
                    items.append({
                        "type": "function_call",
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    })
            else:
                items.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
            })
    return items


def _parse_response(response) -> AIMessage:
    """Parse Archia Responses API output into a LangChain AIMessage."""
    content = ""
    tool_calls = []

    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if hasattr(block, "text"):
                    content += block.text
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({
                "id": item.call_id,
                "name": item.name,
                "args": args,
            })

    return AIMessage(content=content, tool_calls=tool_calls)


def get_llm(model: str | None = None, temperature: float = 0.0) -> ArchiaChatModel:
    """Create an ArchiaChatModel instance.

    Args:
        model: Model ID (e.g. "priv-claude-sonnet-4-5-20250929").
        temperature: Sampling temperature.

    Returns:
        ArchiaChatModel configured for Archia.
    """
    if not ARCHIA_TOKEN:
        raise ValueError(
            "ARCHIA_TOKEN not set. Get your token from "
            "console.archia.app -> MUIDSI Hackathon 2026 -> API Keys, "
            "then add it to your .env file."
        )

    return ArchiaChatModel(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
    )
