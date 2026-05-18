import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from langfuse import observe, Langfuse


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[dict] = field(default_factory=list)
    model_name: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}


class LLMClient:
    """Bedrock Anthropic client behind the app's internal LLM boundary."""

    def __init__(self, settings, client=None):
        self.settings = settings
        self.provider = "bedrock"
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self._client = client or anthropic.AsyncAnthropicBedrock(
            aws_region=settings.AWS_REGION,
            timeout=self.timeout,
        )

    @observe(as_type="generation")
    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        system_msg, chat_messages = self._to_anthropic_messages(messages)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
            "temperature": self.temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [self._to_anthropic_tool(tool) for tool in tools]

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.messages.create(**kwargs)
                result = self._parse_response(response)
                Langfuse().update_current_generation(
                    model=self.model,
                    usage_details={
                        "input": result.prompt_tokens,
                        "output": result.completion_tokens,
                    },
                )
                return result
            except anthropic.APITimeoutError as e:
                raise LLMError("The model provider did not respond in time.") from e
            except anthropic.APIStatusError as e:
                if (
                    e.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    await asyncio.sleep((2**attempt) + random.uniform(0, 1))
                    continue
                raise LLMError(f"LLM provider error (status {e.status_code})") from e
            except Exception as e:
                raise LLMError(f"Unexpected LLM error: {e}") from e

        raise LLMError("LLM provider unavailable after retries")

    def _to_anthropic_messages(self, messages: list) -> tuple[str, list[dict]]:
        system_msg = ""
        chat_messages = []
        for message in messages:
            role = self._role(message)
            content = self._content(message)

            if role == "system":
                system_msg = content
            elif role == "assistant":
                tool_calls = self._tool_calls(message)
                if tool_calls:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call["id"],
                                    "name": call["function"]["name"],
                                    "input": self._json_args(
                                        call["function"].get("arguments", "{}")
                                    ),
                                }
                                for call in tool_calls
                            ],
                        }
                    )
                else:
                    chat_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                chat_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": self._field(
                                    message, "tool_call_id", "unknown"
                                ),
                                "content": content,
                            }
                        ],
                    }
                )
            elif role == "user":
                chat_messages.append({"role": "user", "content": content})

        return system_msg, chat_messages

    def _parse_response(self, response) -> LLMResponse:
        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        usage = response.usage
        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            model_name=response.model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
        )

    def _role(self, message) -> str:
        role = self._field(message, "role")
        if role:
            return role
        return {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }.get(self._field(message, "type"), "")

    def _content(self, message) -> str:
        content = self._field(message, "content")
        if isinstance(content, list):
            return " ".join(
                str(block.get("text", "") if isinstance(block, dict) else block.text)
                for block in content
            )
        return content or ""

    def _tool_calls(self, message) -> list[dict]:
        calls = self._field(message, "tool_calls") or []
        normalized = []
        for call in calls:
            if isinstance(call, dict) and "function" in call:
                normalized.append(call)
            elif isinstance(call, dict):
                normalized.append(
                    {
                        "id": call.get("id", "unknown"),
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                )
            else:
                normalized.append(
                    {
                        "id": getattr(call, "id", "unknown"),
                        "function": {
                            "name": getattr(call, "name", ""),
                            "arguments": json.dumps(getattr(call, "args", {})),
                        },
                    }
                )
        return normalized

    def _to_anthropic_tool(self, tool: dict) -> dict:
        func = tool.get("function", {})
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }

    def _json_args(self, value):
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _field(self, message, name: str, default=""):
        if isinstance(message, dict):
            return message.get(name, default)
        return getattr(message, name, default)


class LLMError(Exception):
    pass
