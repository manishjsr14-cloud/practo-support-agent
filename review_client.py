"""Deterministic offline AutoGen model client: its reply is computed from the input messages."""
# What: a ChatCompletionClient that never calls an API. Each instance is given a plain
#   function that computes the reply from the messages AutoGen sends, so the reply is not
#   pre-written (unlike autogen_ext's ReplayChatCompletionClient).
# Task: Part 4, Task 14. Used by: review.py.
# Facts from the installed autogen 0.7.5 source: AssistantAgent calls create() once per turn
#   (no tools, no streaming); with output_content_type set it passes json_output=<model> and
#   validates the reply with model_validate_json; model_info must declare vision,
#   function_calling, json_output and family (structured_output only warns if missing).

import settings  # noqa: F401  (sets the offline flags before any framework import)

from autogen_core.models import ChatCompletionClient, CreateResult, ModelFamily, RequestUsage

from governance import charge, prompt_text, tokens


class RuleClient(ChatCompletionClient):
    """Offline model client whose reply comes from a plain rule function."""

    # Takes: the agent's name (used to count its calls), the rule function
    #   (messages -> reply text) and the shared call counter dict.
    def __init__(self, name, rule, calls):
        self.name, self.rule, self.calls = name, rule, calls
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # Answer one request with the rule's reply, counting the call and charging the budget.
    # Takes: AutoGen's messages and options. Returns: a CreateResult with the reply text.
    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None):
        self.calls[self.name] += 1
        # Part 4 budget meter: the prompt is charged before any work, the reply after it.
        charge(prompt_text(messages), call=True)
        reply = self.rule(messages)
        charge(reply)
        self._usage = RequestUsage(prompt_tokens=tokens(prompt_text(messages)), completion_tokens=tokens(reply))
        return CreateResult(finish_reason="stop", content=reply, usage=self._usage, cached=False)

    # Streaming is not used; it yields the same single result.
    async def create_stream(self, messages, **kwargs):
        yield await self.create(messages, **kwargs)

    # Nothing to close: there is no connection.
    async def close(self):
        pass

    # Usage of the latest call; totals are not tracked (AutoGen's required interface).
    def actual_usage(self):
        return self._usage

    def total_usage(self):
        return self._usage

    # Tokens estimated as characters / 4, as in the Part 4 budget method.
    def count_tokens(self, messages, *, tools=[]):
        return tokens(prompt_text(messages))

    # No context limit applies to this offline client.
    def remaining_tokens(self, messages, *, tools=[]):
        return 10**9

    # The deprecated name for model_info.
    @property
    def capabilities(self):
        return self.model_info

    # What this client supports: JSON and structured output, no tools and no images.
    @property
    def model_info(self):
        return {"vision": False, "function_calling": False, "json_output": True,
                "structured_output": True, "family": ModelFamily.UNKNOWN}
