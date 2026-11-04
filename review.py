"""AutoGen review stage: a reviewer and an editor check the crew's draft against its sources."""
# What: a two-agent RoundRobinGroupChat (Policy_Compliance_Reviewer, Final_Editor), bounded
#   by max_turns=2. The reviewer lists every draft sentence not found word for word in the
#   retrieved context or the record; the editor returns a VerdictModel that approves the
#   draft unchanged or removes those sentences.
# Task: Part 4, Task 14. Used by: pipeline.py, run_part4_review.py and tests/test_part4.py.
# Both agents run on ReviewClient, a deterministic offline model client that computes its
#   reply from the messages it receives (no API, no pre-written replies).
# Facts from the installed autogen 0.7.5 source: agent names must be Python identifiers;
#   max_turns counts agent turns only; StructuredMessage[VerdictModel] must be listed in
#   custom_message_types, because the team registers message types by class name.

import settings  # noqa: F401  (sets the offline flags before any framework import)

import asyncio
import json
import math
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, ModelFamily, RequestUsage
from pydantic import BaseModel

from mock_llm import KB08_NOTE, NO_CONTEXT, NO_POLICY_QUESTION, NO_RECORD_ID, UNSUPPORTED_TOOL

REVIEWER, EDITOR = "Policy_Compliance_Reviewer", "Final_Editor"
# Markers that split the task message into its two parts.
DRAFT_MARK, CONTEXT_MARK = "=== DRAFT ===", "=== CONTEXT ==="
# Lines that are fixed labels or tool formatting, not policy sentences.
LABELS = ("Retrieval result:", "Lookup result:", "Sources:", "Top-1 similarity:")
# Fixed sentences the crew writes itself; they are never policy claims.
FIXED = (KB08_NOTE, NO_RECORD_ID, NO_POLICY_QUESTION, NO_CONTEXT, UNSUPPORTED_TOOL)
# How many times the review client was called, per agent (reset by each review_draft()).
CALLS = {REVIEWER: 0, EDITOR: 0}


class VerdictModel(BaseModel):
    """The Final_Editor's structured verdict."""

    approved: bool
    final_answer: str
    reason: str


# Split a draft into the sentences that need a source.
# Takes: the draft text. Returns: a list of sentences, skipping labels and fixed sentences.
# Rule: one line at a time; a sentence ends at ".", "!" or "?" followed by a space.
def policy_sentences(draft):
    found = []
    for line in draft.splitlines():
        if not line.strip() or line.strip().startswith(LABELS):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if sentence and sentence not in FIXED:
                found.append(sentence)
    return found


# The reviewer's rule: list every policy sentence not found word for word in the context.
# Takes: the draft and the context text. Returns: the unsupported sentences, in draft order.
def unsupported(draft, context):
    return [sentence for sentence in policy_sentences(draft) if sentence not in context]


# The editor's rule: approve unchanged, or remove the unsupported sentences.
# Takes: the draft and the reviewer's list. Returns: a VerdictModel.
def edit(draft, flagged):
    if not flagged:
        return VerdictModel(approved=True, final_answer=draft, reason="Every policy sentence is in the sources.")
    text = draft
    for sentence in flagged:
        text = text.replace(" " + sentence, "").replace(sentence, "")
    names = "; ".join(repr(sentence) for sentence in flagged)
    return VerdictModel(approved=False, final_answer=text.strip(),
                        reason=f"Removed {len(flagged)} sentence(s) not found in the sources: {names}")


# Build the task message the team receives.
# Takes: the draft and the context. Returns: one text with both parts under markers.
def task_text(draft, context):
    return f"{DRAFT_MARK}\n{draft}\n{CONTEXT_MARK}\n{context}"


class ReviewClient(ChatCompletionClient):
    """Deterministic offline model client; its reply is computed from the input messages."""

    # Takes: which agent this client serves (REVIEWER or EDITOR).
    def __init__(self, role):
        self.role = role
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # Answer one request: the reviewer returns its list as JSON; the editor returns Verdict JSON.
    # Takes: the messages AutoGen sends (system, the task, then the reviewer's reply for the
    #   editor) and json_output (VerdictModel for the editor). Returns: a CreateResult.
    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None):
        CALLS[self.role] += 1
        task = next(m.content for m in messages if getattr(m, "source", None) == "user")
        draft, context = task.split(DRAFT_MARK + "\n", 1)[1].split("\n" + CONTEXT_MARK + "\n", 1)
        if self.role == REVIEWER:
            reply = json.dumps({"unsupported": unsupported(draft, context)})
        else:
            review = next(m.content for m in messages if getattr(m, "source", None) == REVIEWER)
            reply = edit(draft, json.loads(review)["unsupported"]).model_dump_json()
        prompt = sum(len(str(m.content)) for m in messages)
        self._usage = RequestUsage(prompt_tokens=math.ceil(prompt / 4), completion_tokens=math.ceil(len(reply) / 4))
        return CreateResult(finish_reason="stop", content=reply, usage=self._usage, cached=False)

    # Streaming is not used; it yields the same single result.
    async def create_stream(self, messages, **kwargs):
        yield await self.create(messages, **kwargs)

    # Nothing to close: there is no connection.
    async def close(self):
        pass

    # Usage of the latest call (AutoGen's required interface).
    def actual_usage(self):
        return self._usage

    # Same as actual_usage; totals are not tracked here.
    def total_usage(self):
        return self._usage

    # Tokens estimated as characters / 4, as in the Part 4 budget method.
    def count_tokens(self, messages, *, tools=[]):
        return math.ceil(sum(len(str(m.content)) for m in messages) / 4)

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


# Build the two-agent team, bounded by max_turns=2 (reviewer once, editor once).
# Returns: a RoundRobinGroupChat. custom_message_types is required for the editor's message.
def build_team():
    reviewer = AssistantAgent(REVIEWER, model_client=ReviewClient(REVIEWER),
                              system_message="List every draft sentence not found verbatim in the context.")
    editor = AssistantAgent(EDITOR, model_client=ReviewClient(EDITOR), output_content_type=VerdictModel,
                            system_message="Approve the draft unchanged, or remove the listed sentences.")
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[VerdictModel]])


# Review one draft from synchronous code.
# Takes: the Composer's draft and the original retrieved context (chunks and record text).
# Returns: (VerdictModel, the team's messages). The pipeline runs in a worker thread, so
#   asyncio.run() never meets a running event loop.
def review_draft(draft, context):
    for name in CALLS:
        CALLS[name] = 0
    result = asyncio.run(build_team().run(task=task_text(draft, context)))
    verdict = result.messages[-1]
    if not isinstance(verdict, StructuredMessage) or not isinstance(verdict.content, VerdictModel):
        raise ValueError(f"The Final_Editor did not return a VerdictModel (got {type(verdict).__name__}).")
    return verdict.content, result.messages


# The original retrieved context of one crew run, as the tools returned it.
# Takes: the dict from crew.run_crew(). Returns: the retrieved chunks and record fields as text.
def crew_context(result):
    parts = [rag["answer"] for rag in result["rag_results"]]
    parts += ["; ".join(f"{key}: {value}" for key, value in row.items()) for row in result["lookups"]]
    return "\n\n".join(parts)
