"""The structured reply every pipeline answer must fit."""
# What: the Pydantic model CrewResponse and one function that checks a reply against it.
# Task: Part 2, Task 9. Used by: crew.py (as the Compose task's output_pydantic),
#   pipeline.py, run_task8_9.py and tests/test_task8_9.py.
# CrewAI checks the Composer's JSON against this model; pipeline.py then fills the factual
#   fields from the tools and checks the finished reply again with validate_response().

from pydantic import BaseModel, ConfigDict, ValidationError


class CrewResponse(BaseModel):
    """One answer to the customer, with the facts that back it."""

    # Unknown keys are an error, so a misspelt field cannot slip through unnoticed.
    model_config = ConfigDict(extra="forbid")

    # The reply text shown to the customer.
    answer: str
    # KB parent ids quoted in the answer, such as "02_cancellation_rescheduling".
    sources: list[str]
    # Names of the tools that actually ran for this turn.
    tools_used: list[str]
    # The appointment that was looked up, or None if no record was found.
    record_id: str | None
    # tools.recommend_escalation() for that record, or None if there was no record.
    escalation_recommended: bool | None
    # Provisional rule until Task 10: True when at least one tool returned real data.
    grounded: bool
    # Always False until Task 10's guardrail can refuse.
    refused: bool
    refusal_reason: str | None


# Check one reply against CrewResponse.
# Takes: a dict (or a CrewResponse). Returns: a validated CrewResponse.
# Why: every reply is checked in code, not only by CrewAI, and a failure names the fields.
def validate_response(data) -> CrewResponse:
    try:
        return CrewResponse.model_validate(data)
    except ValidationError as error:
        fields = ", ".join(".".join(str(part) for part in item["loc"]) or "(whole reply)"
                           for item in error.errors())
        raise ValueError(f"Reply does not match CrewResponse; invalid fields: {fields}\n{error}") from error
