"""Per-session conversation memory with LangChain, kept in this process only."""
# What: one InMemoryChatMessageHistory per session id, wrapped around a turn function with
#   RunnableWithMessageHistory, plus the rule for reusing a record id from earlier turns.
# Task: Part 2, Task 8. Used by: pipeline.py, run_task8_9.py and tests/test_task8_9.py.
# Facts read from the installed langchain-core 1.6.10: the session id is passed per call as
#   config={"configurable": {"session_id": ...}}; the history is injected into the input
#   under history_messages_key before the turn runs, and afterwards the input text and the
#   output text are appended as a HumanMessage and an AIMessage. Both classes are
#   deprecated in 1.6.x, so a LangChainDeprecationWarning is expected; it is not silenced.

import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from rules import REFER_BACK_WORDS

# Session id -> that session's history. Nothing is written to disk.
STORE = {}
# An appointment record id such as APT-007.
RECORD_ID = re.compile(r"APT-[0-9]{3}")
# Any refer-back word or phrase from rules.py, as whole words, in any case.
REFER_BACK = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in REFER_BACK_WORDS) + r")\b", re.I)


# The single entry point for all user text (Task 10 will add phone masking here).
# Takes: the raw user text. Returns: the text to use and store; unchanged for now.
# Why: history only ever stores what this returns, so masking will cover history too.
def prepare_user_text(text):
    return text


# Return the history for one session, creating an empty one the first time.
# Takes: a session id. Returns: that session's InMemoryChatMessageHistory.
def get_history(session_id):
    if session_id not in STORE:
        STORE[session_id] = InMemoryChatMessageHistory()
    return STORE[session_id]


# Decide which remembered record id, if any, this turn should look up.
# Takes: the current question and this session's earlier messages. Returns: an id or None.
# Rule: only when the question names no id itself AND refers back to an appointment
#   ("What is its fee?"); then the newest id the user wrote earlier in this session.
def remembered_record_id(question, history):
    if RECORD_ID.search(question) or not REFER_BACK.search(question):
        return None
    for message in reversed(history):
        found = RECORD_ID.findall(message.content) if isinstance(message, HumanMessage) else []
        if found:
            return found[-1]
    return None


# Wrap a turn function so each call reads and extends its own session's history.
# Takes: a function from {"question", "history"} to a dict with an "answer" string.
# Returns: a runnable; call it with .invoke({"question": text}, config=session_config(id)).
def with_memory(turn):
    return RunnableWithMessageHistory(RunnableLambda(turn), get_history,
                                      input_messages_key="question",
                                      history_messages_key="history",
                                      output_messages_key="answer")


# Build the config that tells RunnableWithMessageHistory which session to use.
def session_config(session_id):
    return {"configurable": {"session_id": session_id}}


# List a session's stored messages as (type, text) pairs, for reports and tests.
def stored_history(session_id):
    return [(message.type, message.content) for message in get_history(session_id).messages]
