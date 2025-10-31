from typing import TypedDict


class AgentState(TypedDict):
    question: str
    mode: str
    context: str
    answer: str
    output_dir: str
