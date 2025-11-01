from typing import TypedDict, Dict


class AgentState(TypedDict):
    question: str
    mode: str
    context: str
    answer: str
    output_dir: str
    evaluation_scores: Dict[str, float]
