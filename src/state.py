from typing import TypedDict, Dict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    question: str
    mode: str
    context: str
    answer: str
    output_dir: str
    evaluation_scores: Dict[str, float]

    messages: List[BaseMessage]
    retrieval_attempts: int
    iteration: int
    max_iterations: int
    needs_refinement: bool
    next_action: str
    refinement_notes: str
    skip_retrieval: bool
    extracted_keywords: List[str]

    last_node: str
    node_history: List[str]
    context_is_sufficient: bool
    context_is_relevant: bool
    quality_score: int
    routing_error: str

    # Retrieval control fields
    current_query: str
    restrict_to_official: bool
