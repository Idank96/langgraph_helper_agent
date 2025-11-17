from typing import TypedDict, Dict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Original fields
    question: str
    mode: str
    context: str
    answer: str
    output_dir: str
    evaluation_scores: Dict[str, float]

    # New agentic fields
    messages: List[BaseMessage]  # Conversation tracking
    retrieval_attempts: int  # Number of retrieval attempts made
    iteration: int  # Current refinement iteration
    max_iterations: int  # Maximum iterations allowed (default 3)
    needs_refinement: bool  # Flag if answer needs regeneration
    next_action: str  # Router's decision: "retrieve", "respond", "reflect", "end"
    refinement_notes: str  # Suggestions for improvement in next iteration
    skip_retrieval: bool  # If LLM decides it can answer without docs
    extracted_keywords: List[str]  # Keywords extracted for multi-query retrieval
