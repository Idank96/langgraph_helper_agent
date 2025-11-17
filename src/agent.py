import os
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agent_nodes import router_node, extract_keywords_node, retrieve_node, respond_node, reflect_node
from src.evaluation import LLMJudgeEvaluator


def evaluate_node(state: AgentState) -> AgentState:
    """Evaluate the answer using LLM-as-a-Judge metrics."""
    evaluator = LLMJudgeEvaluator()
    scores = evaluator.evaluate_all(state["question"], state["context"], state["answer"])
    state["evaluation_scores"] = scores
    open(f"{state['output_dir']}/evaluation.json", "w").write(json.dumps(scores, indent=2))
    return state


def route_by_next_action(state: AgentState) -> str:
    """Universal router: all nodes use next_action to decide destination."""
    next_action = state.get("next_action", "end")

    # Validate not a self-loop (safety check)
    last_node = state.get("last_node", "")
    if next_action == last_node and next_action != "":
        # Error: node tried to call itself
        error_msg = f"Self-loop detected: {next_action} tried to call itself"
        state["routing_error"] = error_msg
        print(f"⚠ WARNING: {error_msg}, routing to END")
        return "end"

    return next_action


def build_agent_graph(with_evaluation: bool = False):
    """Build the agentic graph with conditional routing and iterative refinement."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("reflect", reflect_node)

    if with_evaluation:
        workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("router")

    # Universal routing: all nodes connect via next_action
    # Each node can route to any other node based on router's decision
    routing_map = {
        "router": "router",
        "extract_keywords": "extract_keywords",
        "retrieve": "retrieve",
        "respond": "respond",
        "reflect": "reflect",
        "end": "evaluate" if with_evaluation else END
    }

    # Add conditional edges from ALL nodes using universal router
    for node_name in ["router", "extract_keywords", "retrieve", "respond", "reflect"]:
        workflow.add_conditional_edges(
            node_name,
            route_by_next_action,
            routing_map
        )

    if with_evaluation:
        workflow.add_edge("evaluate", END)

    return workflow.compile()


def run_agent(question: str, mode: str = "offline", evaluate: bool = False):
    """Run the agentic system with iterative refinement and routing."""
    output_dir = f"outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    initial_state = {
        "question": question,
        "mode": mode,
        "context": "",
        "answer": "",
        "output_dir": output_dir,
        "evaluation_scores": {},
        "messages": [],
        "retrieval_attempts": 0,
        "iteration": 0,
        "max_iterations": 3,
        "needs_refinement": False,
        "next_action": "",
        "refinement_notes": "", # Notes on why refinement is needed
        "skip_retrieval": False,
        "extracted_keywords": [],
        # Autonomous routing fields
        "last_node": "",
        "node_history": [],
        "context_is_sufficient": False,
        "context_is_relevant": False,
        "quality_score": 0,
        "routing_error": ""
    }

    result = build_agent_graph(with_evaluation=evaluate).invoke(
        initial_state,
        config={"recursion_limit": 50} # To prevent infinite loops
    )

    agent_trace = {
        "question": question,
        "mode": mode,
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "iterations": result.get("iteration", 0),
        "skip_retrieval": result.get("skip_retrieval", False),
        "refinement_notes": result.get("refinement_notes", ""),
        "timestamp": datetime.now().isoformat()
    }
    open(f"{output_dir}/agent_trace.json", "w").write(json.dumps(agent_trace, indent=2))

    if evaluate and result.get("evaluation_scores"):
        return result["answer"], result["evaluation_scores"]

    return result["answer"], None
