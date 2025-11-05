import os
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agent_nodes import router_node, retrieve_node, respond_node, reflect_node
from src.evaluation import LLMJudgeEvaluator


def evaluate_node(state: AgentState) -> AgentState:
    """Evaluate the answer using LLM-as-a-Judge metrics."""
    evaluator = LLMJudgeEvaluator()
    scores = evaluator.evaluate_all(state["question"], state["context"], state["answer"])
    state["evaluation_scores"] = scores
    open(f"{state['output_dir']}/evaluation.json", "w").write(json.dumps(scores, indent=2))
    return state


def route_from_router(state: AgentState) -> str:
    """Router conditional edge: decides where to go based on next_action."""
    next_action = state.get("next_action", "end")

    if next_action == "retrieve":
        return "retrieve"
    elif next_action == "respond":
        return "respond"
    elif next_action == "reflect":
        return "reflect"
    else:
        return "end"


def route_from_reflect(state: AgentState) -> str:
    """Reflect conditional edge: regenerate answer or go to end."""
    if state.get("needs_refinement", False):
        return "respond"
    else:
        return "end"


def build_agent_graph(with_evaluation: bool = False):
    """Build the agentic graph with conditional routing and iterative refinement."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("reflect", reflect_node)

    if with_evaluation:
        workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("router")

    # Edges
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "retrieve": "retrieve",
            "respond": "respond",
            "reflect": "reflect",
            "end": "evaluate" if with_evaluation else END
        }
    )

    workflow.add_edge("retrieve", "router")
    workflow.add_edge("respond", "router")

    workflow.add_conditional_edges(
        "reflect",
        route_from_reflect,
        {
            "respond": "respond",
            "end": "evaluate" if with_evaluation else END
        }
    )

    if with_evaluation:
        workflow.add_edge("evaluate", END)

    return workflow.compile()


def run_agent(question: str, mode: str = "offline", evaluate: bool = False):
    """Run the agentic system with iterative refinement and smart routing."""
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
        "refinement_notes": "",
        "skip_retrieval": False
    }

    result = build_agent_graph(with_evaluation=evaluate).invoke(
        initial_state,
        config={"recursion_limit": 50}
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
