import os
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from src.offline import retrieve_context
from src.online import search_web


def retrieve_node(state: AgentState) -> AgentState:
    if state["mode"] == "offline":
        state["context"] = retrieve_context(state["question"])
    else:
        try:
            state["context"] = search_web(state["question"])
            open(f"{state['output_dir']}/sources.txt", "w").write(state["context"])
        except Exception:
            state["context"] = retrieve_context(state["question"])
            state["context"] = f"[Note: Online search unavailable, using offline docs]\n\n{state['context']}"
    return state


def respond_node(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt = f"""You are a helpful assistant for LangGraph and LangChain developers.

    Question:
    {state["question"]}

    Context:
    {state["context"]}

    Provide a clear, practical answer based on the context above. Include code examples when relevant."""

    state["answer"] = llm.invoke(prompt).content
    open(f"{state['output_dir']}/answer.md", "w").write(state["answer"])
    open(f"{state['output_dir']}/chat.md", "w").write(f"{prompt}\n\n---\n\n{state['answer']}")
    return state


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("respond", respond_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


def run_agent(question: str, mode: str = "offline") -> str:
    output_dir = f"outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    result = build_graph().invoke({
        "question": question,
        "mode": mode,
        "context": "",
        "answer": "",
        "output_dir": output_dir
    })
    return result["answer"]
