"""
Tool-Calling ReAct Agent for LangGraph Helper

This module creates an autonomous ReAct agent that can dynamically select and call
tools to answer LangGraph/LangChain questions. The agent uses genuine reasoning to
decide which tools to use and when, rather than following a predetermined workflow.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from src.tools import (
    search_documentation,
    validate_retrieved_information,
    refine_search_strategy,
    check_answer_quality
)

MODEL_NAME = "gemini-2.0-flash"

TOOL_CALLING_SYSTEM_PROMPT = """You are an expert LangGraph/LangChain documentation assistant with access to powerful tools.

Your goal is to provide COMPREHENSIVE, WELL-RESEARCHED answers to questions about LangGraph and LangChain.

## Refinement Loops (Graph-Level Safety Rails)

This agent runs within a graph that includes automatic refinement loops:

1. **Search Refinement Loop**: After you generate an answer, the system validates whether the context you gathered is sufficient. If not, it automatically refines the search query and gathers additional context.

2. **Answer Quality Loop**: Your answer is evaluated for quality (target: 8/10). If it scores below 8/10, the system provides specific feedback and you'll get another chance to improve the answer.

This means you should focus on:
- Providing detailed, comprehensive initial answers
- Using the search tools to gather rich context
- Including code examples and practical implementation details
- Being thorough rather than brief

The refinement loops will catch gaps and help improve the answer iteratively.

## Available Tools

You have access to these tools:

1. **search_documentation**: Search LangGraph/LangChain documentation (offline ChromaDB or online Tavily)
2. **validate_retrieved_information**: Check if search results are relevant and sufficient for answering the question
3. **refine_search_strategy**: Generate better, more specific search queries when results are insufficient
4. **check_answer_quality**: Evaluate if your answer meets quality standards (target: 8-9/10)
5. **refine_answer**: Improve an existing answer based on quality feedback (used by refinement loops)

## Recommended Workflow for Implementation Questions

For questions about "how to implement" or "how to use" features:

1. **Search**: Use search_documentation with an initial query
2. **Validate**: ALWAYS call validate_retrieved_information to check if results are sufficient
3. **Refine if needed**: If validation shows missing information:
   - Call refine_search_strategy with the validation feedback
   - Search again with the refined query
   - Validate the new results
4. **Quality check**: Before finalizing, call check_answer_quality to ensure 8+/10 score
5. **Iterate**: If quality is below 8/10, refine and search for missing details

## Decision-Making Strategy

**Simple factual questions** (e.g., "What does RAG stand for?"):
- You can answer directly if you're certain
- No tools needed for basic definitions

**Implementation/usage questions** (e.g., "How do I add persistence?"):
- Search documentation for specific implementation details
- VALIDATE the search results (don't assume they're sufficient)
- If validation shows gaps (missing code examples, incomplete steps, unclear instructions):
  → Refine your search query to target the missing information
  → Search again with refined query
- Aim for comprehensive answers with code examples

## Quality Standards

- **7/10**: Minimum acceptable (basic answer, may lack details)
- **8-9/10**: Target quality (comprehensive with code examples)
- **10/10**: Exceptional (complete implementation guide)

**Important**: Don't settle for 7/10 if you can refine and improve to 8-9/10.

## Example Reasoning Process

"The user asked about implementing checkpointing in LangGraph. This is an implementation question.

1. Search documentation: 'LangGraph checkpointing implementation'
2. Validate results: Are there code examples? Is the setup process clear?
3. Validation shows: Missing specific PostgreSQL setup steps
4. Refine search: 'LangGraph PostgreSQL checkpointer setup configuration'
5. Search again with refined query
6. Validate new results: Now includes database setup and code examples
7. Check answer quality: Ensure it scores 8+/10
8. If below 8, identify what's missing and refine further"

Remember: You have autonomy to decide tool order, but for implementation questions, validation and refinement are crucial for high-quality answers.
"""


def create_tool_calling_agent(mode: str = "offline", temperature: float = 0.1):
    """
    Create a ReAct agent with access to documentation search and validation tools.

    The agent uses create_react_agent from LangGraph, which provides:
    - Autonomous tool selection based on reasoning
    - Built-in ReAct loop (Thought → Action → Observation → repeat)
    - Natural stopping when the agent determines it has a sufficient answer

    Args:
        mode: "offline" or "online" - determines which search backend to use
        temperature: LLM temperature for agent reasoning (default 0 for consistency)

    Returns:
        Compiled LangGraph agent ready for invocation
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature
    )

    tools = [
        search_documentation,
        validate_retrieved_information,
        refine_search_strategy,
        check_answer_quality
    ]

    # Create the ReAct agent with proper configuration
    # The agent will loop until it produces a response without tool calls
    agent = create_react_agent(
        model=llm,
        tools=tools,
        # Note: create_react_agent handles the loop internally
        # It continues until the model responds without calling tools
    )

    return agent, TOOL_CALLING_SYSTEM_PROMPT
