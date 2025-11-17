import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from src.tools import (
    retrieve_documentation,
    validate_context_quality,
    refine_search_query,
    check_answer_completeness,
    extract_keywords,
    retrieve_with_keywords
)

MODEL_NAME = "gemini-2.0-flash"


def _log(message: str, state: AgentState):
    """Helper function for conditional logging based on verbose flag."""
    # Check if verbose flag exists in output_dir path (will be set by main.py)
    if os.environ.get("AGENT_VERBOSE", "false") == "true":
        print(f"{message}")


def router_node(state: AgentState) -> AgentState:
    """LLM-driven router that decides the next action based on current state."""
    _log("\n------- ROUTER NODE -------", state)

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    # Determine what decision to make based on state
    has_context = bool(state.get("context", "").strip())
    has_answer = bool(state.get("answer", "").strip())
    skip_retrieval = state.get("skip_retrieval", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    _log(f"Current state: context={has_context}, answer={has_answer}, skip_retrieval={skip_retrieval}, iteration={iteration}/{max_iterations}", state)

    # Decision 1: Have answer - should we reflect or end?
    if has_answer:
        if iteration < max_iterations:
            _log(f"→ Router decision: REFLECT (iteration {iteration}/{max_iterations})", state)
            state["next_action"] = "reflect"
        else:
            _log(f"→ Router decision: END (reached max iterations: {max_iterations})", state)
            state["next_action"] = "end"

    # Decision 2: Have context but no answer yet - generate answer
    elif has_context and not has_answer:
        _log("→ Router decision: RESPOND (have context, need answer)", state)
        state["next_action"] = "respond"

    # Decision 3: No context and haven't decided yet - should we retrieve or answer from knowledge?
    elif not has_context and not skip_retrieval:
        _log(f"→ Asking LLM: Should we retrieve documentation?", state)
        prompt = f"""You are a LangGraph/LangChain documentation assistant. Your PRIMARY responsibility is to provide answers grounded in OFFICIAL DOCUMENTATION.

Question: {state['question']}

CRITICAL RULES:
1. You MUST respond "RETRIEVE" for ANY question about:
   - How to implement/use features (e.g., "How do I...", "How to...")
   - Code examples or API usage
   - Configuration, setup, or best practices
   - Specific LangGraph/LangChain features or components
   - Recent updates, versions, or changes
   - Troubleshooting or debugging
   - Comparisons between features

2. You may ONLY respond "ANSWER" for:
   - Extremely simple definitions (e.g., "What is LangGraph?", "What does LLM mean?")
   - Basic conceptual questions with one-word answers

Examples:
- "How do I add persistence to a LangGraph agent?" → RETRIEVE
- "What is a node in LangGraph?" → RETRIEVE (needs documentation for accuracy)
- "Show me how to use StateGraph" → RETRIEVE
- "What does LLM stand for?" → ANSWER (trivial)

Respond with ONLY one word - "RETRIEVE" or "ANSWER":"""

        try:
            response = llm.invoke(prompt).content.strip().upper()
            _log(f"   LLM decision: {response}", state)

            if "ANSWER" in response:
                _log("→ Router decision: RESPOND (LLM determined question is trivial)", state)
                state["next_action"] = "respond"
                state["skip_retrieval"] = True
            else:
                _log("→ Router decision: RETRIEVE", state)
                state["next_action"] = "retrieve"
                state["skip_retrieval"] = False
        except Exception as e:
            _log(f"   Router error: {str(e)}, defaulting to RETRIEVE", state)
            state["next_action"] = "retrieve"
            state["skip_retrieval"] = False

    # Decision 4: No context but skip_retrieval is True - answer from knowledge
    elif not has_context and skip_retrieval:
        _log("→ Router decision: RESPOND (skip retrieval, answer from knowledge)", state)
        state["next_action"] = "respond"

    # Fallback
    else:
        _log("→ Router decision: END (no clear next step)", state)
        state["next_action"] = "end"

    return state


def extract_keywords_node(state: AgentState) -> AgentState:
    """Extract keywords from the question for multi-query retrieval."""
    _log("\n------- EXTRACT KEYWORDS NODE -------", state)

    question = state["question"]
    _log(f"Extracting keywords from: '{question}'", state)

    try:
        keywords = extract_keywords(question)
        state["extracted_keywords"] = keywords

        if keywords:
            _log(f"  ✓ Extracted {len(keywords)} keyword(s): {', '.join(keywords)}", state)
            _log(f"  → Will perform {len(keywords) + 1} searches: original + {len(keywords)} keyword(s)\n", state)
        else:
            _log(f"  → No additional keywords extracted (simple query)", state)
            _log(f"  → Will perform 1 search with original question\n", state)

    except Exception as e:
        _log(f"  ✗ Error extracting keywords: {str(e)}", state)
        _log(f"  → Fallback: using original question only\n", state)
        state["extracted_keywords"] = []

    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve documentation with validation and retry logic."""
    _log("\n------- RETRIEVE NODE -------", state)

    max_attempts = 3
    current_query = state["question"]
    keywords = state.get("extracted_keywords", [])
    mode = state.get("mode", "offline")
    restrict_to_official = True  # Start with official docs only
    _log(f"Retrieval mode: '{mode}', max attempts: {max_attempts}", state)

    for attempt in range(1, max_attempts + 1):
        _log(f"\n  Attempt {attempt}/{max_attempts}", state)
        _log(f"  Query: '{current_query[:80]}...'", state) if len(current_query) > 80 else _log(f"  Query: '{current_query}'", state)

        # Special handling for online mode: if first attempt failed, try unrestricted search
        if mode == "online" and attempt == 2 and restrict_to_official:
            _log(f"  → Switching to unrestricted web search (searching beyond official docs)", state)
            restrict_to_official = False
            # Reset query to original for unrestricted search
            current_query = state["question"]

        try:
            # Retrieve documentation using keyword-enhanced retrieval if keywords exist
            if keywords and attempt == 1:
                # Use multi-query retrieval on first attempt
                _log(f"  Using multi-query retrieval with {len(keywords)} keyword(s): {keywords}", state)
                if mode == "offline":
                    context = retrieve_with_keywords(current_query, keywords, mode)
                else:
                    context = retrieve_with_keywords(current_query, keywords, mode, restrict_to_official=restrict_to_official)
            else:
                # Use standard retrieval for refinement attempts
                if mode == "offline":
                    context = retrieve_documentation.invoke({"query": current_query, "mode": mode})
                else:
                    context = retrieve_documentation.invoke({
                        "query": current_query,
                        "mode": mode,
                        "restrict_to_official": restrict_to_official
                    })

            # Validate the retrieved context
            validation = validate_context_quality(
                question=state["question"],
                context=context
            )

            is_relevant = validation.get("is_relevant", True)
            is_sufficient = validation.get("is_sufficient", True)
            missing_info = validation.get("missing_info", "")

            # Calculate retrieval statistics
            char_count = len(context)
            # Rough token estimate: ~4 characters per token (common approximation)
            estimated_tokens = char_count // 4

            # Count sources (separated by "=== Results for:" markers if multi-query retrieval was used)
            source_count = context.count("=== Results for:")
            if source_count == 0:
                source_count = 1  # Standard single retrieval

            search_scope = "official docs only" if (mode == "online" and restrict_to_official) else ("unrestricted web" if mode == "online" else "offline vector DB")
            _log(f"  ✓ Retrieved {char_count} characters (~{estimated_tokens:,} tokens) from {source_count} source(s) [{search_scope}]", state)

            # Create a one-line summary of the retrieved context for debugging
            context_summary = context[:500].replace('\n', ' ').strip()
            if len(context) > 500:
                context_summary += "..."
            _log(f"  Context Summary: {context_summary}", state)

            _log(f"  Validation:", state)
            _log(f"    - Is Relevant: {'✓ Yes' if is_relevant else '✗ No'}", state)
            _log(f"    - Is Sufficient: {'✓ Yes' if is_sufficient else '✗ No'}", state)
            if missing_info:
                _log(f"    - Missing info: {missing_info}", state)

            # If context is good, save it and break
            if is_relevant and is_sufficient:
                state["context"] = context
                state["retrieval_attempts"] = attempt
                _log(f"  ✓ SUCCESS: Context quality acceptable after {attempt} attempt(s)\n", state)

                # Save context to file
                if mode == "offline":
                    open(f"{state['output_dir']}/context.txt", "w", encoding="utf-8").write(context)
                else:
                    open(f"{state['output_dir']}/sources.txt", "w", encoding="utf-8").write(context)

                break

            # If not good and we have attempts remaining, refine the query or try unrestricted search
            if attempt < max_attempts:
                # For online mode: if first attempt with official docs failed, try unrestricted next
                if mode == "online" and attempt == 1 and restrict_to_official:
                    _log(f"  ⚠ Official docs insufficient, will try unrestricted web search next", state)
                    # Don't refine query - we'll use original question with unrestricted search
                else:
                    # Refine query for subsequent attempts
                    _log(f"  ⚠ Context quality insufficient, refining query...", state)
                    refined_query = refine_search_query(
                        original_question=state["question"],
                        feedback=missing_info or "Context not specific enough"
                    )
                    _log(f"  → Refined query: '{refined_query}'", state)
                    current_query = refined_query
            else:
                # Last attempt - accept what we have
                _log(f"  Max attempts reached, accepting current context\n", state)
                state["context"] = context
                state["retrieval_attempts"] = attempt

                # Save context to file
                if mode == "offline":
                    open(f"{state['output_dir']}/context.txt", "w", encoding="utf-8").write(context)
                else:
                    open(f"{state['output_dir']}/sources.txt", "w", encoding="utf-8").write(context)

        except Exception as e:
            _log(f"Retrieval error on attempt {attempt}: {str(e)}", state)
            if attempt == max_attempts:
                # Final fallback
                state["context"] = f"Error: Unable to retrieve documentation after {max_attempts} attempts"
                state["retrieval_attempts"] = attempt

    return state


def respond_node(state: AgentState) -> AgentState:
    """Generate answer using LLM, with or without retrieved context."""
    _log("\n------- RESPOND NODE -------", state)

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    iteration = state.get("iteration", 0)
    is_regeneration = bool(state.get("refinement_notes", ""))

    # Check if we're skipping retrieval (answering from knowledge)
    if state.get("skip_retrieval", False):
        if is_regeneration:
            _log(f"Regenerating answer (iteration {iteration}) from knowledge with improvements", state)
        else:
            _log("Generating answer from LLM knowledge (no documentation retrieval)", state)

        prompt = f"""You are a helpful assistant for LangGraph and LangChain developers.

Question:
{state['question']}

Provide a clear, practical answer based on your knowledge. Include code examples when relevant.
"""
        if state.get("refinement_notes"):
            prompt += f"""
Previous answer needs improvement. Please address these suggestions:
{state['refinement_notes']}
"""
            _log(f"  Applying improvements: {state['refinement_notes']}", state)
    else:
        # Normal mode with context
        context_size = len(state.get('context', ''))
        if is_regeneration:
            _log(f"Regenerating answer (iteration {iteration}) using {context_size} chars of context", state)
        else:
            _log(f"Generating answer using {context_size} characters of retrieved context", state)

        prompt = f"""You are a helpful assistant for LangGraph and LangChain developers.

Question:
{state['question']}

Context:
{state.get('context', '')}
"""
        # Add refinement notes if this is a regeneration
        if state.get("refinement_notes"):
            prompt += f"""
Previous answer needs improvement. Please address these suggestions:
{state['refinement_notes']}
"""
            _log(f"  Applying improvements: {state['refinement_notes']}", state)

        prompt += "\nProvide a clear, practical answer based on the context above. Include code examples when relevant."

    try:
        _log("  Invoking LLM to generate answer...", state)
        answer = llm.invoke(prompt).content
        state["answer"] = answer

        # Save answer and chat to files
        open(f"{state['output_dir']}/answer.md", "w", encoding="utf-8").write(answer)
        open(f"{state['output_dir']}/chat.md", "w", encoding="utf-8").write(
            f"{prompt}\n\n---\n\n{answer}"
        )

        answer_length = len(answer)
        _log(f"  ✓ Answer generated successfully ({answer_length} characters)\n", state)

    except Exception as e:
        _log(f"  ✗ Error generating answer: {str(e)}\n", state)
        state["answer"] = f"Error generating answer: {str(e)}"

    return state


def reflect_node(state: AgentState) -> AgentState:
    """Self-critique the generated answer and decide if refinement is needed."""
    _log("\n------- REFLECT NODE -------", state)

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    _log(f"Evaluating answer quality using LLM critic (iteration {iteration}/{max_iterations})...", state)

    try:
        # Evaluate answer completeness
        evaluation = check_answer_completeness(
            question=state["question"],
            answer=state["answer"]
        )

        quality_score = evaluation.get("quality_score", 7)
        suggestions = evaluation.get("suggestions", "")

        _log(f"\n  Quality Assessment:", state)
        _log(f"    Score: {quality_score}/10", state)

        # Visual score indicator
        if quality_score >= 8:
            score_indicator = "✓ Excellent"
        elif quality_score >= 7:
            score_indicator = "✓ Good"
        elif quality_score >= 5:
            score_indicator = "⚠ Fair"
        else:
            score_indicator = "✗ Poor"
        _log(f"    Rating: {score_indicator}", state)

        # Decide if we need to refine
        if quality_score < 7 and iteration < max_iterations:
            state["needs_refinement"] = True
            state["iteration"] = iteration + 1
            state["refinement_notes"] = suggestions
            _log(f"    Decision: NEEDS REFINEMENT", state)
            _log(f"    Suggestions: {suggestions}", state)
            _log(f"  → Routing to RESPOND to regenerate answer (iteration {state['iteration']}/{max_iterations})\n", state)
        else:
            state["needs_refinement"] = False
            if quality_score >= 7:
                _log(f"    Decision: ACCEPT (quality threshold met)", state)
                _log(f"  → Routing to END (answer is acceptable)\n", state)
            else:
                _log(f"    Decision: ACCEPT (max iterations reached)", state)
                _log(f"  → Routing to END (iteration limit: {max_iterations})\n", state)

    except Exception as e:
        _log(f"  ✗ Error during reflection: {str(e)}", state)
        _log(f"  → Accepting current answer due to error\n", state)
        state["needs_refinement"] = False

    return state
