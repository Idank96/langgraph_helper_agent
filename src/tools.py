import json
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.offline import retrieve_context
from src.online import search_web

MODEL_NAME = "gemini-2.0-flash"


@tool
def retrieve_documentation(query: str, mode: str = "offline") -> str:
    """Retrieve documentation from offline vector store or online search.

    Args:
        query: The search query
        mode: "offline" for local ChromaDB or "online" for Tavily web search

    Returns:
        Retrieved documentation as string
    """
    try:
        if mode == "offline":
            return retrieve_context(query)
        else:
            return search_web(query)
    except Exception as e:
        # Fallback to offline if online fails
        try:
            fallback_context = retrieve_context(query)
            return f"[Online search failed, using offline docs]\n\n{fallback_context}"
        except Exception as fallback_error:
            return f"Error retrieving documentation: {str(e)}, Fallback error: {str(fallback_error)}"


def validate_context_quality(question: str, context: str) -> dict:
    """Validate if retrieved context is relevant and sufficient to answer the question.

    Args:
        question: The user's question
        context: Retrieved documentation context

    Returns:
        Dictionary with keys: is_relevant (bool), is_sufficient (bool), missing_info (str)
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = f"""Analyze if the provided context is relevant and sufficient to answer the question.

Question: {question}

Context:
{context[:2000]}  # Limit context length for validation

Respond in JSON format with:
- is_relevant: true/false (Is the context about the right topic?)
- is_sufficient: true/false (Does it contain enough information to answer fully?)
- missing_info: string (What key information is missing, if any?)

Example response:
{{"is_relevant": true, "is_sufficient": false, "missing_info": "Missing code examples for implementation"}}

Your analysis:"""

    try:
        response = llm.invoke(prompt).content
        # Try to parse JSON from response
        # Handle both raw JSON and markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)

        # Ensure required keys exist
        return {
            "is_relevant": result.get("is_relevant", True),
            "is_sufficient": result.get("is_sufficient", True),
            "missing_info": result.get("missing_info", "")
        }
    except Exception as e:
        # Fallback: assume context is valid if validation fails
        return {
            "is_relevant": True,
            "is_sufficient": True,
            "missing_info": f"Validation error: {str(e)}"
        }


def refine_search_query(original_question: str, feedback: str) -> str:
    """Generate an improved search query based on feedback about missing information.

    Args:
        original_question: The original user question
        feedback: Description of what information is missing

    Returns:
        Improved search query as string
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = f"""Generate a more specific search query to find better documentation.

Original Question: {original_question}

Feedback on Previous Results: {feedback}

Create a refined search query that:
1. Targets the missing information mentioned in the feedback
2. Uses specific technical terms related to LangGraph/LangChain
3. Is concise (5-10 words maximum)

Respond with ONLY the improved query, no explanation.

Improved Query:"""

    try:
        refined_query = llm.invoke(prompt).content.strip()
        # Remove quotes if LLM added them
        refined_query = refined_query.strip('"\'')
        return refined_query if refined_query else original_question
    except Exception as e:
        # Fallback to original question if refinement fails
        return original_question


def check_answer_completeness(question: str, answer: str) -> dict:
    """Evaluate the quality and completeness of a generated answer.

    Args:
        question: The original question
        answer: The generated answer to evaluate

    Returns:
        Dictionary with keys: quality_score (int 0-10), needs_improvement (bool), suggestions (str)
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = f"""Evaluate the quality of this answer to a LangGraph/LangChain question.

Question: {question}

Answer:
{answer}

Rate the answer on these criteria:
1. Completeness: Does it fully answer the question?
2. Accuracy: Is the information correct?
3. Clarity: Is it easy to understand?
4. Practicality: Does it include code examples when relevant?

Respond in JSON format with:
- quality_score: integer from 0-10 (0=very poor, 10=excellent)
- needs_improvement: true/false (true if score < 7)
- suggestions: string (specific improvements needed, if any)

Example response:
{{"quality_score": 6, "needs_improvement": true, "suggestions": "Add code examples and explain the workflow more clearly"}}

Your evaluation:"""

    try:
        response = llm.invoke(prompt).content
        # Try to parse JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)

        quality_score = int(result.get("quality_score", 7))

        return {
            "quality_score": quality_score,
            "needs_improvement": quality_score < 7,
            "suggestions": result.get("suggestions", "")
        }
    except Exception as e:
        # Fallback: assume answer is acceptable if evaluation fails
        return {
            "quality_score": 7,
            "needs_improvement": False,
            "suggestions": f"Evaluation error: {str(e)}"
        }
