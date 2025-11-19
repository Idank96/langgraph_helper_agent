import json
from langchain_google_genai import ChatGoogleGenerativeAI
from src.offline import retrieve_context
from src.online import search_web

MODEL_NAME = "gemini-2.0-flash"


def retrieve_documentation(query: str, mode: str = "offline", restrict_to_official: bool = True) -> str:
    """Retrieve documentation from offline vector store or online search.

    Args:
        query: The search query
        mode: "offline" for local ChromaDB or "online" for Tavily web search
        restrict_to_official: If True (default), only search official docs. If False, search anywhere.

    Returns:
        Retrieved documentation as string
    """
    try:
        if mode == "offline":
            return retrieve_context(query)
        else:
            return search_web(query, restrict_to_official=restrict_to_official)
    except Exception as e:
        try:
            fallback_context = retrieve_context(query)
            return f"[Online search failed, using offline docs]\n\n{fallback_context}"
        except Exception as fallback_error:
            return f"Error retrieving documentation: {str(e)}, Fallback error: {str(fallback_error)}"


def validate_context_quality(question: str, context: str) -> dict:
    """Validate if retrieved context is relevant and sufficient to answer the question."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = f"""Analyze if the provided context is relevant and sufficient to answer the question.

Question: {question}

Context:
{context[:2000]}

CRITICAL RULES FOR is_relevant:
1. Set is_relevant to TRUE only if the context DIRECTLY addresses the specific question being asked
2. Set is_relevant to FALSE if:
   - The context mentions related topics but doesn't answer the actual question
   - The context only provides partial/tangential information about one aspect of the question
   - For comparison questions (e.g., "What's the difference between X and Y?"), the context must explain BOTH concepts and their differences, not just mention that one replaces the other
   - For "how-to" questions, the context must provide implementation steps, not just conceptual overview
   - The context is about a different feature/topic entirely

CRITICAL RULES FOR is_sufficient:
1. Set is_sufficient to TRUE **ONLY IF** the context contains ALL necessary information with NO significant gaps
2. If ANY key information is missing, set is_sufficient to FALSE
3. If is_sufficient is TRUE, missing_info should be empty or "None"
4. If is_sufficient is FALSE, explain what's missing in missing_info

Respond in JSON format with:
- is_relevant: true/false (Does the context DIRECTLY and COMPLETELY address what the question asks?)
- is_sufficient: true/false (Does it contain ALL information needed to fully answer the question?)
- missing_info: string (What key information is missing? Use "None" if nothing is missing)

Example responses:
Question: "What's the difference between X and Y?"
Context: "X is deprecated, use Y instead"
Response: {{"is_relevant": false, "is_sufficient": false, "missing_info": "Context doesn't explain what X is/was, what Y is, or their actual differences - only mentions deprecation"}}

Question: "How do I implement feature X?"
Context: "Feature X allows you to do Y"
Response: {{"is_relevant": false, "is_sufficient": false, "missing_info": "Context describes what feature X is, but doesn't provide implementation steps or code examples"}}

Question: "What is X?"
Context: "X is a graph-based framework for building agents"
Response: {{"is_relevant": true, "is_sufficient": true, "missing_info": "None"}}

Question: "How do I use X with Y?"
Context: "Here's how to use X with Y: [detailed steps and code]"
Response: {{"is_relevant": true, "is_sufficient": true, "missing_info": "None"}}

Your analysis:"""

    try:
        response = llm.invoke(prompt).content
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)

        return {
            "is_relevant": result.get("is_relevant", True),
            "is_sufficient": result.get("is_sufficient", True),
            "missing_info": result.get("missing_info", "")
        }
    except Exception as e:
        return {
            "is_relevant": True,
            "is_sufficient": True,
            "missing_info": f"Validation error: {str(e)}"
        }


def refine_search_query(original_question: str, feedback: str) -> str:
    """Generate an improved search query based on feedback about missing information."""
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
        refined_query = refined_query.strip('"\'')
        return refined_query if refined_query else original_question
    except Exception as e:
        return original_question


def extract_keywords(question: str) -> list:
    """Extract key technical terms and concepts from a question for targeted searching."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = f"""Extract the key technical terms, concepts, or class names from this question that should be searched separately.

Question: {question}

Guidelines:
1. Extract specific class names, functions, or technical concepts (e.g., "StateGraph", "MessageGraph", "SqliteSaver")
2. Focus on specific LangGraph/LangChain features, NOT the general terms "LangGraph" or "LangChain" themselves
3. Extract 1-4 key terms maximum
4. If the question is simple with only one concept, return an empty list
5. Don't extract common words like "difference", "how", "what", etc.
6. NEVER extract "LangGraph" or "LangChain" as keywords - we're already searching LangGraph docs
7. Extract domain-specific concepts like "human-in-the-loop", "checkpointing", "persistence" when relevant

Respond in JSON format with:
- keywords: list of strings (the extracted key terms)

Examples:
Question: "What's the difference between StateGraph and MessageGraph?"
Response: {{"keywords": ["StateGraph", "MessageGraph"]}}

Question: "How do I use checkpointing with SqliteSaver?"
Response: {{"keywords": ["checkpointing", "SqliteSaver"]}}

Question: "Show me how to implement human-in-the-loop with LangGraph"
Response: {{"keywords": ["human-in-the-loop"]}}

Question: "What is StateGraph?"
Response: {{"keywords": []}}

Question: "Compare LangGraph and LangChain agents"
Response: {{"keywords": []}}

Your response:"""

    try:
        response = llm.invoke(prompt).content
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)
        keywords = result.get("keywords", [])

        if isinstance(keywords, list):
            return keywords
        else:
            return []
    except Exception as e:
        return []


def retrieve_with_keywords(question: str, keywords: list, mode: str = "offline", restrict_to_official: bool = True) -> str:
    """Retrieve documentation using both the original question and extracted keywords.

    Performs multiple searches and combines results, removing duplicates."""
    all_results = []
    seen_content = set() # to avoid duplicate sources

    try:
        if mode == "offline":
            original_result = retrieve_context(question)
        else:
            original_result = search_web(question, restrict_to_official=restrict_to_official)

        if original_result and original_result not in seen_content:
            all_results.append(f"=== Results for: {question} ===\n{original_result}")
            seen_content.add(original_result)
    except Exception as e:
        pass

    for keyword in keywords:
        try:
            if mode == "offline":
                keyword_result = retrieve_context(keyword)
            else:
                keyword_result = search_web(keyword, restrict_to_official=restrict_to_official)

            if keyword_result and keyword_result not in seen_content:
                all_results.append(f"=== Results for: {keyword} ===\n{keyword_result}")
                seen_content.add(keyword_result)
        except Exception as e:
            continue

    if all_results:
        return "\n\n".join(all_results)
    else:
        return "Error: Unable to retrieve any documentation"


def check_answer_completeness(question: str, answer: str) -> dict:
    """Evaluate the quality and completeness of a generated answer."""
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
        return {
            "quality_score": 7,
            "needs_improvement": False,
            "suggestions": f"Evaluation error: {str(e)}"
        }
