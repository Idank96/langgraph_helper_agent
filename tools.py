"""
Tools for the LangGraph/LangChain Helper React Agent.

This module contains 5 specialized tools that enable intelligent documentation retrieval,
context validation, query refinement, and answer quality assessment.
"""

import os
import json
from typing import Dict, List, Any, Union
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb import PersistentClient
from tavily import TavilyClient


# Initialize shared resources
def get_embeddings():
    """Get HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def get_llm():
    """Get Gemini LLM for validation tasks."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


# Tool 1: Offline Documentation Retrieval
@tool
def retrieve_offline_documentation(query: str) -> str:
    """
    Retrieve documentation from the local ChromaDB vector store.

    Use this tool when in offline mode or when you need fast, reliable retrieval
    from the indexed LangGraph and LangChain documentation.

    Args:
        query: The search query to find relevant documentation

    Returns:
        Retrieved documentation as formatted text with sources
    """
    try:
        embeddings = get_embeddings()
        client = PersistentClient(path="./data/vectorstore")
        collection = client.get_or_create_collection(
            name="langgraph_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # Get query embedding
        query_embedding = embeddings.embed_query(query)

        # Search collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

        if not results['documents'] or not results['documents'][0]:
            return "No relevant documentation found in the local database."

        # Format results
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)

        formatted_docs = []
        for i, (doc, metadata) in enumerate(zip(docs, metadatas), 1):
            source = metadata.get('source', 'Unknown')
            formatted_docs.append(f"--- Document {i} (Source: {source}) ---\n{doc}\n")

        return "\n".join(formatted_docs)

    except Exception as e:
        return f"Error retrieving offline documentation: {str(e)}"


# Tool 2: Online Documentation Retrieval
@tool
def retrieve_online_documentation(query: str, restrict_to_official: bool = True) -> str:
    """
    Retrieve documentation from the web using Tavily search API.

    Use this tool when in online mode or when local documentation is insufficient.
    Can search official docs only or unrestricted web search.

    Args:
        query: The search query to find relevant documentation
        restrict_to_official: If True, restrict to official LangChain/LangGraph domains.
                            If False, search the entire web.

    Returns:
        Retrieved documentation as formatted text with source URLs
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not found. Cannot perform online search."

        client = TavilyClient(api_key=api_key)

        # Configure search domains
        include_domains = None
        if restrict_to_official:
            include_domains = [
                "langchain-ai.github.io",
                "python.langchain.com",
                "docs.langchain.com"
            ]

        # Perform search
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            include_domains=include_domains
        )

        if not response.get('results'):
            return "No relevant documentation found online."

        # Format results
        formatted_results = []
        for i, result in enumerate(response['results'], 1):
            url = result.get('url', 'Unknown')
            content = result.get('content', '')
            formatted_results.append(f"--- Source {i}: {url} ---\n{content}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error retrieving online documentation: {str(e)}"


# Tool 3: Context Quality Validation
@tool
def validate_context_quality(input_data: str) -> Dict[str, Any]:
    """
    Validate if the retrieved context is relevant and sufficient to answer the question.

    Use this tool AFTER retrieving documentation to assess whether you have
    enough information to provide a high-quality answer.

    Args:
        input_data: JSON string with question and context fields.
                   Example: {{"question": "How to use StateGraph?", "context": "StateGraph is..."}}

    Returns:
        Dictionary with:
            - is_relevant (bool): Whether context is relevant to the question
            - is_sufficient (bool): Whether context has enough info to answer
            - missing_info (str): Description of what's missing (if insufficient)
    """
    try:
        # Parse input
        try:
            params = json.loads(input_data)
        except json.JSONDecodeError:
            return {
                "is_relevant": False,
                "is_sufficient": False,
                "missing_info": f"Invalid input format. Expected JSON with 'question' and 'context' fields."
            }

        question = params.get('question', '')
        context = params.get('context', '')

        if not question or not context:
            return {
                "is_relevant": False,
                "is_sufficient": False,
                "missing_info": "Both 'question' and 'context' are required fields."
            }

        llm = get_llm()

        prompt = f"""You are evaluating the quality of retrieved documentation context.

Question: {question}

Retrieved Context:
{context}

Evaluate the context on two criteria:

1. RELEVANCE: Does the context directly relate to the question?
   - Consider: Is it about the right topic/concept?
   - For comparison questions: Does it cover ALL items being compared?
   - For "how-to" questions: Does it address the specific task?

2. SUFFICIENCY: Does the context contain enough information to fully answer the question?
   - Consider: Are there code examples if needed?
   - Are key concepts explained?
   - For comparison questions: Are differences/similarities clearly described?
   - For implementation questions: Are steps/methods provided?

CRITICAL RULES:
- For comparison questions (e.g., "difference between X and Y"): Context MUST discuss BOTH X and Y
- For "how-to" questions: Context MUST include practical implementation details or examples
- If context only mentions a concept but doesn't explain it: Mark as INSUFFICIENT

Respond in this exact format:
RELEVANT: [yes/no]
SUFFICIENT: [yes/no]
MISSING_INFO: [If SUFFICIENT=no, describe what specific information is missing. If SUFFICIENT=yes, write "None"]

Your evaluation:"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse response
        is_relevant = "RELEVANT: yes" in content.lower()
        is_sufficient = "SUFFICIENT: yes" in content.lower()

        # Extract missing info
        missing_info = "None"
        if "MISSING_INFO:" in content:
            missing_info = content.split("MISSING_INFO:")[-1].strip()

        return {
            "is_relevant": is_relevant,
            "is_sufficient": is_sufficient,
            "missing_info": missing_info
        }

    except Exception as e:
        return {
            "is_relevant": False,
            "is_sufficient": False,
            "missing_info": f"Error during validation: {str(e)}"
        }


# Tool 4: Query Refinement
@tool
def refine_search_query(input_data: str) -> str:
    """
    Generate an improved search query based on validation feedback.

    Use this tool when context validation indicates missing information.
    Creates a more targeted query to find the specific information needed.

    Args:
        input_data: JSON string with original_question and feedback fields.
                   Example: {{"original_question": "How to use StateGraph?", "feedback": "Missing code examples"}}

    Returns:
        Refined search query (5-10 words) targeting the missing information
    """
    try:
        # Parse input
        try:
            params = json.loads(input_data)
        except json.JSONDecodeError:
            # Try to extract from string if it's not valid JSON
            return input_data  # Fallback to using raw input as query

        original_question = params.get('original_question', '')
        feedback = params.get('feedback', '')

        if not original_question:
            return input_data  # Fallback

        llm = get_llm()

        prompt = f"""You are refining a search query to find missing information.

Original Question: {original_question}

Missing Information: {feedback}

Create a refined search query that specifically targets the missing information.
The query should be:
- Concise (5-10 words)
- Focused on the missing details
- Use technical terms from LangGraph/LangChain domain
- Phrased to retrieve practical implementation details

Examples:
- Original: "How do I use checkpointing?"
  Missing: "No code examples showing checkpoint configuration"
  Refined: "LangGraph checkpointing code example configuration"

- Original: "Difference between MemorySaver and SqliteSaver?"
  Missing: "Only has info about MemorySaver, nothing on SqliteSaver"
  Refined: "SqliteSaver checkpoint persistence LangGraph"

Your refined query (just the query, nothing else):"""

        response = llm.invoke(prompt)
        refined_query = response.content.strip()

        # Clean up the query (remove quotes if present)
        refined_query = refined_query.strip('"\'')

        return refined_query

    except Exception as e:
        return original_question  # Fallback to original question


# Tool 5: Keyword Extraction
@tool
def extract_keywords(question: str) -> Dict[str, List[str]]:
    """
    Extract 1-4 technical keywords from a question for multi-query retrieval.

    Use this tool to identify specific technical terms, classes, functions, or concepts
    that should be searched independently for comprehensive documentation coverage.

    Args:
        question: The user's question

    Returns:
        Dictionary with 'keywords' list containing 1-4 technical terms
    """
    try:
        llm = get_llm()

        prompt = f"""Extract 1-4 specific technical keywords from this question for documentation search.

Question: {question}

RULES:
- Focus on: specific classes, functions, concepts, features (e.g., "StateGraph", "SqliteSaver", "checkpointing")
- EXCLUDE: generic terms like "LangGraph", "LangChain", "difference", "how", "what"
- Prioritize: Technical terminology that would appear in documentation
- Limit: 1-4 keywords maximum

Examples:
- "How do I use StateGraph with checkpointing?" → ["StateGraph", "checkpointing"]
- "Difference between MemorySaver and SqliteSaver?" → ["MemorySaver", "SqliteSaver"]
- "What is prebuilt ReAct agent?" → ["ReAct", "prebuilt"]
- "How to add tools to an agent?" → ["tools", "agent"]

Respond with ONLY the keywords, comma-separated, no other text:"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse keywords
        keywords = [k.strip().strip('"\'') for k in content.split(',')]
        keywords = [k for k in keywords if k]  # Remove empty strings
        keywords = keywords[:4]  # Limit to 4

        return {"keywords": keywords}

    except Exception as e:
        return {"keywords": []}


# Helper function for multi-query retrieval logic
def _retrieve_with_keywords_impl(keywords: List[str], question: str = "", mode: str = "offline",
                                  restrict_to_official: bool = True) -> str:
    """Internal implementation of multi-query retrieval."""
    try:
        all_results = []
        seen_content = set()

        # Add base question as a query if provided
        queries = ([question] if question else []) + keywords

        for query in queries:
            if mode == "offline":
                result = retrieve_offline_documentation(query)
            else:
                result = retrieve_online_documentation(query, restrict_to_official)

            # Avoid duplicate content
            if result and result not in seen_content:
                seen_content.add(result)
                all_results.append(f"\n=== Results for: {query} ===\n{result}")

        if not all_results:
            return "No documentation found for any of the queries."

        return "\n".join(all_results)

    except Exception as e:
        return f"Error in multi-query retrieval: {str(e)}"


# Tool 6: Multi-Query Retrieval with Keywords
@tool
def retrieve_with_keywords(input_data: Union[str, Dict[str, Any]]) -> str:
    """
    Perform parallel retrieval using multiple keyword-based queries and combine results.

    Use this for the FIRST retrieval attempt to get comprehensive documentation coverage.
    Searches for each keyword separately and combines unique results.

    Args:
        input_data: Either a JSON string or dict containing:
            - keywords: List of technical keywords to search for (REQUIRED)
            - question: The original user question (optional)
            - mode: "offline" or "online" (default: "offline")
            - restrict_to_official: For online mode, whether to restrict to official docs (default: True)

    Returns:
        Combined documentation from all keyword searches

    Example input:
        {"keywords": ["StateGraph", "checkpointing"], "question": "How do I use StateGraph?", "mode": "offline"}
    """
    try:
        # Parse input if it's a string
        if isinstance(input_data, str):
            # Try to parse as JSON
            try:
                params = json.loads(input_data)
            except json.JSONDecodeError:
                # Try to evaluate as Python dict literal
                try:
                    params = eval(input_data)
                except:
                    return f"Error: Could not parse input. Expected JSON string or dict, got: {input_data[:100]}"
        else:
            params = input_data

        # Extract parameters with defaults
        keywords = params.get('keywords', [])
        question = params.get('question', '')
        mode = params.get('mode', 'offline')
        restrict_to_official = params.get('restrict_to_official', True)

        # Validate keywords
        if not isinstance(keywords, list):
            return f"Error: 'keywords' must be a list, got {type(keywords)}"
        if not keywords:
            return "Error: 'keywords' list cannot be empty"

        # Call internal implementation
        return _retrieve_with_keywords_impl(keywords, question, mode, restrict_to_official)

    except Exception as e:
        return f"Error in retrieve_with_keywords: {str(e)}"


# Tool 5: Answer Quality Check
@tool
def check_answer_completeness(input_data: str) -> Dict[str, Any]:
    """
    Evaluate the quality and completeness of a generated answer.

    Use this tool AFTER generating an answer to self-assess quality and determine
    if refinement is needed.

    Args:
        input_data: JSON string with question and answer fields.
                   Example: {{"question": "What is StateGraph?", "answer": "StateGraph is a class..."}}

    Returns:
        Dictionary with:
            - quality_score (int): Score from 0-10
            - needs_improvement (bool): Whether answer should be regenerated
            - suggestions (str): Specific improvement suggestions
    """
    try:
        # Parse input
        try:
            params = json.loads(input_data)
        except json.JSONDecodeError:
            return {
                "quality_score": 0,
                "needs_improvement": True,
                "suggestions": "Invalid input format. Expected JSON with 'question' and 'answer' fields."
            }

        question = params.get('question', '')
        answer = params.get('answer', '')

        if not question or not answer:
            return {
                "quality_score": 0,
                "needs_improvement": True,
                "suggestions": "Both 'question' and 'answer' are required fields."
            }

        llm = get_llm()

        prompt = f"""Evaluate the quality of this answer on a scale of 0-10.

Question: {question}

Answer:
{answer}

Evaluate on these criteria:

1. COMPLETENESS (0-3): Does it fully address all parts of the question?
2. ACCURACY (0-3): Is the information technically correct based on LangChain/LangGraph?
3. CLARITY (0-2): Is it well-structured and easy to understand?
4. PRACTICALITY (0-2): Does it include actionable details, code examples when relevant?

Scoring guidance:
- 8-10: Excellent - comprehensive, accurate, clear, practical
- 7: Good - adequate but minor improvements possible
- 5-6: Fair - missing key details or clarity issues
- 0-4: Poor - incomplete, unclear, or potentially inaccurate

Respond in this exact format:
SCORE: [0-10]
NEEDS_IMPROVEMENT: [yes/no]
SUGGESTIONS: [Specific suggestions for improvement, or "None" if score >= 7]

Your evaluation:"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse response
        score = 5  # Default
        if "SCORE:" in content:
            try:
                score_str = content.split("SCORE:")[1].split("\n")[0].strip()
                score = int(score_str)
            except:
                pass

        needs_improvement = "NEEDS_IMPROVEMENT: yes" in content.lower()

        suggestions = "None"
        if "SUGGESTIONS:" in content:
            suggestions = content.split("SUGGESTIONS:")[-1].strip()

        return {
            "quality_score": score,
            "needs_improvement": needs_improvement,
            "suggestions": suggestions
        }

    except Exception as e:
        return {
            "quality_score": 5,
            "needs_improvement": False,
            "suggestions": f"Error during evaluation: {str(e)}"
        }
