import os
from tavily import TavilyClient


def search_web(question: str, max_results: int = 10, restrict_to_official: bool = True) -> str:
    """Search the web for documentation.

    Args:
        question: The search query
        max_results: Maximum number of results to return
        restrict_to_official: If True, only search official LangChain docs. If False, search anywhere.

    Returns:
        Formatted search results as string
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found")

    try:
        client = TavilyClient(api_key=api_key)

        # Build search parameters
        search_params = {
            "query": question,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False
        }

        # Only restrict domains if restrict_to_official is True
        if restrict_to_official:
            search_params["include_domains"] = ["langchain-ai.github.io", "python.langchain.com"]

        response = client.search(**search_params)

        results_list = response.get("results", [])
        if not results_list:
            raise Exception("No search results found")

        formatted = [f"Source: {r['url']}\n{r['content']}" for r in results_list if r.get("content")]
        if not formatted:
            raise Exception("No usable content in results")

        return "\n\n".join(formatted)

    except Exception as e:
        raise Exception(f"Tavily search failed: {str(e)}")
