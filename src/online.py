import os
from tavily import TavilyClient


def search_web(question: str, max_results: int = 5) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found")

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=question,
            search_depth="basic", # Could use advanced
            max_results=max_results,
            include_domains=["langchain-ai.github.io", "python.langchain.com"],
            include_answer=False
        )

        results_list = response.get("results", [])
        if not results_list:
            raise Exception("No search results found")

        formatted = [f"Source: {r['url']}\n{r['content']}" for r in results_list if r.get("content")]
        if not formatted:
            raise Exception("No usable content in results")

        return "\n\n".join(formatted)

    except Exception as e:
        raise Exception(f"Tavily search failed: {str(e)}")
