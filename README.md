# LangGraph Helper Agent

An AI agent that helps developers work with LangGraph and LangChain. Supports both offline (local documentation) and online (web search) modes.



https://github.com/user-attachments/assets/1a149a59-8cb2-492a-bb85-a4e85ef7596e




## Architecture

```
User Question → Retrieve Node → Respond Node → [Evaluate Node] → Answer
                     ↓
            [Offline: ChromaDB]
            [Online: Tavily API]
```

**State Graph Components:**
- `AgentState`: Tracks question, mode, context, answer, and evaluation scores
- `retrieve_node`: Gets context from ChromaDB (offline) or Tavily (online)
- `respond_node`: Uses Gemini to generate answer from context
- `evaluate_node`: Optional LLM-as-a-Judge evaluation (faithfulness, relevancy, precision)

## Project Structure

```
langgraph_helper_agent/
├── src/
│   ├── agent.py       # LangGraph state graph
│   ├── state.py       # AgentState definition
│   ├── offline.py     # ChromaDB retrieval
│   ├── online.py      # Tavily search
│   └── evaluation.py  # LLM-as-a-Judge metrics
├── data/
│   ├── raw/           # Downloaded llms.txt files
│   └── vectorstore/   # ChromaDB storage
├── prepare_data.py    # Data preparation script
├── main.py            # CLI entry point
├── evaluate_demo.py   # Evaluation demo script
└── .env               # API keys (not in git)
```

## Setup

### 1. Install Dependencies

#### Option 1: Using Conda
```bash
conda create -n helper_agent_env python=3.12
conda activate helper_agent_env
pip install -r requirements.txt
```

#### Option 2: Using venv
```bash
python3.12 -m venv helper_agent_env
source helper_agent_env/bin/activate  # On Windows: helper_agent_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file in the root directory and add your keys:
- `GOOGLE_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com)
```bash
# For example
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### 3. Prepare Offline Data

Run this once to download documentation and build the vector store:

```bash
python prepare_data.py
```

This downloads:
- LangGraph docs: `https://langchain-ai.github.io/langgraph/llms-full.txt`
- LangChain docs: `https://python.langchain.com/llms.txt`

And creates a ChromaDB vector store in `data/vectorstore/`.

build_vectorstore(force_rebuild=False) - Incrementally updates the vector store with new data when rate limits hit becuase of API constraints.
build_vectorstore(force_rebuild=True) - Forces a complete rebuild of the vector store


## Usage

### Offline Mode (default)

```bash
python main.py "How do I add persistence to a LangGraph agent?"
```

Or explicitly:

```bash
python main.py --mode offline "What's the difference between StateGraph and MessageGraph?"
```

### Online Mode

```bash
python main.py --mode online "What are the latest LangGraph features?"
```

Or set via environment variable:

```bash
export AGENT_MODE=online  # windows: set AGENT_MODE=online
python main.py "How do I implement human-in-the-loop?"
```

### Updating Documentation Data

**Update only** (downloads latest docs and rebuilds vector store):
```bash
python main.py --update_data
```

**Update and answer** (updates data, then answers your question):
```bash
python main.py --update_data --mode offline "How do I add persistence to a LangGraph agent?"
```

### RAG Evaluation with LLM-as-a-Judge

Enable automatic evaluation of RAG quality using three key metrics:

```bash
python main.py --evaluate "How do I add persistence to a LangGraph agent?"
```

**Metrics:**
- **Faithfulness**: Measures if the answer is grounded in the retrieved context (0.0-1.0)
- **Answer Relevancy**: Evaluates how well the answer addresses the question (0.0-1.0)
- **Context Precision**: Rates the quality of retrieved context for answering (0.0-1.0)

**Output:**
```
============================================================
LLM-AS-A-JUDGE EVALUATION SCORES
============================================================
  Faithfulness        : 0.92
  Answer Relevancy    : 0.88
  Context Precision   : 0.85
============================================================
```

Scores are also saved to `outputs/{timestamp}/evaluation.json`.

**Run standalone demo:**
```bash
python evaluate_demo.py
```

### Debug Mode

In `main.py:13`: Set `debug = True` to use the hardcoded variables.

## Operating Modes

### Offline Mode
- **How it works**: Uses ChromaDB RAG with HuggingFace embeddings to retrieve relevant chunks from local documentation
- **Data sources**:
  - LangGraph: `https://langchain-ai.github.io/langgraph/llms-full.txt`
  - LangChain: `https://python.langchain.com/llms.txt`
- **Embeddings**: Local HuggingFace sentence-transformers/all-MiniLM-L6-v2 (no API key required)

### Online Mode
- **How it works**: Uses Tavily search API to find current information from the web, specifically restricted to LangGraph and LangChain documentation sites only.
- **Configuration**:
  - `search_depth="advanced"` for higher quality results
  - `include_domains=["langchain-ai.github.io", "python.langchain.com"]` to use official documentation
  - `max_results=5` for comprehensive coverage
- **Error Handling**: Automatically falls back to offline mode if online search fails

## Data Freshness Strategy

### Offline Mode
**Initial setup**: Run `python prepare_data.py` to download docs and build index.

**Updating data**: Use the built-in `--update_data` flag:

**Option 1**: Update only
```bash
python main.py --update_data
```

**Option 2**: Update and then answer a question
```bash
python main.py --update_data --mode offline "How do I add persistence to a LangGraph agent?"
```

**Automation**: Schedule weekly updates

*Linux/Mac (cron):*
```bash
0 0 * * 0 cd /path/to/project && python main.py --update_data
```

*Windows (Task Scheduler):*
```powershell
schtasks /create /tn "Update LangGraph Docs" /tr "python C:\path\to\project\main.py --update_data" /sc weekly /d SUN /st 00:00
```

**Alternative method** (manual cleanup):
```bash
rm -rf data/raw/* data/vectorstore/
python prepare_data.py
```

### Online Mode
Always fetches current information via Tavily search. No manual updates needed.



## Examples

```bash
# Update offline documentation data (option 1: update only)
python main.py --update_data

# Update and answer question (option 2: update + query)
python main.py --update_data --mode offline "How do I add persistence to a LangGraph agent?"

# Example questions your agent should handle
python main.py "How do I add persistence to a LangGraph agent?"
python main.py "What's the difference between StateGraph and MessageGraph?"
python main.py "Show me how to implement human-in-the-loop with LangGraph"
python main.py "How do I handle errors and retries in LangGraph nodes?"
python main.py "What are best practices for state management in LangGraph?"

# Run with evaluation
python main.py --evaluate "How do I add persistence to a LangGraph agent?"
python main.py --mode online --evaluate "What are the latest LangGraph features?"
```

## Output Example

```
(helper_agent_env) C:\idan\langgraph_helper_agent>python main.py --evaluate --mode online "How do I handle errors and retries in LangGraph nodes?"

Question: "How do I handle errors and retries in LangGraph nodes?"

Mode: online

Evaluate: True

You can handle errors and retries in LangGraph nodes by using the `retryPolicy` parameter when adding a node to the graph. This allows you to define conditions under which a node should be retried.

Here's how you can add a retry policy to a node:

import { RetryPolicy } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";

// Define a function for your node
async function queryDatabase() {
  // ... your database query logic ...
}

// Create a new graph
const graph = new StateGraph({});

// Add a node with a retry policy
graph.addNode("query_database", queryDatabase, {
  retryPolicy: {
    retryOn: (e: any): boolean => {
      // Define the condition for retrying the node
      return e instanceof Error; // Retry if an error occurs
    },
  },
});

In this example, the `retryPolicy` is defined with a `retryOn` function. This function checks if the error `e` is an instance of the `Error` class. If it is, the node will be retried. You can customize the `retryOn` function to implement more specific retry conditions based on the type of error or other criteria.

Additionally, LangChain's `RunnableRetry` class (Python) offers more advanced retry mechanisms, including exponential backoff and jitter.  While the provided documents don't show a direct equivalent in LangGraphJS, the underlying principles are similar.  You'd configure the `retryPolicy` to use a similar strategy.

from langchain_core.runnables import RunnableRetry

# Example using RunnableRetry (Python)
runnable = RunnableRetry.with_retry(
    retry_if_exception_type=(Exception,),  # Retry on any Exception
    stop_after_attempt=3  # Retry up to 3 times
)

Key points:

*   **`retryPolicy` parameter:**  Use this when adding a node to the graph to configure retry behavior.
*   **`retryOn` function:** Define the conditions for retrying within the `retryPolicy`.
*   **Error Handling:** Implement robust error handling within your node functions to catch and appropriately handle exceptions.
*   **Custom Conditions:** Tailor the `retryOn` function to retry based on specific error types or conditions relevant to your application.

============================================================
LLM-AS-A-JUDGE EVALUATION SCORES
============================================================
  Faithfulness        : 0.95
  Answer Relevancy    : 0.95
  Context Precision   : 0.85
============================================================
```

## Technical Details

- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (local, no API needed)
- **Vector Store**: ChromaDB (file-based, no server needed)
- **Search**: Tavily API (1000 free searches/month)
- **Framework**: LangGraph for agent orchestration
- **Evaluation**: LLM-as-a-Judge pattern with Gemini for RAG metrics (faithfulness, answer relevancy, context precision)

## Troubleshooting

**"ERROR: GOOGLE_API_KEY not found"**
- Add `GOOGLE_API_KEY=your_key` to `.env` file

**"ERROR: TAVILY_API_KEY required for online mode"**
- Get free key from https://tavily.com and add to `.env`

**"Vector store not found" in offline mode**
- Run `python prepare_data.py` first

**"[Note: Online search unavailable, using offline docs]" appears in answer**
- Online mode was selected but Tavily search failed
- System automatically fell back to offline documentation
- Possible causes: invalid API key, rate limit exceeded, network error, or no results found

**Rate limits**
- Gemini free tier: 15 requests/minute
- Tavily free tier: 1000 searches/month
