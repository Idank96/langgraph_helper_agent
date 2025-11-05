# LangGraph Helper Agent

An AI agent that helps developers work with LangGraph and LangChain. Supports both offline (local documentation) and online (web search) modes.








## Architecture

This system uses an **agentic architecture** where an LLM makes intelligent routing decisions rather than following a fixed workflow. The agent can iteratively refine its retrieval and responses based on quality validation.

### Agentic Flow

```
                    ┌──────────────┐
                    │    Router    │ ← LLM-driven decision maker
                    │  (decides)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │Retrieve │       │ Respond │       │ Reflect │
   │ (with   │       │ (with   │       │ (self-  │
   │validate)│       │ refine) │       │critique)│
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                  │
        └─────────────────┴──────────────────┘
                          │
                          ▼
                      [Evaluate] → END
```

### Intelligent Agent Features

**1. Smart Routing** (`router_node`)
- **Documentation-First Approach**: Strongly biased toward retrieving official documentation for 95%+ of questions
- Only skips retrieval for trivial definitional questions (e.g., "What does LLM stand for?")
- Routes to retrieve, respond, reflect, or end based on current state
- LLM makes intelligent decisions about when documentation is truly unnecessary

**2. Validated Retrieval** (`retrieve_node`)
- Retrieves documentation with up to 3 retry attempts
- Validates context quality using LLM
- Refines search queries if initial retrieval is insufficient
- Auto-fallback: online→offline if web search fails

**3. Adaptive Response** (`respond_node`)
- Generates answers from context or knowledge
- Incorporates refinement suggestions from reflection
- Iteratively improves based on quality feedback

**4. Self-Reflection** (`reflect_node`)
- Evaluates answer quality on 0-10 scale
- Triggers regeneration if score < 7
- Limited to 3 iterations to prevent infinite loops

### Agent Tools

The system uses 4 specialized tools:

| Tool | Purpose | Returns |
|------|---------|---------|
| `retrieve_documentation` | Fetch docs from offline/online sources | Retrieved text |
| `validate_context_quality` | Check if context is relevant & sufficient | `{is_relevant, is_sufficient, missing_info}` |
| `refine_search_query` | Generate improved query based on feedback | Refined query string |
| `check_answer_completeness` | Score answer quality and suggest improvements | `{quality_score, needs_improvement, suggestions}` |

### State Graph Components

- **AgentState**: Tracks question, mode, context, answer, evaluation scores, retrieval attempts, iteration count, refinement notes, and routing decisions
- **Conditional Routing**: Graph uses conditional edges based on LLM decisions, not fixed paths
- **Iteration Limits**: Max 3 iterations for both retrieval refinement and answer regeneration
- **Agent Trace**: Saves decision-making history to `outputs/{timestamp}/agent_trace.json`

## Project Structure

```
langgraph_helper_agent/
├── src/
│   ├── agent.py       # Agentic graph with conditional routing
│   ├── agent_nodes.py # Router, retrieve, respond, reflect nodes
│   ├── tools.py       # LangChain tools for retrieval & validation
│   ├── state.py       # AgentState with agentic fields
│   ├── offline.py     # ChromaDB retrieval
│   ├── online.py      # Tavily search
│   └── evaluation.py  # LLM-as-a-Judge metrics
├── data/
│   ├── raw/           # Downloaded llms.txt files
│   └── vectorstore/   # ChromaDB storage
├── outputs/
│   └── {timestamp}/
│       ├── answer.md       # Generated answer
│       ├── context.txt     # Retrieved context (offline mode)
│       ├── sources.txt     # Retrieved sources (online mode)
│       ├── chat.md         # Full prompt and answer for debugging
│       ├── agent_trace.json # Agent decision history
│       └── evaluation.json # Optional eval scores
├── prepare_data.py    # Data preparation script
├── main.py            # CLI entry point with --verbose flag
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

**Options:**
- **Default (incremental mode)**: `python prepare_data.py`
  - Resumes from existing vector store
  - Useful when recovering from rate limits or interrupted builds
- **Force rebuild**: `python prepare_data.py --force-rebuild`
  - Deletes existing collection and rebuilds from scratch
  - Use when you want fresh data or to fix corrupted vector store
  - Note: Both `--force-rebuild` and `--force_rebuild` work (argparse auto-converts)

You can also update data directly through the main script:
```bash
# Incremental update
python main.py --update_data "Your question here"

# Force complete rebuild
python main.py --update_data --force_rebuild "Your question here"
```


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

### Verbose Mode (Agent Decision Logging)

Enable verbose logging to see the agent's decision-making process in real-time:

```bash
python main.py --verbose "How do I add persistence to a LangGraph agent?"
```

**What you'll see with --verbose:**
- Router decisions (retrieve/respond/reflect/end)
- Retrieval attempts and validation results
- Context quality assessments (relevant/sufficient)
- Query refinement when needed
- Answer quality scores and improvement suggestions
- Iteration counts

**Example verbose output (documentation retrieval):**
```
━━━ ROUTER NODE ━━━
Current state: context=False, answer=False, skip_retrieval=False, iteration=0/3
→ Asking LLM: Should we retrieve documentation?
   LLM decision: RETRIEVE
→ Router decision: RETRIEVE

━━━ RETRIEVE NODE ━━━
Retrieval mode: 'offline', max attempts: 3

  Attempt 1/3
  Query: 'How do I add persistence to a LangGraph agent?'
  ✓ Retrieved 8432 characters
  Validation:
    - Is Relevant: ✓ Yes
    - Is Sufficient: ✓ Yes
  ✓ SUCCESS: Context quality acceptable after 1 attempt(s)

━━━ ROUTER NODE ━━━
Current state: context=True, answer=False, skip_retrieval=False, iteration=0/3
→ Router decision: RESPOND (have context, need answer)

━━━ RESPOND NODE ━━━
Generating answer using 8432 characters of retrieved context
  Invoking LLM to generate answer...
  ✓ Answer generated successfully (1247 characters)

━━━ ROUTER NODE ━━━
Current state: context=True, answer=True, skip_retrieval=False, iteration=0/3
→ Router decision: REFLECT (iteration 0/3)

━━━ REFLECT NODE ━━━
Evaluating answer quality using LLM critic (iteration 0/3)...

  Quality Assessment:
    Score: 8/10
    Rating: ✓ Excellent
    Decision: ACCEPT (quality threshold met)
  ✓ Answer is acceptable, proceeding to completion
```

**Benefits:**
- Understand how the agent makes decisions
- Debug retrieval and validation issues
- See quality scores and refinement loops
- Track iteration counts and limits

**Agent trace file:** All decisions are saved to `outputs/{timestamp}/agent_trace.json` regardless of verbose mode.

**When does the agent retrieve documentation vs answer from knowledge?**

The agent uses a documentation-first approach:

✅ **Will RETRIEVE documentation** (95%+ of questions):
- "How do I add persistence to a LangGraph agent?"
- "Show me an example of using StateGraph"
- "What's the difference between StateGraph and MessageGraph?"
- "How to implement human-in-the-loop?"
- "Best practices for error handling in LangGraph"

❌ **Will SKIP retrieval** (only trivial questions):
- "What does LLM stand for?"
- "What is an API?"

This ensures answers are always grounded in official documentation, meeting the assignment's core requirement.

### Evaluation with LLM-as-a-Judge

Enable evaluation using three key metrics:

**Metrics:**
- **Faithfulness**: Measures if the answer is grounded in the retrieved context (0.0-1.0)
- **Answer Relevancy**: Evaluates how well the answer addresses the question (0.0-1.0)
- **Context Precision**: Rates the quality of retrieved context for answering (0.0-1.0)

**Command Example:**
```bash
python main.py --evaluate "How do I add persistence to a LangGraph agent?"
```

**Combine with verbose mode:**
```bash
python main.py --verbose --evaluate "How do I add persistence to a LangGraph agent?"
```

**This will append the following to the output:**
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

### Debug Mode

In `main.py`: Set `debug = True` to enable debug mode with preset arguments and verbose logging.

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
  - `search_depth="advanced"` (hardcoded for higher quality results)
  - `include_domains=["langchain-ai.github.io", "python.langchain.com"]` to use official documentation
  - `max_results=10`
- **Error Handling**: Automatically falls back to offline mode if online search fails

## Data Freshness Strategy

### Offline Mode
**Initial setup**: Run `python prepare_data.py` to download docs and build index.

**Updating data**: Use the built-in `--update_data` flag:

**Option 1**: Incremental update (default - resumes from existing data)
```bash
python main.py --update_data
```

**Option 2**: Force complete rebuild (fresh start - recommended for ensuring data freshness)
```bash
python main.py --update_data --force_rebuild
```

**Option 3**: Update and then answer a question
```bash
# Incremental update + question
python main.py --update_data --mode offline "How do I add persistence to a LangGraph agent?"

# Force rebuild + question
python main.py --update_data --force_rebuild "How do I add persistence to a LangGraph agent?"
```

**Automation**: Schedule weekly updates with force rebuild for fresh data

*Linux/Mac (cron):*
```bash
0 0 * * 0 cd /path/to/project && python main.py --update_data --force_rebuild
```

*Windows (Task Scheduler):*
```powershell
schtasks /create /tn "Update LangGraph Docs" /tr "python C:\path\to\project\main.py --update_data --force_rebuild" /sc weekly /d SUN /st 00:00
```

**Alternative method**:
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

# Example questions
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
- **Text Chunking**:
  - Chunk size: 1000 characters
  - Chunk overlap: 200 characters
  - Batch size: 100 documents per batch for vectorstore building

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

## Future Improvements
- Add more data sources (e.g., GitHub repos, StackOverflow)
- Use structured output parsing for LLM-as-a-Judge evaluation (langchain output parsers)
- Add testing node: if the user asks for code samples, run the code and verify correctness
- Add multi-agent collaboration for complex tasks
- Implement persistent memory with LangGraph checkpointing for multi-turn conversations
- Add visualization of agent decision tree
