# LangGraph Helper Agent

An AI agent that helps developers work with LangGraph and LangChain. Supports both offline (local documentation) and online (web search) modes.

[Example Video](https://www.youtube.com/watch?v=MztJmr0hu2U&feature=youtu.be)

## Architecture

This system uses an **agentic architecture** where an LLM makes intelligent routing decisions rather than following a fixed workflow. The agent can iteratively refine its retrieval and responses based on quality validation.

### Agentic Flow

```
                    ┌──────────────┐
                    │    Router    │ ← LLM-driven decision maker
                    │  (decides)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────┐
        │                  │                  │              │
        ▼                  ▼                  ▼              ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐   ┌─────────┐
   │Extract  │       │Retrieve │       │ Respond │   │ Reflect │
   │Keywords │       │ (with   │       │ (with   │   │ (self-  │
   │         │       │validate)│       │ refine) │   │critique)│
   └────┬────┘       └────┬────┘       └────┬────┘   └────┬────┘
        │                 │                  │              │
        └─────────────────┴──────────────────┴──────────────┘
                          │
                          ▼
                      [Evaluate] → END
```

### Intelligent Agent Features

**1. Smart Routing** (`router_node`)
- **Documentation-First Approach**: Strongly biased toward retrieving official documentation for 95%+ of questions
- **Safety Gates**: Relevance check (filters non-LangGraph/LangChain questions) and safety check (detects jailbreak attempts)
- Only skips retrieval for trivial definitional questions (e.g., "What does LLM stand for?")
- Routes to extract_keywords, retrieve, respond, reflect, or end based on current state
- LLM makes intelligent decisions about when documentation is truly unnecessary
- **Loop Prevention**: Tracks router invocations (>15 = infinite loop detection)

**2. Keyword Extraction** (`extract_keywords_node`)
- Uses LLM to extract 1-4 technical terms from the question (e.g., "StateGraph", "SqliteSaver")
- Excludes generic terms ("LangGraph", "LangChain", "difference")
- Enables multi-query retrieval for comprehensive context coverage

**3. Validated Retrieval** (`retrieve_node`)
- **Multi-Query Strategy**: First attempt uses extracted keywords for parallel searches; subsequent attempts use refined single queries
- Retrieves documentation with up to 3 retry attempts
- Validates context quality using LLM (checks both relevance and sufficiency)
- Refines search queries if initial retrieval is insufficient
- **Online Mode Fallback**: Attempt 1 uses official docs only; Attempt 2 uses unrestricted web search
- Auto-fallback: online→offline if web search fails
- Saves context to `context.txt` (offline) or `sources.txt` (online)

**4. Adaptive Response** (`respond_node`)
- Generates answers from context or knowledge
- **Disclaimer System**: Adds warning when context is insufficient/irrelevant
- Incorporates refinement suggestions from reflection
- Iteratively improves based on quality feedback
- Saves `answer.md` and `chat.md` (prompt + answer)

**5. Self-Reflection** (`reflect_node`)
- Evaluates answer quality on 0-10 scale
- **Quality Thresholds**: 8-10 = Excellent (ACCEPT), 7 = Good (ACCEPT), 5-6 = Fair (REFINE if iterations remain), 0-4 = Poor (REFINE)
- Triggers regeneration if score < 7 and iterations < 3
- Sets `needs_refinement` flag and provides specific improvement suggestions
- Limited to 3 iterations to prevent infinite loops

### Agent Tools

The system uses 6 specialized LangChain tools:

| Tool | Purpose | Returns |
|------|---------|---------|
| `retrieve_documentation` | Fetch docs from offline/online sources | Retrieved text |
| `validate_context_quality` | Check if context is relevant & sufficient | `{is_relevant, is_sufficient, missing_info}` |
| `refine_search_query` | Generate improved query based on feedback | Refined query string |
| `extract_keywords` | Extract 1-4 technical terms from question | `{keywords: list}` |
| `retrieve_with_keywords` | Multi-query retrieval using extracted keywords | Combined context with section headers |
| `check_answer_completeness` | Score answer quality on 4 criteria (Completeness, Accuracy, Clarity, Practicality) | `{quality_score: 0-10, needs_improvement: bool, suggestions: str}` |

### State Graph Components

- **AgentState**: TypedDict tracking complete state including:
  - Core fields: `question`, `mode`, `context`, `answer`, `output_dir`
  - Evaluation: `evaluation_scores` (dict)
  - Iteration control: `retrieval_attempts`, `iteration`, `max_iterations`
  - Refinement: `needs_refinement`, `refinement_notes`
  - Routing: `next_action`, `last_node`, `node_history`
  - Keywords: `skip_retrieval`, `extracted_keywords`
  - Validation: `context_is_sufficient`, `context_is_relevant`
  - Quality: `quality_score`, `routing_error`
- **Conditional Routing**: Universal `route_by_next_action()` function based on LLM decisions, not fixed paths
- **Iteration Limits**: Max 3 iterations for both retrieval refinement and answer regeneration; Max 50 recursion limit safety
- **Agent Trace**: Saves decision-making history to `outputs/{timestamp}/agent_trace.json`
- **Graph Nodes**: 5 core nodes (router, extract_keywords, retrieve, respond, reflect) + optional evaluate node

## Project Structure

```
langgraph_helper_agent/
├── src/
│   ├── agent.py           # Agentic graph with conditional routing
│   ├── agent_nodes.py     # Router, extract_keywords, retrieve, respond, reflect nodes
│   ├── tools.py           # LangChain tools for retrieval & validation
│   ├── state.py           # AgentState TypedDict with agentic fields
│   ├── offline.py         # ChromaDB retrieval
│   ├── online.py          # Tavily search
│   ├── evaluation.py      # LLM-as-a-Judge metrics
│   ├── rate_limiter.py    # API quota management (10 RPM for Gemini)
│   └── tool_calling_agent.py  # Alternative ReAct agent (experimental)
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
├── examples/          # Example outputs and test cases
├── prepare_data.py    # Data preparation script
├── main.py            # CLI entry point with --verbose flag
├── test_keyword_extraction.py  # Testing keyword extraction
├── ARCHITECTURE.md    # Hybrid tool-calling design documentation
├── REFINEMENT_LOOPS.md  # Advanced refinement features
├── BUGFIX_REFINEMENT.md  # Implementation notes
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
- **GOOGLE_API_KEY** (required): Get from [Google AI Studio](https://aistudio.google.com/app/apikey) - Used for Gemini 2.0 Flash LLM
- **TAVILY_API_KEY** (optional): Get from [Tavily](https://tavily.com) - Only needed for online mode

```bash
# .env file example
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"  # Optional - only for online mode
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
- **Retrieval**: k=10 top documents using cosine similarity search
- **Performance**: Fast retrieval (<100ms typical) with no API rate limits

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

- **LLM**: Google Gemini 2.0 Flash (free tier, 15 RPM limit)
- **Rate Limiting**: Conservative 10 RPM limit enforced via `RateLimiter` class to prevent API exhaustion
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, local, no API needed)
- **Vector Store**: ChromaDB (file-based persistent client, no server needed)
  - Collection: "langgraph_docs"
  - Metadata tracking: source field ("langgraph" or "langchain")
- **Search**: Tavily API (1000 free searches/month)
  - Search depth: "advanced" (hardcoded for quality)
  - Domain restriction: langchain-ai.github.io, python.langchain.com (first attempt)
  - Fallback: Unrestricted web search (second attempt)
- **Framework**: LangGraph StateGraph for agent orchestration
- **Evaluation**: LLM-as-a-Judge pattern with Gemini for RAG metrics (faithfulness, answer relevancy, context precision)
- **Text Chunking**:
  - Splitter: RecursiveCharacterTextSplitter
  - Chunk size: 1000 characters
  - Chunk overlap: 200 characters
  - Batch size: 100 documents per batch for vectorstore building
  - Typical result: Thousands of chunks from full documentation

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
- Gemini free tier: 15 requests/minute (system uses conservative 10 RPM limit with automatic wait management)
- Tavily free tier: 1000 searches/month

**Verbose mode shows no agent decisions**
- Make sure you're using the `--verbose` flag: `python main.py --verbose "your question"`
- Agent trace is always saved to `outputs/{timestamp}/agent_trace.json` regardless of verbose mode

## Key Design Principles

1. **Documentation-First Philosophy**: 95%+ questions trigger retrieval for accuracy and grounding
2. **Centralized Routing**: Router node makes ALL routing decisions (not distributed across nodes)
3. **Quality Over Speed**: Iterative refinement loops prioritize answer quality
4. **Graceful Degradation**: Multiple fallback layers (online→offline, best-effort answers with disclaimers)
5. **Transparency**: Explicit disclaimers when context insufficient, detailed agent traces
6. **Safety-First**: Multiple validation gates (relevance, safety, loop prevention, iteration limits)
7. **Production-Ready**: Rate limiting, error handling, observability features built-in

## Additional Documentation

- **ARCHITECTURE.md**: Hybrid tool-calling design and trade-offs between ReAct autonomy vs. predictability
- **REFINEMENT_LOOPS.md**: Search refinement loop and quality refinement loop implementation details
- **BUGFIX_REFINEMENT.md**: Implementation notes on validation flow fixes and rate limiting

## Future Improvements
- Add more data sources (e.g., GitHub repos, StackOverflow)
- Use structured output parsing for LLM-as-a-Judge evaluation (langchain output parsers)
- Add testing node: if the user asks for code samples, run the code and verify correctness
- Add multi-agent collaboration for complex tasks
- Implement persistent memory with LangGraph checkpointing for multi-turn conversations
- Add visualization of agent decision tree
- Enhance keyword extraction with more sophisticated NLP techniques
