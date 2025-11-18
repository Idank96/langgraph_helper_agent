# LangGraph/LangChain Helper - ReAct Agent

An intelligent documentation assistant powered by LangChain's ReAct (Reasoning + Acting) agent architecture. This agent helps developers find accurate, comprehensive answers to questions about LangGraph and LangChain frameworks.

## Features

- **Intelligent Documentation Retrieval**: Searches official LangGraph and LangChain documentation
- **Dual Retrieval Modes**:
  - **Offline**: Fast local retrieval using ChromaDB vector store
  - **Online**: Real-time web search using Tavily API
- **Quality Validation**: Automatically validates context relevance and sufficiency
- **Iterative Refinement**: Refines search queries and regenerates answers for optimal quality
- **Safety Gates**: Blocks jailbreak attempts and filters irrelevant questions
- **Self-Reflection**: Evaluates answer quality and improves iteratively
- **Comprehensive Logging**: Saves answers, reasoning traces, and conversation history

## Architecture

### ReAct Agent Workflow

The agent follows a sophisticated 6-phase workflow encoded in a comprehensive system prompt:

```
Phase 1: Safety & Relevance Gates
    ↓
Phase 2: Documentation-First Decision
    ↓
Phase 3: Multi-Query Retrieval (keyword extraction + parallel searches)
    ↓
Phase 4: Context Validation & Iterative Refinement (max 3 attempts)
    ↓
Phase 5: Answer Generation (with disclaimer system)
    ↓
Phase 6: Self-Reflection & Improvement (max 3 iterations)
```

### Key Components

- **agent.py**: ReAct agent setup with comprehensive system prompt
- **tools.py**: 7 specialized tools for retrieval, validation, and quality assessment
- **main.py**: CLI interface with argument parsing and output management
- **prepare_data.py**: Documentation download and vector store preparation

### Tools (7 total)

1. **retrieve_offline_documentation**: ChromaDB vector search for local docs
2. **retrieve_online_documentation**: Tavily web search (official docs or unrestricted)
3. **validate_context_quality**: LLM-based context relevance and sufficiency check
4. **refine_search_query**: Generates improved queries based on validation feedback
5. **extract_keywords**: Extracts 1-4 technical terms for multi-query retrieval
6. **retrieve_with_keywords**: Parallel retrieval using multiple keyword queries
7. **check_answer_completeness**: Self-evaluation of answer quality (0-10 score)

## Installation

### Prerequisites

- Python 3.9+
- Google AI Studio API key (required)
- Tavily API key (optional, for online mode)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_ai_studio_key_here
   TAVILY_API_KEY=your_tavily_key_here  # Optional, for online mode
   ```

   - Get Google AI Studio API key: https://makersuite.google.com/app/apikey
   - Get Tavily API key: https://tavily.com (1000 free searches/month)

4. **Prepare documentation** (for offline mode):
   ```bash
   python prepare_data.py
   ```

   This will:
   - Download LangGraph and LangChain documentation (~970 KB total)
   - Chunk documents (1000 chars, 200 overlap)
   - Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Build ChromaDB vector store at `data/chroma/`

   **Force rebuild** (if needed):
   ```bash
   python prepare_data.py --force-rebuild
   ```

## Usage

### Basic Usage

```bash
# Offline mode (default) - uses local vector store
python main.py "How do I use StateGraph with checkpointing?"

# Online mode - uses Tavily web search
python main.py --mode online "What's new in LangGraph 1.0?"

# Enable verbose output to see agent reasoning
python main.py --verbose "Difference between MemorySaver and SqliteSaver?"

# Don't save outputs to files
python main.py --no-save "How to create a ReAct agent?"
```

### Command-Line Options

```
positional arguments:
  question              Your question about LangGraph or LangChain

options:
  -h, --help            Show help message
  --mode {offline,online}
                        Retrieval mode (default: offline)
  --verbose             Print agent reasoning steps and tool calls
  --no-save             Don't save outputs to files
```

### Example Questions

**Implementation Questions**:
- "How do I implement checkpointing in LangGraph?"
- "Show me how to create a ReAct agent with tools"
- "How to use SqliteSaver for persistence?"

**Comparison Questions**:
- "What's the difference between MemorySaver and SqliteSaver?"
- "StateGraph vs LangChain LCEL - which should I use?"

**Conceptual Questions**:
- "What is prebuilt in LangGraph?"
- "Explain how agent state works in LangGraph"

**Troubleshooting**:
- "Why is my StateGraph not saving checkpoints?"
- "Error: 'NoneType' object has no attribute 'config' in LangGraph"

## Output Files

Each query saves outputs to `outputs/YYYYMMDD_HHMMSS/`:

- **answer.md**: Final answer with question
- **agent_trace.json**: Detailed trace of agent reasoning (tool calls, observations)
- **chat.md**: Full conversation log with all intermediate steps

## How It Works

### 1. Safety & Relevance Filtering

The agent first checks if the question is:
- Safe (no jailbreak attempts, prompt injection)
- Relevant (about LangGraph/LangChain frameworks)

**Rejected questions receive a polite explanation.**

### 2. Documentation-First Philosophy

Unless the question is trivial, the agent retrieves documentation to ensure accuracy.

**First retrieval attempt**:
- Extracts technical keywords (e.g., "StateGraph", "checkpointing")
- Performs parallel searches for each keyword
- Combines unique results for comprehensive coverage

### 3. Context Validation

After retrieval, the agent validates:
- **Relevance**: Does context relate to the question?
- **Sufficiency**: Does context have enough info to answer fully?

**If context is insufficient**:
- Refines search query based on missing information
- Retrieves again (max 3 attempts)
- For online mode: Falls back to unrestricted web search if official docs fail

### 4. Answer Generation

Generates a comprehensive answer that:
- Stays grounded in retrieved documentation
- Includes code examples when available
- Uses clear structure and formatting
- Adds **disclaimer** if context was insufficient

### 5. Self-Reflection

After generating an answer, the agent:
- Evaluates quality on 4 criteria (completeness, accuracy, clarity, practicality)
- Scores from 0-10
- Regenerates if score < 7 and iterations remain (max 3)

### Workflow Example

```
User: "How do I use StateGraph with checkpointing?"

Phase 1: ✓ Safe and relevant

Phase 2: Needs documentation retrieval

Phase 3:
  - Extract keywords: ["StateGraph", "checkpointing"]
  - Multi-query retrieval → 10 documents retrieved

Phase 4:
  - Validate context → Relevant: Yes, Sufficient: Yes
  - ✓ Context is good, proceed

Phase 5:
  - Generate answer with code examples from context

Phase 6:
  - Check completeness → Score: 9/10
  - ✓ Quality excellent, done!

Output: Comprehensive answer with StateGraph checkpointing examples
```

## Architecture Comparison

### vs. Original LangGraph Implementation

| Aspect | LangGraph Version | ReAct Version |
|--------|------------------|---------------|
| **Orchestration** | StateGraph with 5 nodes + conditional routing | ReAct agent with AgentExecutor |
| **Decision Logic** | Explicit router_node with LLM-based routing | Encoded in system prompt, LLM decides |
| **State Management** | TypedDict with 21 fields | Agent's internal reasoning trace |
| **Complexity** | 8 Python files, explicit graph structure | 3 core files, prompt-driven |
| **Workflow** | Node-to-node transitions via edges | Tool selection via ReAct reasoning |
| **Observability** | agent_trace.json with state transitions | intermediate_steps with tool calls |
| **Flexibility** | Structured, predictable flow | More flexible, LLM-guided decisions |

**Key Insight**: The ReAct version achieves equivalent functionality through a comprehensive system prompt that encodes the same workflow logic as the LangGraph nodes and routing decisions.

## Design Principles

1. **Documentation-First**: Bias toward retrieving official docs for accuracy
2. **Quality Over Speed**: Iterative refinement prioritizes answer quality
3. **Transparency**: Explicit disclaimers when context is insufficient
4. **Safety-First**: Multiple validation gates (jailbreak, relevance, quality)
5. **Graceful Degradation**: Fallback layers (online→offline, unrestricted search)
6. **Loop Prevention**: Hard limits (3 retrieval attempts, 3 answer iterations, 20 total tool calls)

## Troubleshooting

### "ChromaDB data not found"
Run `python prepare_data.py` to download and index documentation.

### "GOOGLE_API_KEY not found"
Create a `.env` file with your Google AI Studio API key.

### "Online mode will not work"
Get a Tavily API key at https://tavily.com and add to `.env`.

### Agent seems stuck or slow
- Enable `--verbose` to see what it's doing
- Check `outputs/*/agent_trace.json` for detailed execution log
- Reduce iteration limits in `agent.py` if needed

### Poor answer quality
- Try online mode for latest information: `--mode online`
- Check if question is clearly phrased
- Review `agent_trace.json` to see what context was retrieved

## Technical Details

### LLM
- **Model**: Google Gemini 2.0 Flash Exp
- **Temperature**: 0.3 (balanced reasoning)
- **Rate Limits**: 15 RPM (free tier)

### Embeddings
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Runs locally** (no API required)

### Vector Store
- **Database**: ChromaDB (persistent client)
- **Similarity**: Cosine similarity
- **Chunks**: ~2000+ documents (1000 chars, 200 overlap)

### Search
- **Provider**: Tavily API
- **Depth**: Advanced
- **Free Tier**: 1000 searches/month
- **Domains**: Official LangChain/LangGraph docs by default

## Contributing

This is a reference implementation demonstrating ReAct agent best practices. Feel free to:
- Adapt for other documentation domains
- Add new tools for enhanced functionality
- Modify system prompt for different behaviors
- Extend with evaluation frameworks

## License

MIT License - feel free to use and modify for your projects.

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - Agent framework
- [Google Gemini](https://ai.google.dev/) - LLM
- [ChromaDB](https://www.trychroma.com/) - Vector store
- [Tavily](https://tavily.com/) - Web search API
- [HuggingFace](https://huggingface.co/) - Embeddings

Inspired by the original LangGraph StateGraph implementation.
