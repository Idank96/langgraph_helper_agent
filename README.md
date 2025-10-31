# LangGraph Helper Agent

An AI agent that helps developers work with LangGraph and LangChain. Supports both offline (local documentation) and online (web search) modes.

## Architecture

```
User Question → Retrieve Node → Respond Node → Answer
                     ↓
            [Offline: ChromaDB]
            [Online: Tavily API]
```

**State Graph Components:**
- `AgentState`: Tracks question, mode, context, and answer
- `retrieve_node`: Gets context from ChromaDB (offline) or Tavily (online)
- `respond_node`: Uses Gemini to generate answer from context

## Project Structure

```
langgraph_helper_agent/
├── src/
│   ├── agent.py       # LangGraph state graph
│   ├── state.py       # AgentState definition
│   ├── offline.py     # ChromaDB retrieval
│   └── online.py      # Tavily search
├── data/
│   ├── raw/           # Downloaded llms.txt files
│   └── vectorstore/   # ChromaDB storage
├── prepare_data.py    # Data preparation script
├── main.py            # CLI entry point
└── .env               # API keys (not in git)
```

## Setup

### 1. Install Dependencies

```bash
#### Python 3.12 / 3.11 recommended

```bash
conda create -n helper_agent_env python=3.12
conda activate helper_agent_env
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file in the root directory and add your keys:
- `GOOGLE_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com)

### 3. Prepare Offline Data

Run this once to download documentation and build the vector store:

```bash
python prepare_data.py
```

This downloads:
- LangGraph docs: `https://langchain-ai.github.io/langgraph/llms-full.txt`
- LangChain docs: `https://python.langchain.com/llms.txt`

And creates a ChromaDB vector store in `data/vectorstore/`.

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
- **How it works**: Uses Tavily search API to find current information from the web
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
```

## Technical Details

- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (local, no API needed)
- **Vector Store**: ChromaDB (file-based, no server needed)
- **Search**: Tavily API (1000 free searches/month)
- **Framework**: LangGraph for agent orchestration

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
