# Bug Fix: Refinement Loops Error

## Problem

When running with `--refinement` flag, the system crashed with:

```
AttributeError: 'NoneType' object has no attribute 'get'
```

at `src/agent.py:203` in the `should_refine_search()` function.

## Root Cause

The `should_refine_search()` routing function was trying to access `state["last_search_validation"]`, which was only set inside the `search_refinement_node`. However, when the flow went directly from `agent_node` → `should_refine_search()`, the validation hadn't been performed yet, causing `last_search_validation` to be `None`.

**Flow issue:**
```
agent_node → should_refine_search() [tries to access validation]
                    ↓
              search_refinement_node [sets validation]  ← Too late!
```

## Solution

### 1. Fixed Validation Flow

Moved the validation logic into the routing function itself:

**Before:**
```python
def should_refine_search(state: AgentState):
    validation = state.get("last_search_validation", {})  # Could be None!
    if not validation.get("is_sufficient", True):  # ERROR: NoneType
        return "refine_search"
```

**After:**
```python
def should_refine_search(state: AgentState):
    # Perform validation in the routing function
    validation = validate_retrieved_information(
        question=state["question"],
        context=state.get("context", "")
    )
    state["last_search_validation"] = validation

    if not validation.get("is_sufficient", True):
        return "refine_search"
```

### 2. Added Rate Limiting

The refinement loops make many LLM calls in quick succession:
- Agent invocation (multiple calls for tool selection)
- Validation calls
- Quality check calls
- Refinement calls
- Repeated agent calls

This quickly exceeds Gemini's free tier limit (15 RPM).

**Solution:**
1. Created `src/rate_limiter.py` with automatic rate limiting
2. Added rate limiting to all LLM calls in `src/tools.py`
3. Set conservative default limits:
   - `MAX_SEARCH_REFINEMENTS = 1` (was 3)
   - `MAX_QUALITY_REFINEMENTS = 2` (was 5)
   - `TARGET_QUALITY_SCORE = 8` (maintained original target)

### 3. Made Limits Configurable

Users can now configure refinement behavior via environment variables:

```bash
# .env file
MAX_SEARCH_REFINEMENTS=1    # How many search refinements
MAX_QUALITY_REFINEMENTS=2   # How many quality refinements
TARGET_QUALITY_SCORE=7      # Target quality 1-10
```

## Files Changed

1. **src/agent.py**
   - Fixed `should_refine_search()` to perform validation before routing
   - Updated `search_refinement_node()` to reuse validation result
   - Made refinement limits configurable via env vars
   - Added rate limiter import and usage

2. **src/rate_limiter.py** (NEW)
   - Implements automatic rate limiting
   - Tracks API call timestamps
   - Automatically waits when approaching rate limits
   - Global `gemini_rate_limiter` instance (10 RPM limit)

3. **src/tools.py**
   - Added rate limiting before all `llm.invoke()` calls
   - Imports and uses `gemini_rate_limiter`

4. **README.md**
   - Added rate limiting documentation
   - Documented configuration environment variables
   - Added notes about free tier limits

## Testing

The fix can be tested with:

```bash
# Test with refinement loops (will use rate limiting)
python main.py "What's the difference between StateGraph and MessageGraph?" \
    --verbose --mode online --refinement

# Test with custom limits (more aggressive)
MAX_SEARCH_REFINEMENTS=3 MAX_QUALITY_REFINEMENTS=5 TARGET_QUALITY_SCORE=8 \
    python main.py "How do I add persistence?" --refinement --verbose
```

## Result

- ✅ No more `NoneType` errors
- ✅ Automatic rate limiting prevents quota errors
- ✅ Configurable limits for different use cases
- ✅ Better documentation for users
