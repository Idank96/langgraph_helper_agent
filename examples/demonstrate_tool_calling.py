#!/usr/bin/env python3
"""
Demonstrate Tool-Calling Agent Architecture

This script showcases the hybrid architecture with three scenarios:
1. Simple question - No tools needed (assessment skips to direct answer)
2. Complex question - Multiple tool calls with autonomous selection
3. Error scenario - Graceful fallback from online to offline mode

Run with:
    python examples/demonstrate_tool_calling.py           # Run all scenarios
    python examples/demonstrate_tool_calling.py --interactive  # Interactive mode
    python examples/demonstrate_tool_calling.py --verbose      # With detailed logging
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.agent import run_hybrid_agent

load_dotenv()


SCENARIOS = {
    "simple": {
        "title": "Simple Question (No Tools Needed)",
        "question": "What does RAG stand for in the context of LLMs?",
        "expected_behavior": [
            "Assessment node detects this is a simple definitional question",
            "Skips tool-calling agent entirely",
            "Uses direct_answer_node for efficient response",
            "No tools called (tools_called list should be empty)"
        ],
        "mode": "offline"
    },
    "complex": {
        "title": "Complex Question (Multiple Tool Calls)",
        "question": "How do I implement custom checkpointing with PostgreSQL in LangGraph? Please include code examples.",
        "expected_behavior": [
            "Assessment node determines tools are needed",
            "Agent autonomously selects search_documentation",
            "Agent may validate results with validate_retrieved_information",
            "If insufficient, agent refines search with refine_search_strategy",
            "Agent searches again with improved query",
            "Final answer evaluated with check_answer_quality",
            "Multiple tools called in sequence based on agent's reasoning"
        ],
        "mode": "offline"
    },
    "error": {
        "title": "Error Scenario (Graceful Fallback)",
        "question": "What are the best practices for LangGraph agent debugging?",
        "expected_behavior": [
            "Attempts online search (will fail if TAVILY_API_KEY not set)",
            "search_documentation tool auto-falls back to offline mode",
            "Agent continues normally with offline documentation",
            "Error handled gracefully at tool level",
            "Final answer still provided despite initial failure"
        ],
        "mode": "online"  # Intentionally use online to trigger fallback
    }
}


def print_section_header(title, char="="):
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_trace_analysis(output_dir, scenario_name):
    """Analyze and print the detailed agent trace."""
    trace_path = Path(output_dir) / "detailed_agent_trace.json"

    if not trace_path.exists():
        print("⚠️  Detailed trace not found")
        return

    with open(trace_path) as f:
        trace = json.load(f)

    print_section_header("Tool-Calling Analysis", "-")

    assessment = trace.get("workflow", {}).get("assessment", {})
    skip_retrieval = assessment.get("skip_retrieval", False)

    print(f"📊 Assessment Decision:")
    print(f"   Skip retrieval: {skip_retrieval}")
    if skip_retrieval:
        print(f"   → Question answered directly without tools\n")
    else:
        print(f"   → Question requires documentation search\n")

    agent_exec = trace.get("workflow", {}).get("agent_execution", {})
    iterations = agent_exec.get("iterations", 0)
    tools_summary = agent_exec.get("tools_summary", {})
    tools_called = agent_exec.get("tools_called", [])

    print(f"🤖 Agent Execution:")
    print(f"   Iterations: {iterations}")
    print(f"   Tools used: {len(tools_called)}\n")

    if tools_called:
        print(f"📞 Tool Call Sequence:")
        for i, tool_call in enumerate(tools_called, 1):
            tool_name = tool_call.get("tool", "unknown")
            iteration = tool_call.get("iteration", 0)
            print(f"   {i}. [{tool_name}] (iteration {iteration})")

            args = tool_call.get("args", {})
            if "query" in args:
                print(f"      Query: {args['query'][:60]}...")
            elif "question" in args:
                print(f"      Question: {args['question'][:60]}...")

        print(f"\n📈 Tools Summary:")
        for tool_name, count in tools_summary.items():
            print(f"   {tool_name}: {count}x")
    else:
        print("   ℹ️  No tools were called (direct answer)")

    print()


def run_scenario(scenario_name, scenario, verbose=False):
    """Run a single scenario and display results."""
    print_section_header(f"Scenario: {scenario['title']}")

    print("📝 Question:")
    print(f"   {scenario['question']}\n")

    print("🎯 Expected Behavior:")
    for behavior in scenario['expected_behavior']:
        print(f"   • {behavior}")
    print()

    if verbose:
        os.environ["VERBOSE"] = "1"

    print("🚀 Running hybrid tool-calling agent...\n")

    try:
        answer, _ = run_hybrid_agent(
            question=scenario['question'],
            mode=scenario['mode'],
            evaluate=False,
            max_iterations=5
        )

        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            output_dirs = sorted(outputs_dir.glob("*"), key=lambda x: x.stat().st_mtime)
            if output_dirs:
                latest_output = output_dirs[-1]

                print_trace_analysis(latest_output, scenario_name)

                print_section_header("Answer Preview", "-")
                answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
                print(answer_preview)
                print(f"\n📁 Full output saved to: {latest_output}\n")

    except Exception as e:
        print(f"❌ Error running scenario: {e}\n")
        import traceback
        traceback.print_exc()

    finally:
        if verbose:
            os.environ.pop("VERBOSE", None)


def run_interactive_mode():
    """Interactive mode - prompt for questions and show live tool-calling."""
    print_section_header("Interactive Tool-Calling Demonstration")

    print("This mode lets you ask questions and see the agent's tool-calling decisions in real-time.\n")

    os.environ["VERBOSE"] = "1"  # Always verbose in interactive mode

    while True:
        print("\n" + "-" * 80)
        question = input("\n💬 Your question (or 'quit' to exit): ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break

        if not question:
            continue

        mode = input("🔍 Search mode [offline/online] (default: offline): ").strip().lower()
        mode = mode if mode in ['online', 'offline'] else 'offline'

        print()
        try:
            answer, _ = run_hybrid_agent(
                question=question,
                mode=mode,
                evaluate=False,
                max_iterations=5
            )

            print(f"\n{'='*80}")
            print("ANSWER")
            print(f"{'='*80}\n")
            print(answer)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    os.environ.pop("VERBOSE", None)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate hybrid tool-calling agent architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/demonstrate_tool_calling.py                    # Run all scenarios
  python examples/demonstrate_tool_calling.py --verbose          # With detailed logging
  python examples/demonstrate_tool_calling.py --interactive      # Interactive mode
  python examples/demonstrate_tool_calling.py --scenario simple  # Run one scenario
        """
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - ask questions and see live tool-calling"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show agent reasoning and tool calls)"
    )

    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Run a specific scenario only"
    )

    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in environment")
        print("Please set it in your .env file\n")
        sys.exit(1)

    if args.interactive:
        run_interactive_mode()
        return

    print_section_header("Hybrid Tool-Calling Agent Demonstration")

    print("This demonstration shows three key scenarios:\n")
    print("1. ✅ Simple questions - No unnecessary tool usage")
    print("2. 🔧 Complex questions - Autonomous multi-tool workflows")
    print("3. 🛡️  Error scenarios - Graceful fallback handling\n")

    if args.verbose:
        print("📊 Verbose mode enabled - detailed agent decisions will be shown\n")

    scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS

    for name, scenario in scenarios_to_run.items():
        run_scenario(name, scenario, verbose=args.verbose)

    print_section_header("Demonstration Complete")

    print("Key Takeaways:\n")
    print("✅ The agent genuinely decides which tools to call based on reasoning")
    print("✅ Safety rails prevent infinite loops (max iterations enforced)")
    print("✅ Quality gates ensure answers meet 7/10 threshold")
    print("✅ Error handling provides graceful fallbacks at multiple levels")
    print("✅ Detailed traces show complete decision-making process\n")

    print("📖 For architecture details, see ARCHITECTURE.md\n")


if __name__ == "__main__":
    main()
