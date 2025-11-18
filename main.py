#!/usr/bin/env python3
"""
LangGraph/LangChain Helper - ReAct Agent CLI

A command-line interface for an intelligent documentation assistant that helps
developers with LangGraph and LangChain questions using ReAct agent architecture.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from agent import run_agent


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        True if environment is valid, False otherwise
    """
    # Load environment variables
    load_dotenv()

    # Check for required API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your Google AI Studio API key:")
        print("  GOOGLE_API_KEY=your_key_here")
        print("\nGet your API key at: https://makersuite.google.com/app/apikey")
        return False

    # Check for optional Tavily key if online mode might be used
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("⚠️  Warning: TAVILY_API_KEY not found")
        print("Online mode will not work without this key.")
        print("Get your API key at: https://tavily.com")
        print()

    return True


def create_output_directory() -> Path:
    """
    Create a timestamped output directory for saving results.

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_outputs(output_dir: Path, question: str, answer: str,
                intermediate_steps: list) -> None:
    """
    Save agent outputs to files for later review.

    Args:
        output_dir: Directory to save outputs
        question: Original user question
        answer: Final answer from agent
        intermediate_steps: List of (action, observation) tuples from agent execution
    """
    # Save answer
    answer_file = output_dir / "answer.md"
    with open(answer_file, 'w', encoding='utf-8') as f:
        f.write(f"# Question\n\n{question}\n\n")
        f.write(f"# Answer\n\n{answer}\n")

    print(f"\n📄 Answer saved to: {answer_file}")

    # Save agent trace (intermediate steps)
    if intermediate_steps:
        trace_file = output_dir / "agent_trace.json"
        trace_data = {
            "question": question,
            "steps": []
        }

        for i, (action, observation) in enumerate(intermediate_steps, 1):
            step_data = {
                "step": i,
                "tool": action.tool if hasattr(action, 'tool') else str(action),
                "input": action.tool_input if hasattr(action, 'tool_input') else str(action),
                "observation": str(observation)[:500]  # Truncate long observations
            }
            trace_data["steps"].append(step_data)

        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2)

        print(f"🔍 Agent trace saved to: {trace_file}")

    # Save full conversation log
    chat_file = output_dir / "chat.md"
    with open(chat_file, 'w', encoding='utf-8') as f:
        f.write(f"# LangGraph Helper - Conversation Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Question\n\n{question}\n\n")

        if intermediate_steps:
            f.write(f"## Agent Reasoning\n\n")
            for i, (action, observation) in enumerate(intermediate_steps, 1):
                tool_name = action.tool if hasattr(action, 'tool') else str(action)
                tool_input = action.tool_input if hasattr(action, 'tool_input') else str(action)
                f.write(f"### Step {i}: {tool_name}\n\n")
                f.write(f"**Input:** {tool_input}\n\n")
                f.write(f"**Output:** {str(observation)[:1000]}...\n\n")

        f.write(f"## Final Answer\n\n{answer}\n")

    print(f"💬 Full chat log saved to: {chat_file}")


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print(" LangGraph/LangChain Helper - ReAct Agent")
    print("=" * 70)
    print()


def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Intelligent assistant for LangGraph and LangChain questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question in offline mode (using local vector store)
  python main.py "How do I use StateGraph with checkpointing?"

  # Use online mode for web search
  python main.py --mode online "What's new in LangGraph 1.0?"

  # Enable verbose output to see agent reasoning
  python main.py --verbose "Difference between MemorySaver and SqliteSaver?"

  # Combine options
  python main.py --mode online --verbose "How to create a ReAct agent?"
        """
    )

    parser.add_argument(
        "question",
        type=str,
        help="Your question about LangGraph or LangChain"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="Retrieval mode: offline (local ChromaDB) or online (Tavily web search)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print agent reasoning steps and tool calls"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save outputs to files"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Check if data is prepared for offline mode
    if args.mode == "offline":
        chroma_path = Path("data/vectorstore")
        if not chroma_path.exists():
            print("❌ Error: ChromaDB data not found!")
            print("Please run prepare_data.py first to download and index documentation:")
            print("  python prepare_data.py")
            sys.exit(1)

    # Display mode info
    mode_emoji = "💾" if args.mode == "offline" else "🌐"
    print(f"{mode_emoji} Mode: {args.mode.upper()}")
    if args.verbose:
        print("🔍 Verbose: Enabled")
    print(f"❓ Question: {args.question}")
    print()
    print("-" * 70)
    print("🤔 Agent is thinking...")
    print("-" * 70)
    print()

    # Run agent
    result = run_agent(
        question=args.question,
        mode=args.mode,
        verbose=args.verbose
    )

    # Check for errors
    if result["error"]:
        print(f"\n❌ Error occurred: {result['error']}")
        sys.exit(1)

    # Display answer
    print("\n" + "=" * 70)
    print(" ANSWER")
    print("=" * 70)
    print()
    print(result["answer"])
    print()

    # Save outputs unless --no-save flag is set
    if not args.no_save:
        output_dir = create_output_directory()
        save_outputs(
            output_dir=output_dir,
            question=args.question,
            answer=result["answer"],
            intermediate_steps=result.get("intermediate_steps", [])
        )

    # Print statistics
    num_steps = len(result.get("intermediate_steps", []))
    print(f"\n📊 Agent used {num_steps} tool calls to answer this question")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
