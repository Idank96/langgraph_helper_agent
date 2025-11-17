import os
import argparse
from dotenv import load_dotenv
from src.agent import run_agent
from prepare_data import download_docs, build_vectorstore

load_dotenv()


def main():
    debug = False
    if debug:
        debug_args = {
            "question": "How do I add persistence to a LangGraph agent?",
            "mode": "online",
            "evaluate": True,
            "verbose": True
        }
        if debug_args.get("verbose"):
            os.environ["AGENT_VERBOSE"] = "true"
            print("Verbose mode enabled - agent decisions will be logged\n")

        print(f"Debug Mode: Using preset arguments\n")
        print(f"Mode: {debug_args['mode']}\n")
        answer, scores = run_agent(debug_args["question"], debug_args["mode"], evaluate=debug_args["evaluate"])
        print(answer)

        if scores:
            print(f"\n{'='*60}\nLLM-AS-A-JUDGE EVALUATION SCORES\n{'='*60}")
            for metric, score in scores.items():
                print(f"  {metric.replace('_', ' ').title():20s}: {score:.2f}")
            print(f"{'='*60}\n")
        return

    parser = argparse.ArgumentParser(description="LangGraph Helper Agent - Agentic System")
    parser.add_argument("question", nargs="?", help="Your question about LangGraph/LangChain")
    parser.add_argument("--mode", choices=["offline", "online"], help="offline (default) or online")
    parser.add_argument("--update_data", action="store_true", help="Update offline documentation data")
    parser.add_argument("--force_rebuild", action="store_true", help="Force complete rebuild of vector store (use with --update_data)")
    parser.add_argument("--evaluate", action="store_true", help="Run LLM-as-a-Judge evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of agent decisions")
    args = parser.parse_args()

    mode = args.mode or os.getenv("AGENT_MODE", "offline")

    if args.verbose:
        os.environ["AGENT_VERBOSE"] = "true"
        print("Verbose mode enabled - agent decisions will be logged\n")

    if args.update_data:
        download_docs()
        build_vectorstore(force_rebuild=args.force_rebuild)
        print("Documentation data updated successfully!\n")

    if not args.question:
        parser.error("the following arguments are required: question")
    print(f'\nQuestion: "{args.question}"\n')

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found")
        exit(1)

    if mode == "online" and not os.getenv("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY required for online mode")
        exit(1)

    print(f"Mode: {mode}\n")
    print(f"Evaluate: {args.evaluate}\n")
    answer, scores = run_agent(args.question, mode, evaluate=args.evaluate)
    print("\nFinal Answer:\n")
    print(answer)

    if scores:
        print(f"\n{'='*60}\nLLM-AS-A-JUDGE EVALUATION SCORES\n{'='*60}")
        for metric, score in scores.items():
            print(f"  {metric.replace('_', ' ').title():20s}: {score:.2f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()




