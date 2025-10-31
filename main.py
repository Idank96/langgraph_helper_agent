import os
import argparse
from dotenv import load_dotenv
from src.agent import run_agent
from prepare_data import download_docs, build_vectorstore

load_dotenv()


def main():

    # Debug section - comment out when not debugging
    debug = False
    if debug:
        debug_args = {
            "question": "How do I add persistence to a LangGraph agent?",
            "mode": "online"
        }
        print(f"Debug Mode: Using preset arguments\n")
        print(f"Mode: {debug_args['mode']}\n")
        print(run_agent(debug_args["question"], debug_args["mode"]))
        return



    parser = argparse.ArgumentParser(description="LangGraph Helper Agent")
    parser.add_argument("question", nargs="?", help="Your question about LangGraph/LangChain")
    parser.add_argument("--mode", choices=["offline", "online"], help="offline (default) or online")
    parser.add_argument("--update_data", action="store_true", help="Update offline documentation data")
    args = parser.parse_args()

    mode = args.mode or os.getenv("AGENT_MODE", "offline")

    if args.update_data:
        download_docs()
        build_vectorstore(force_rebuild=True)
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
    print(run_agent(args.question, mode))


if __name__ == "__main__":
    main()




