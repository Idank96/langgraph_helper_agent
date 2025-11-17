#!/usr/bin/env python3
"""Test script for keyword extraction functionality."""

import os
from dotenv import load_dotenv
from src.tools import extract_keywords

load_dotenv()

def test_keyword_extraction():
    """Test the keyword extraction on various queries."""

    test_cases = [
        "What's the difference between StateGraph and MessageGraph?",
        "How do I use checkpointing with SqliteSaver?",
        "What is StateGraph?",
        "Explain how to implement persistence",
        "Compare LangGraph and LangChain agents",
        "How do nodes work in StateGraph?",
    ]

    print("=" * 60)
    print("KEYWORD EXTRACTION TEST")
    print("=" * 60)

    for i, question in enumerate(test_cases, 1):
        print(f"\n{i}. Question: {question}")
        keywords = extract_keywords(question)

        if keywords:
            print(f"   Keywords: {', '.join(keywords)}")
            print(f"   → Will perform {len(keywords) + 1} searches")
        else:
            print(f"   Keywords: (none)")
            print(f"   → Will perform 1 search (original question only)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_keyword_extraction()
