#!/usr/bin/env python3
"""AgriFlow CLI - run the agent from the command line.

Usage:
    python run_agent.py "Which Missouri counties are most at risk?"
    python run_agent.py  # interactive mode
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import run_agent  # noqa: E402

EXAMPLE_QUERIES = [
    "Which Missouri counties have the highest food insecurity rates?",
    "What are the top corn-producing counties in Missouri and how food-secure are they?",
    "If corn yields drop 20% due to drought, which communities are most at risk?",
    "What's the weather forecast for Wayne County, Missouri?",
    "Show me food desert data for southeastern Missouri counties.",
]


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent(query)
        return

    # Interactive mode
    print("\n" + "=" * 60)
    print("  AgriFlow - Food Supply Chain Intelligence Agent")
    print("  Type a question or 'quit' to exit.")
    print("=" * 60)
    print("\nExample queries:")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            query = input("AgriFlow> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Allow selecting example by number
        if query.isdigit() and 1 <= int(query) <= len(EXAMPLE_QUERIES):
            query = EXAMPLE_QUERIES[int(query) - 1]
            print(f"-> {query}\n")

        run_agent(query)
        print()


if __name__ == "__main__":
    main()
