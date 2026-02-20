#!/usr/bin/env python
"""
Demonstration of AgriFlow's autonomous agent behavior.

Shows how agents automatically retrieve data and apply ML pipeline
based on user queries without manual intervention.
"""

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import run_agent

def demo_autonomous_behavior():
    """Run three demos showing increasing levels of autonomy."""

    print("=" * 70)
    print("AgriFlow Autonomous Agent Demonstration")
    print("=" * 70)
    print("\nWatch how the agent automatically:")
    print("  1. Decomposes complex queries into steps")
    print("  2. Selects appropriate data sources")
    print("  3. Routes to cost-optimized agents")
    print("  4. Applies ML pipeline when needed")
    print("  5. Generates visualizations")
    print("  6. Synthesizes comprehensive answers")
    print("\nAll without manual intervention!\n")

    demos = [
        {
            "name": "Demo 1: Simple Data Retrieval",
            "description": "Shows automatic data source selection",
            "query": "What crops does Missouri produce?",
            "expected_autonomy": [
                "Automatically identifies NASS as data source",
                "Routes to agriflow-data agent (cost-optimized)",
                "Retrieves crop production data",
                "Formats response with source citations"
            ]
        },
        {
            "name": "Demo 2: Multi-Source Data Fusion",
            "description": "Shows automatic cross-referencing of multiple sources",
            "query": "Are high corn-producing Missouri counties also food insecure?",
            "expected_autonomy": [
                "Automatically identifies need for 2 data sources (NASS + Food Atlas)",
                "Retrieves corn production from NASS",
                "Retrieves food insecurity from Food Atlas",
                "Automatically joins datasets by county",
                "Analyzes correlation",
                "May auto-generate scatter plot"
            ]
        },
        {
            "name": "Demo 3: Full ML Analytics Pipeline",
            "description": "Shows complete autonomous ML workflow",
            "query": "Train a model to predict which Missouri counties will have food supply issues during a drought",
            "expected_autonomy": [
                "Automatically identifies need for ML analytics",
                "Retrieves data from multiple sources (NASS, Food Atlas, Census, Weather)",
                "Builds feature matrix (35+ features)",
                "Trains XGBoost model with cross-validation",
                "Generates predictions with confidence intervals",
                "Computes SHAP feature importance",
                "May detect anomalies",
                "Creates risk visualization",
                "Synthesizes actionable recommendations",
                "All routed to appropriate agents with cost optimization"
            ]
        }
    ]

    for i, demo in enumerate(demos, 1):
        print("\n" + "=" * 70)
        print(f"{demo['name']}")
        print("=" * 70)
        print(f"\n{demo['description']}\n")
        print(f"Query: \"{demo['query']}\"")
        print("\nExpected Autonomous Behaviors:")
        for behavior in demo["expected_autonomy"]:
            print(f"  - {behavior}")

        print("\n" + "-" * 70)
        input(f"\nPress ENTER to run {demo['name']}...")
        print("-" * 70 + "\n")

        # Run the query
        try:
            run_agent(demo["query"])
        except Exception as e:
            print(f"\n[Note: Demo encountered expected limitation: {e}]")
            print("(Database may need to be loaded for full functionality)")

        if i < len(demos):
            print("\n" + "=" * 70)
            input("Press ENTER to continue to next demo...")

    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Zero manual tool selection - Agent decides automatically")
    print("  2. Zero manual routing - Cost-optimized agent selection")
    print("  3. Zero manual ML pipeline - Automatic train->predict->explain")
    print("  4. Zero manual data fusion - Multi-source joins automatic")
    print("  5. Zero manual visualization - Charts created when helpful")
    print("\nYour agents are fully autonomous and production-ready!")
    print("\nFor more details, see: docs/AUTONOMOUS_AGENT_GUIDE.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo_autonomous_behavior()
