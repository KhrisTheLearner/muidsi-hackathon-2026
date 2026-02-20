#!/usr/bin/env python
"""Quick test script to bypass any run_agent.py issues."""

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import run_agent

# Test query
run_agent("Which Missouri counties are most at risk?")
