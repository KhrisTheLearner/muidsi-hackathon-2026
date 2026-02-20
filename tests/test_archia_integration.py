#!/usr/bin/env python3
"""
End-to-end integration test for Archia + LangGraph deployment.
Tests: API connectivity, agents, skills, local graph, and hybrid workflow.
"""
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://registry.archia.app")
ARCHIA_TOKEN = os.getenv("ARCHIA_TOKEN")

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_archia_connectivity():
    """Test 1: Archia API connectivity"""
    print_section("TEST 1: Archia API Connectivity")

    if not ARCHIA_TOKEN or ARCHIA_TOKEN == "your_archia_token_here":
        print("[SKIP] ARCHIA_TOKEN not configured in .env")
        return False

    try:
        response = requests.get(
            f"{ARCHIA_BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {ARCHIA_TOKEN}"},
            timeout=10
        )

        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"[PASS] Connected to Archia API")
            print(f"       {len(models)} models available")
            return True
        else:
            print(f"[FAIL] API returned {response.status_code}")
            print(f"       {response.text[:200]}")
            return False

    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        return False

def test_archia_agents():
    """Test 2: List deployed agents"""
    print_section("TEST 2: Archia Agents")

    if not ARCHIA_TOKEN or ARCHIA_TOKEN == "your_archia_token_here":
        print("[SKIP] ARCHIA_TOKEN not configured")
        return None

    try:
        response = requests.get(
            f"{ARCHIA_BASE_URL}/v1/agent",
            headers={"Authorization": f"Bearer {ARCHIA_TOKEN}"},
            timeout=10
        )

        if response.status_code == 200:
            agents = response.json().get("agents", [])
            print(f"[PASS] {len(agents)} agents deployed:\n")

            expected_agents = {
                "agriflow-planner": "agriflow-planner",
                "agriflow-system": ["agriflow-system", "AGRIFLOW_SYSTEM"],  # Accept both
                "agriflow-data": "agriflow-data",
                "agriflow-viz": "agriflow-viz",
                "agriflow-logistics": "agriflow-logistics",
                "agriflow-analytics": "agriflow-analytics"
            }

            agent_names = [a.get("name") for a in agents]

            for display_name, expected_name in expected_agents.items():
                # Handle both string and list of acceptable names
                acceptable = [expected_name] if isinstance(expected_name, str) else expected_name

                if any(name in agent_names for name in acceptable):
                    found = next(name for name in acceptable if name in agent_names)
                    print(f"       [OK] {display_name} (as {found})")
                else:
                    print(f"       [MISSING] {display_name}")

            # Check for deprecated ml agent
            if "agriflow-ml" in agent_names:
                print(f"       [WARN] agriflow-ml still deployed (should be disabled)")

            return agents
        else:
            print(f"[FAIL] API returned {response.status_code}")
            return None

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return None

def test_local_skills():
    """Test 3: Check local skills configuration"""
    print_section("TEST 3: Local Skills Configuration")

    skills_file = Path("archia/skills.json")
    if not skills_file.exists():
        print("[FAIL] archia/skills.json not found")
        return None

    with open(skills_file) as f:
        config = json.load(f)

    agents = config.get("agents", [])
    tools = config.get("tools", {})

    print(f"[PASS] Configuration loaded")
    print(f"       {len(agents)} agents defined")

    total_tools = sum(len(tool_list) for tool_list in tools.values())
    print(f"       {total_tools} tools across {len(tools)} categories\n")

    print("Agents:")
    for agent in agents:
        name = agent.get("name")
        tools_bound = agent.get("tools_bound", 0)
        print(f"  {name}: {tools_bound} tools")

    print("\nTool categories:")
    for category, tool_list in tools.items():
        print(f"  [{category}]: {len(tool_list)} tools")

    return config

def test_local_graph():
    """Test 4: Local LangGraph compilation"""
    print_section("TEST 4: Local LangGraph")

    try:
        from src.agent.graph import create_agent
        from src.agent.nodes.tool_executor import ALL_TOOLS

        agent = create_agent()
        print(f"[PASS] Graph compiled successfully")
        print(f"       {len(ALL_TOOLS)} tools loaded")

        # List tool categories
        from src.agent.nodes.tool_executor import (
            DATA_TOOLS, SQL_TOOLS, ML_TOOLS, ANALYTICS_TOOLS, VIZ_TOOLS, ROUTE_TOOLS
        )

        print(f"\nTool distribution:")
        print(f"  [data]      {len(DATA_TOOLS)} tools")
        print(f"  [sql]       {len(SQL_TOOLS)} tools")
        print(f"  [ml]        {len(ML_TOOLS)} tools")
        print(f"  [analytics] {len(ANALYTICS_TOOLS)} tools")
        print(f"  [viz]       {len(VIZ_TOOLS)} tools")
        print(f"  [route]     {len(ROUTE_TOOLS)} tools")

        return True

    except Exception as e:
        print(f"[FAIL] Graph compilation error: {e}")
        return False

def test_archia_agent_call():
    """Test 5: Call Archia agent end-to-end"""
    print_section("TEST 5: Archia Agent Call")

    if not ARCHIA_TOKEN or ARCHIA_TOKEN == "your_archia_token_here":
        print("[SKIP] ARCHIA_TOKEN not configured")
        return None

    try:
        # Test with agriflow-data agent (simple data query)
        payload = {
            "model": "agent:agriflow-data",
            "input": "What data sources do you have access to?",
            "stream": False
        }

        print("Calling agent:agriflow-data...")

        response = requests.post(
            f"{ARCHIA_BASE_URL}/v1/responses",
            headers={
                "Authorization": f"Bearer {ARCHIA_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            status = result.get("status")

            if status == "completed":
                # Extract output text
                output_text = None
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                output_text = content.get("text")
                                break

                if output_text:
                    print(f"[PASS] Agent responded successfully")
                    print(f"\nResponse preview:")
                    print(f"  {output_text[:200]}...")
                    return True
                else:
                    print(f"[WARN] No output text in response")
                    return False
            else:
                print(f"[FAIL] Response status: {status}")
                if "error" in result:
                    print(f"       Error: {result['error']}")
                return False
        else:
            print(f"[FAIL] API returned {response.status_code}")
            print(f"       {response.text[:200]}")
            return False

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def test_local_agent():
    """Test 6: Local agent execution"""
    print_section("TEST 6: Local Agent Execution")

    try:
        from src.agent.graph import create_agent
        from langchain_core.messages import HumanMessage

        agent = create_agent()

        print("Running local agent query...")

        result = agent.invoke({
            "messages": [HumanMessage("List the available tool categories")],
            "plan": [],
            "current_step": 0,
            "tool_results": {},
            "reasoning_trace": [],
            "final_answer": None,
        })

        answer = result.get("final_answer")

        if answer:
            print(f"[PASS] Local agent responded")
            print(f"\nResponse preview:")
            print(f"  {answer[:200]}...")
            return True
        else:
            print(f"[FAIL] No final_answer in result")
            return False

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AgriFlow Integration Test Suite")
    print("  Archia Cloud + LangGraph Verification")
    print("="*60)

    results = {
        "archia_connectivity": test_archia_connectivity(),
        "archia_agents": test_archia_agents(),
        "local_skills": test_local_skills(),
        "local_graph": test_local_graph(),
        "archia_agent_call": test_archia_agent_call(),
        "local_agent": test_local_agent(),
    }

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {len(results)}")

    if failed == 0 and passed > 0:
        print(f"\n[SUCCESS] All active tests passed!")

        if skipped > 0:
            print(f"\nNote: {skipped} tests skipped - configure ARCHIA_TOKEN in .env to enable")
    else:
        print(f"\n[INCOMPLETE] {failed} test(s) failed")

    print("\n" + "="*60 + "\n")

    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
