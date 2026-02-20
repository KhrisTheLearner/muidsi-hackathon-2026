#!/usr/bin/env python3
"""
Comprehensive Archia Cloud deployment script.
Deploys all agents, skills, and verifies the complete pipeline.
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://registry.archia.app")
ARCHIA_TOKEN = os.getenv("ARCHIA_TOKEN")

if not ARCHIA_TOKEN or ARCHIA_TOKEN == "your_archia_token_here":
    print("ERROR: ARCHIA_TOKEN not configured in .env")
    print("Please add your Archia API key from console.archia.app")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {ARCHIA_TOKEN}",
    "Content-Type": "application/json"
}

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_current_agents():
    """Check what agents are currently deployed"""
    print_section("STEP 1: Check Current Agents")

    response = requests.get(f"{ARCHIA_BASE_URL}/v1/agent", headers=HEADERS, timeout=10)

    if response.status_code != 200:
        print(f"[ERROR] Failed to list agents: {response.status_code}")
        print(response.text)
        return None

    agents = response.json().get("agents", [])
    print(f"Found {len(agents)} deployed agents:")
    for agent in agents:
        name = agent.get("name")
        mcps = agent.get("mcps", [])
        print(f"  - {name} ({len(mcps)} MCPs)")

    return {a.get("name"): a for a in agents}

def check_missing_agents():
    """Identify which agents need to be created"""
    print_section("STEP 2: Identify Missing Agents")

    current_agents = check_current_agents()
    if current_agents is None:
        return None

    # Expected agents from skills.json
    with open("archia/skills.json") as f:
        config = json.load(f)

    expected_agents = {a["name"]: a for a in config.get("agents", [])}
    current_names = set(current_agents.keys())
    expected_names = set(expected_agents.keys())

    # Handle AGRIFLOW_SYSTEM vs agriflow-system
    if "AGRIFLOW_SYSTEM" in current_names:
        current_names.add("agriflow-system")

    missing = expected_names - current_names
    extra = current_names - expected_names - {"AGRIFLOW_SYSTEM"}

    print(f"\nExpected: {len(expected_names)} agents")
    print(f"Current:  {len(current_names)} agents")
    print(f"Missing:  {len(missing)} agents")

    if missing:
        print("\nMissing agents:")
        for name in sorted(missing):
            agent = expected_agents[name]
            print(f"  - {name}")
            print(f"    Model: {agent.get('model')}")
            print(f"    Tools: {agent.get('tools_bound', 0)}")

    if extra:
        print("\nExtra/deprecated agents:")
        for name in sorted(extra):
            print(f"  - {name}")

    return {
        "expected": expected_agents,
        "current": current_agents,
        "missing": missing,
        "extra": extra
    }

def provide_manual_instructions(analysis):
    """Provide instructions for manually creating missing agents"""
    print_section("STEP 3: Manual Agent Creation Required")

    if not analysis["missing"]:
        print("[OK] No missing agents - all deployed!")
        return True

    print("Archia Cloud agents must be created via Console UI.")
    print("Follow these steps:\n")

    for name in sorted(analysis["missing"]):
        agent = analysis["expected"][name]
        prompt_file = f"archia/prompts/{name}.md"

        print(f"Agent: {name}")
        print(f"{'-'*70}")
        print(f"1. Go to: https://console.archia.app/agents")
        print(f"2. Click: '+ New Agent'")
        print(f"3. Configuration:")
        print(f"   Name: {name}")
        print(f"   Model: {agent.get('model')}")
        print(f"   Description: {agent.get('description')}")
        print(f"4. System Prompt: Copy from {prompt_file}")

        if Path(prompt_file).exists():
            with open(prompt_file) as f:
                prompt = f.read()
            print(f"\n   Prompt Preview (first 200 chars):")
            print(f"   {prompt[:200]}...")

        print(f"\n5. Save agent\n")

    return False

def deploy_skills():
    """Deploy all skills using the existing setup_skills.py script"""
    print_section("STEP 4: Deploy Skills to Archia")

    print("Running: python archia/setup_skills.py")

    import subprocess
    result = subprocess.run(
        ["python", "archia/setup_skills.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print("\n[OK] Skills deployed successfully")
        return True
    else:
        print(f"\n[ERROR] Skills deployment failed with code {result.returncode}")
        return False

def test_agent_call(agent_name):
    """Test calling a specific agent"""
    payload = {
        "model": f"agent:{agent_name}",
        "input": "What tools and capabilities do you have?",
        "stream": False
    }

    response = requests.post(
        f"{ARCHIA_BASE_URL}/v1/responses",
        headers=HEADERS,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        print(f"[FAIL] Agent call returned {response.status_code}")
        return False

    result = response.json()
    status = result.get("status")

    if status == "completed":
        # Extract output
        for item in result.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        print(f"[OK] Agent responded ({len(text)} chars)")
                        return True

    print(f"[FAIL] Agent status: {status}")
    return False

def test_all_agents(agents):
    """Test each deployed agent"""
    print_section("STEP 5: Test All Agents")

    results = {}
    for agent_name in agents:
        # Skip testing AGRIFLOW_SYSTEM (uppercase) - test lowercase version
        if agent_name == "AGRIFLOW_SYSTEM":
            continue

        print(f"\nTesting: {agent_name}")
        results[agent_name] = test_agent_call(agent_name)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n{'-'*70}")
    print(f"Agent Tests: {passed}/{total} passed")

    return all(results.values())

def test_local_graph():
    """Test local LangGraph compilation"""
    print_section("STEP 6: Test Local LangGraph")

    try:
        from src.agent.graph import create_agent
        from src.agent.nodes.tool_executor import ALL_TOOLS

        agent = create_agent()
        print(f"[OK] Graph compiled successfully")
        print(f"[OK] {len(ALL_TOOLS)} tools loaded")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def verify_pipeline():
    """Verify the complete pipeline end-to-end"""
    print_section("STEP 7: Verify Complete Pipeline")

    checks = {
        "Archia API": False,
        "Local LangGraph": False,
        "Skills Config": False,
    }

    # Check Archia API
    try:
        response = requests.get(f"{ARCHIA_BASE_URL}/v1/models", headers=HEADERS, timeout=10)
        checks["Archia API"] = response.status_code == 200
    except:
        pass

    # Check local graph
    try:
        from src.agent.graph import create_agent
        create_agent()
        checks["Local LangGraph"] = True
    except:
        pass

    # Check skills config
    checks["Skills Config"] = Path("archia/skills.json").exists()

    print("Pipeline Status:")
    for check, status in checks.items():
        print(f"  {check}: {'[OK]' if status else '[FAIL]'}")

    return all(checks.values())

def show_next_steps(analysis):
    """Show what needs to be done next"""
    print_section("NEXT STEPS")

    if analysis["missing"]:
        print("[ACTION REQUIRED] Create missing agents in Archia Console:")
        for name in sorted(analysis["missing"]):
            print(f"  - {name}")
        print("\nSee instructions above for details.")
    else:
        print("[OK] All agents deployed!")

    print("\n[ACTION REQUIRED] Load database:")
    print("  1. Download USDA Food Environment Atlas")
    print("  2. Download USDA Food Access Research Atlas")
    print("  3. Load into data/agriflow.db")
    print("  4. See docs/SETUP.md for details")

    print("\n[OPTIONAL] Test the system:")
    print("  python test_archia_integration.py")
    print("  python run_agent.py")
    print("  uvicorn src.api.main:app --port 8000")

def main():
    print("\n" + "="*70)
    print("  AgriFlow Archia Cloud Deployment")
    print("  Automated agent and skill configuration")
    print("="*70)

    # Step 1-2: Check current state
    analysis = check_missing_agents()
    if analysis is None:
        print("\n[ERROR] Failed to check agents")
        return 1

    # Step 3: Provide manual instructions if needed
    all_deployed = provide_manual_instructions(analysis)

    # Step 4: Deploy skills (whether or not all agents exist)
    print("\nContinuing with skill deployment...")
    skills_ok = deploy_skills()

    # Step 5: Test agents that exist
    current_agents = list(analysis["current"].keys())
    if current_agents:
        agents_ok = test_all_agents(current_agents)
    else:
        print("\n[SKIP] No agents to test")
        agents_ok = False

    # Step 6: Test local graph
    graph_ok = test_local_graph()

    # Step 7: Verify pipeline
    pipeline_ok = verify_pipeline()

    # Show next steps
    show_next_steps(analysis)

    # Summary
    print_section("DEPLOYMENT SUMMARY")

    print(f"Agents Deployed:    {len(analysis['current'])}/{len(analysis['expected'])}")
    print(f"Skills Deployed:    {'Yes' if skills_ok else 'Failed'}")
    print(f"Agent Tests:        {'Passed' if agents_ok else 'Some Failed'}")
    print(f"Local LangGraph:    {'OK' if graph_ok else 'Failed'}")
    print(f"Pipeline Verified:  {'OK' if pipeline_ok else 'Issues'}")

    if not analysis["missing"] and skills_ok and graph_ok:
        print("\n[SUCCESS] Deployment complete! System ready for database loading.")
        return 0
    else:
        print("\n[INCOMPLETE] Manual steps required (see above).")
        return 1

if __name__ == "__main__":
    sys.exit(main())
