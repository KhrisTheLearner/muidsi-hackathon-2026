#!/usr/bin/env python3
"""
Attempt to programmatically create the agriflow-analytics agent via Archia Cloud API.
If not possible, provides detailed console instructions.
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
    print("ERROR: ARCHIA_TOKEN not configured")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {ARCHIA_TOKEN}",
    "Content-Type": "application/json"
}

def read_prompt():
    """Read the analytics agent prompt"""
    prompt_file = Path("archia/prompts/agriflow-analytics.md")
    if not prompt_file.exists():
        print(f"ERROR: {prompt_file} not found")
        return None

    return prompt_file.read_text()

def try_api_creation():
    """Try to create agent via API"""
    print("Attempting to create agriflow-analytics via API...")

    prompt = read_prompt()
    if not prompt:
        return False

    # Try different API endpoints that might work
    endpoints_to_try = [
        f"{ARCHIA_BASE_URL}/v1/agent",
        f"{ARCHIA_BASE_URL}/v1/agent/config",
        f"{ARCHIA_BASE_URL}/v1/agents",
    ]

    payload = {
        "name": "agriflow-analytics",
        "model_name": "priv-claude-sonnet-4-5-20250929",
        "enabled": True,
        "description": "Data analytics supervisor: XGBoost/Random Forest training, SHAP explainability, CCC validation, anomaly detection, web risk search, and full analytics pipelines",
        "system_prompt": prompt
    }

    for endpoint in endpoints_to_try:
        print(f"\nTrying: POST {endpoint}")
        try:
            response = requests.post(endpoint, headers=HEADERS, json=payload, timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

            if response.status_code in [200, 201]:
                print("[SUCCESS] Agent created via API!")
                return True
        except Exception as e:
            print(f"Error: {e}")

    return False

def console_instructions():
    """Provide detailed console instructions"""
    print("\n" + "="*70)
    print("  Manual Agent Creation Required")
    print("="*70)

    print("\nArchia Cloud requires agents to be created via Console UI.")
    print("Follow these steps carefully:\n")

    print("STEP 1: Open Archia Console")
    print("-" * 70)
    print("  1. Go to: https://console.archia.app")
    print("  2. Log in to your account")
    print("  3. Verify you're in the 'MUIDSI Hackathon 2026' organization")
    print("     (check top-left dropdown)")

    print("\nSTEP 2: Navigate to Agents")
    print("-" * 70)
    print("  1. Click 'Agents' in the left sidebar")
    print("  2. Click '+ New Agent' or 'Create Agent' button")

    print("\nSTEP 3: Configure Agent")
    print("-" * 70)
    print("  Name: agriflow-analytics")
    print("  Model: priv-claude-sonnet-4-5-20250929")
    print("  Description: (copy below)")
    print()
    print("    Data analytics supervisor: XGBoost/Random Forest training,")
    print("    SHAP explainability, CCC validation, anomaly detection,")
    print("    web risk search, and full analytics pipelines")

    print("\nSTEP 4: System Prompt")
    print("-" * 70)
    print("  Copy the ENTIRE content from:")
    print("    archia/prompts/agriflow-analytics.md")
    print()

    prompt = read_prompt()
    if prompt:
        print("  Prompt Preview (first 300 characters):")
        print("  " + "-" * 68)
        for line in prompt[:300].split("\n"):
            print(f"  {line}")
        print("  " + "-" * 68)
        print(f"  ... (total {len(prompt)} characters)")

    print("\nSTEP 5: Save")
    print("-" * 70)
    print("  1. Review the configuration")
    print("  2. Click 'Save' or 'Create Agent'")
    print("  3. Wait for confirmation")

    print("\nSTEP 6: Verify")
    print("-" * 70)
    print("  Run: python test_archia_integration.py")
    print("  Should now show 7/7 agents deployed")

    print("\n" + "="*70)
    print("  After Creating Agent")
    print("="*70)
    print("\nRun skill deployment:")
    print("  python archia/setup_skills.py")
    print("\nTest the agent:")
    print("  curl -X POST 'https://registry.archia.app/v1/responses' \\")
    print("    -H 'Authorization: Bearer ${ARCHIA_TOKEN}' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"model\": \"agent:agriflow-analytics\", \"input\": \"What tools do you have?\", \"stream\": false}'")

def check_if_exists():
    """Check if agent already exists"""
    response = requests.get(f"{ARCHIA_BASE_URL}/v1/agent", headers=HEADERS, timeout=10)

    if response.status_code != 200:
        return False

    agents = response.json().get("agents", [])
    return any(a.get("name") == "agriflow-analytics" for a in agents)

def main():
    print("\n" + "="*70)
    print("  AgriFlow Analytics Agent Setup")
    print("="*70)

    # Check if already exists
    if check_if_exists():
        print("\n[OK] agriflow-analytics agent already exists!")
        print("\nNext: Deploy skills")
        print("  python archia/setup_skills.py")
        return 0

    # Try API creation
    if try_api_creation():
        print("\n[SUCCESS] Agent created programmatically!")
        return 0

    # Fall back to manual instructions
    print("\n[INFO] API creation not available - manual creation required")
    console_instructions()

    return 1

if __name__ == "__main__":
    sys.exit(main())
