#!/usr/bin/env python3
"""Test the new agriflow-analytics agent"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://registry.archia.app")
ARCHIA_TOKEN = os.getenv("ARCHIA_TOKEN")

def test_analytics_agent():
    print("Testing agriflow-analytics agent...")
    print("-" * 70)

    payload = {
        "model": "agent:agriflow-analytics",
        "input": "What ML analytics tools and capabilities do you have? List them briefly.",
        "stream": False
    }

    response = requests.post(
        f"{ARCHIA_BASE_URL}/v1/responses",
        headers={
            "Authorization": f"Bearer {ARCHIA_TOKEN}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        print(f"[FAIL] Status: {response.status_code}")
        print(response.text)
        return False

    result = response.json()
    status = result.get("status")

    print(f"Status: {status}")

    if status == "completed":
        # Extract text
        for item in result.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        print(f"\nResponse ({len(text)} chars):")
                        print("-" * 70)
                        print(text[:800])
                        if len(text) > 800:
                            print("...")
                        print("-" * 70)
                        return True

    print("[FAIL] No output text found")
    return False

if __name__ == "__main__":
    success = test_analytics_agent()
    exit(0 if success else 1)
