#!/usr/bin/env python
"""Diagnostic script to check Python environment."""

import sys
import os

print("=" * 60)
print("AgriFlow Environment Diagnostic")
print("=" * 60)

print(f"\n1. Python Version: {sys.version}")
print(f"2. Python Executable: {sys.executable}")
print(f"3. Current Directory: {os.getcwd()}")
print(f"4. Virtual Environment: {'Yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No'}")

print("\n5. Testing f-string syntax:")
try:
    test_var = "working"
    test_num = 42
    result = f"F-strings are {test_var}! Test number: {test_num}"
    print(f"   [OK] {result}")
except SyntaxError as e:
    print(f"   [FAIL] F-string syntax error: {e}")

print("\n6. Testing imports:")
imports_to_test = [
    "langchain",
    "langchain_core",
    "langgraph",
    "anthropic",
    "dotenv",
]

for module in imports_to_test:
    try:
        __import__(module)
        print(f"   [OK] {module}")
    except ImportError:
        print(f"   [FAIL] {module} - NOT INSTALLED")

print("\n7. Testing src.agent.graph import:")
try:
    from src.agent.graph import run_agent
    print("   [OK] src.agent.graph imports successfully")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n8. File existence check:")
files_to_check = ["run_agent.py", ".env", "requirements.txt", "src/agent/graph.py"]
for filepath in files_to_check:
    exists = "[OK]" if os.path.exists(filepath) else "[MISSING]"
    print(f"   {exists} {filepath}")

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)
