# -*- coding: utf-8 -*-
"""
scripts/test_hf_token.py
------------------------
STEP 2 -- Verify your Hugging Face token works BEFORE running the full pipeline.

Run this with:
    python scripts/test_hf_token.py

What it does:
  1. Reads your HF token from .env
  2. Makes a tiny call to the HF Inference API (just a metadata check)
  3. Prints SUCCESS or a clear error message

This lets you catch auth problems early instead of discovering them mid-ingestion.
"""

import os
import sys

# Set stdout to UTF-8 so emoji work on Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env from the project root (one level up from scripts/)
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

def test_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not token or token == "hf_YOUR_TOKEN_HERE":
        print("\n[ERROR] No valid token found.")
        print("    Edit your .env file and set HUGGINGFACEHUB_API_TOKEN=hf_xxx...")
        print("    Get a free token at: https://huggingface.co/settings/tokens\n")
        sys.exit(1)

    print(f"\n[OK] Token found: {token[:8]}...{token[-4:]}")
    print("     Checking token with Hugging Face API...\n")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami()
        print(f"[SUCCESS] Logged in as: {user['name']}")
        print(f"    Email: {user.get('email', 'N/A')}")
        print(f"    Plan:  {user.get('plan', 'free')}")
        print()

        # Also check the generation model is accessible
        model_id = os.getenv("GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        print(f"    Checking access to model: {model_id} ...")
        try:
            model_info = api.model_info(model_id)
            print(f"[OK] Model '{model_id}' is accessible.")
            print(f"     Pipeline tag: {model_info.pipeline_tag}")
        except Exception as e:
            print(f"[WARN] Could not access model '{model_id}': {e}")
            print("       If this is a gated model (like Mistral), visit its HF page and accept the license.")
            print("       Or switch to: GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct (no license needed)")

        print("\n[DONE] Token test complete -- you're ready to proceed to Step 3!\n")

    except Exception as e:
        print(f"\n[ERROR] Token validation failed: {e}")
        print("    Make sure your token is correct and has Read permissions.\n")
        sys.exit(1)

if __name__ == "__main__":
    test_token()
