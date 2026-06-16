from dotenv import load_dotenv
import os
import subprocess
from pathlib import Path

load_dotenv()

TOKEN = os.getenv("DAGSHUB_TOKEN")
USERNAME = os.getenv("REPO_OWNER", "rsm-chl307")

if not TOKEN:
    raise ValueError("DAGSHUB_TOKEN not found in .env")

config_local = Path(".dvc/config.local")

# Skip if already set up
if config_local.exists():
    print("DVC authentication already configured. Skipping.")
    exit(0)

print("Configuring DVC authentication...")

subprocess.run(
    ["dvc", "remote", "modify", "origin", "auth", "basic", "--local"],
    check=True,
)

subprocess.run(
    ["dvc", "remote", "modify", "origin", "user", USERNAME, "--local"],
    check=True,
)

subprocess.run(
    ["dvc", "remote", "modify", "origin", "password", TOKEN, "--local"],
    check=True,
)

print("DVC authentication configured successfully.")