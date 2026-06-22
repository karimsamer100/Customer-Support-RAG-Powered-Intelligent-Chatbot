import subprocess
import sys
from pathlib import Path

BUILD_SCRIPT = Path("build_index.py")

if not BUILD_SCRIPT.exists():
    print("Error: build_index.py not found in repository root.")
    sys.exit(1)

print("Starting vector index refresh (this runs build_index.py)...")
try:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True)
    print("Vector index refresh completed successfully.")
except subprocess.CalledProcessError as e:
    print("Vector index refresh failed:", e)
    sys.exit(2)
except Exception as e:
    print("Unexpected error while running build_index.py:", e)
    sys.exit(3)
