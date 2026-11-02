"""
Vercel Serverless Entrypoint for FastAPI Backend.
Exposes the initialized FastAPI application object for Vercel Python runtime.
"""

import sys
from pathlib import Path

# Ensure root and backend directories are on sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app import app

