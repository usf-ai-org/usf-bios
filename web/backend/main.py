# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Entry Point

Run with: 
  uvicorn main:app --reload --port 8000
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the app from the structured package
from app.main import app

# Re-export for uvicorn
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
