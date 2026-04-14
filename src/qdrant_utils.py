"""Shared Qdrant constants and helpers used across analysis and MCP server modules."""

import os
from pathlib import Path

# Number of points fetched per scroll request.
# With with_vectors=True each point is ~6 KB (1536 floats × 4 bytes),
# so SCROLL_BATCH=256 ≈ 1.5 MB per request — a safe default.
SCROLL_BATCH = 256

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "pages"
REPORTS_DIR = Path("data/reports")
