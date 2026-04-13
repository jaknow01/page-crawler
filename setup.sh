#!/usr/bin/env bash
# Setup script — run once after cloning the repo.
# Installs Claude Code plugins and Python dependencies.

set -euo pipefail

echo "==> Installing Python dependencies (uv)..."
uv sync

echo "==> Installing Claude Code plugins..."
claude plugin install frontend-design
claude plugin install playwright
claude plugin install firecrawl

echo "==> Done. Next steps:"
echo "    1. Copy .env.example to .env and fill in your API keys"
echo "    2. Start Qdrant:  docker compose up -d qdrant"
echo "    3. Run the crawler: uv run python main.py --help"
