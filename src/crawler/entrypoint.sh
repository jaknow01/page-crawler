#!/bin/sh
set -e

if [ -n "$FIRECRAWL_API_KEY" ]; then
    firecrawl login --api-key "$FIRECRAWL_API_KEY"
fi

exec uv run python main.py "$@"
