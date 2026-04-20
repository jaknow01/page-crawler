FROM python:3.11-slim

# System dependencies + Node.js 20 (needed for claude CLI and npx MCP servers)
RUN apt-get update && apt-get install -y curl build-essential && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Claude Code CLI + Firecrawl CLI
RUN npm install -g @anthropic-ai/claude-code firecrawl-cli

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds when pyproject.toml changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code and static config
# config.yaml, queries.yaml and data/ are volume-mounted at runtime
COPY src/ ./src/
COPY main.py ./
COPY mcp_config.json ./
COPY src/crawler/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--help"]
