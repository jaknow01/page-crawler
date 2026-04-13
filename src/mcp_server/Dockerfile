FROM python:3.11-slim

# Instalacja uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Kopiujemy pliki projektu potrzebne do zainstalowania zależności
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Kopiujemy kod serwera
COPY src/mcp_server/db_server.py ./src/mcp_server/db_server.py
COPY src/__init__.py ./src/__init__.py
COPY src/mcp_server/__init__.py ./src/mcp_server/__init__.py

CMD ["uv", "run", "python", "src/mcp_server/db_server.py"]
