FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/ui/app.py",
     "--server.port=8501",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
