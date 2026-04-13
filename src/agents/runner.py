"""
Claude Code agent runner.

Spawns a claude subprocess with a restricted --allowedTools list and
streams its output back to the caller. The agent has no access to the
filesystem or shell — only the explicitly listed MCP tools and WebFetch.
"""

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src.agents.job_queue import JobQueue

logger = logging.getLogger(__name__)

# Tools the agent is allowed to use without user confirmation.
# Covers: Firecrawl (search/crawl/scrape), custom DB MCP, Playwright,
# and WebFetch as a read-only fallback.
ALLOWED_TOOLS = ",".join([
    "mcp__firecrawl__firecrawl_search",
    "mcp__firecrawl__firecrawl_crawl",
    "mcp__firecrawl__firecrawl_scrape",
    "mcp__db__add_page_to_db",
    "mcp__db__check_domain_exists",
    "mcp__db__get_db_stats",
    "mcp__db__list_own_pages",
    "mcp__db__get_token_stats",
    "mcp__playwright__browser_navigate",
    "mcp__playwright__browser_snapshot",
    "mcp__playwright__browser_click",
    "mcp__playwright__browser_type",
    "mcp__playwright__browser_wait_for",
    "mcp__playwright__browser_close",
    "WebFetch",
])


def run_agent(
    prompt: str,
    mcp_config_path: str | Path = "mcp_config.json",
    cwd: str | Path | None = None,
) -> str:
    """
    Run a Claude Code agent as a subprocess and stream its output.

    The agent is restricted to ALLOWED_TOOLS — no filesystem or shell access.
    Output is streamed line-by-line to the caller's stdout and also returned
    as a single string when the process completes.

    Args:
        prompt:          The task prompt to pass to the agent via -p.
        mcp_config_path: Path to the MCP server config JSON file.
        cwd:             Working directory for the subprocess.
                         Defaults to the project root (parent of this file's package).

    Returns:
        The full text response from the agent.

    Raises:
        RuntimeError: If the claude process exits with a non-zero status code.
    """
    if cwd is None:
        cwd = Path(__file__).parent.parent.parent  # repo root

    cmd = [
        "claude",
        "--mcp-config", str(mcp_config_path),
        "--allowedTools", ALLOWED_TOOLS,
        "--output-format", "stream-json",
        "--no-color",
        "-p", prompt,
    ]

    logger.debug("Spawning agent: %s", " ".join(cmd[:4]))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
    )

    output_parts: list[str] = []

    # Stream stdout line by line, parse stream-json events
    for raw_line in process.stdout:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            # Non-JSON line (e.g. MCP server startup messages) — print as-is
            print(raw_line, flush=True)
            continue

        event_type = event.get("type", "")

        if event_type == "assistant":
            # Extract text content from assistant message blocks
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    text = block["text"]
                    output_parts.append(text)
                    print(text, end="", flush=True)

        elif event_type == "tool_use":
            tool = event.get("name", "unknown")
            logger.info("[tool] %s", tool)

        elif event_type == "result":
            # Final result event — subtype indicates success or error
            subtype = event.get("subtype", "")
            if subtype == "error":
                error_msg = event.get("error", "unknown error")
                logger.error("Agent reported error: %s", error_msg)

    process.wait()

    stderr_output = process.stderr.read()
    if stderr_output:
        logger.debug("Agent stderr: %s", stderr_output)

    if process.returncode != 0:
        raise RuntimeError(
            f"claude subprocess exited with code {process.returncode}.\n"
            f"stderr: {stderr_output}"
        )

    return "".join(output_parts)


def run_agents_parallel(
    prompt_fn: Callable[[str], str],
    queue: JobQueue,
    workers: int = 1,
    mcp_config_path: str | Path = "mcp_config.json",
    cwd: str | Path | None = None,
) -> dict:
    """
    Run up to `workers` Claude Code agents in parallel, each consuming one
    query at a time from the job queue.

    Args:
        prompt_fn:       Callable that receives a query string and returns
                         the full prompt to pass to the agent.
        queue:           JobQueue instance — provides claim_next / mark_done /
                         mark_failed. Must be initialized before calling.
        workers:         Number of parallel subprocesses (CRAWLER_WORKERS env var).
        mcp_config_path: Passed through to run_agent.
        cwd:             Working directory for subprocesses.

    Returns:
        {"done": int, "failed": int, "skipped": int}
    """
    done_count = 0
    failed_count = 0

    def _worker() -> None:
        nonlocal done_count, failed_count
        while True:
            query = queue.claim_next()
            if query is None:
                break  # queue exhausted
            logger.info("[worker] claimed query: %r", query)
            try:
                run_agent(
                    prompt=prompt_fn(query),
                    mcp_config_path=mcp_config_path,
                    cwd=cwd,
                )
                queue.mark_done(query)
                done_count += 1
                logger.info("[worker] done: %r", query)
            except Exception as exc:
                error_msg = str(exc)
                queue.mark_failed(query, error_msg)
                failed_count += 1
                logger.error("[worker] failed: %r — %s", query, error_msg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker) for _ in range(workers)]
        for future in as_completed(futures):
            # Re-raise any unexpected exception from a worker thread
            future.result()

    return {"done": done_count, "failed": failed_count}
