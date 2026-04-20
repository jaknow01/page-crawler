"""
Claude Code agent runner.

Spawns a claude subprocess with a restricted --allowedTools list and
streams its output back to the caller. The agent has no access to the
filesystem or shell — only the explicitly listed MCP tools and WebFetch.
"""

import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.agents.job_queue import JobQueue

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Maximum time to wait after a usage limit hit before retrying (5 hours).
USAGE_LIMIT_WAIT_SECONDS = 5 * 3600

# Probe for usage limits if the agent is silent this long.
INACTIVITY_PROBE_SECONDS = _env_int("INACTIVITY_PROBE_SECONDS", 15 * 60)

# Keywords that indicate a Claude Code session usage limit error.
_USAGE_LIMIT_KEYWORDS = [
    "usage limit",
    "rate limit",
    "too many requests",
    "hit your limit",
    "claude ai usage",
    "exceeded your",
    "quota exceeded",
    "please slow down",
    "api error: terminated",  # Claude Code CLI termination due to limit
]


class UsageLimitError(RuntimeError):
    """Raised when the Claude Code CLI hits its session usage limit."""

    def __init__(self, message: str, resume_at: float | None = None) -> None:
        super().__init__(message)
        # True when reset time was parsed from the error message (exact).
        # False when falling back to a 5h ceiling (probe strategy).
        self.resume_at_is_exact = resume_at is not None
        self.resume_at = resume_at if resume_at is not None else (time.time() + USAGE_LIMIT_WAIT_SECONDS)


def _parse_resume_at(error_msg: str) -> float | None:
    """
    Try to extract a reset timestamp from the Claude Code error message.
    Claude Code sometimes emits "Your limit resets at HH:MM AM/PM UTC".
    Returns epoch float (+60s safety buffer), or None if not parseable.
    """
    import re
    # Patterns like "resets at 3:45 PM UTC" or "resets 5pm (UTC)"
    match = re.search(
        r"resets?\s*(?:at\s*)?(\d{1,2}(?::\d{2})?\s*[AP]M|\d{1,2}:\d{2})\s*\(?UTC\)?",
        error_msg,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        from datetime import timedelta
        time_str = match.group(1).strip().upper()
        if "AM" in time_str or "PM" in time_str:
            fmt = "%I%p" if ":" not in time_str else "%I:%M %p"
        else:
            fmt = "%H:%M"
        now = datetime.now(timezone.utc)
        parsed = datetime.strptime(time_str, fmt).replace(
            year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc
        )
        # If the parsed time is in the past, assume it's tomorrow.
        if parsed.timestamp() < now.timestamp():
            parsed += timedelta(days=1)
        # +60s buffer — Anthropic's interval may be open-ended.
        return (parsed + timedelta(seconds=60)).timestamp()
    except Exception:
        return None


def _check_usage_limit(text: str) -> None:
    """Raise UsageLimitError if text contains usage limit indicators."""
    lower = text.lower()
    if any(kw in lower for kw in _USAGE_LIMIT_KEYWORDS):
        resume_at = _parse_resume_at(text)
        raise UsageLimitError(text, resume_at=resume_at)


def _probe_usage_limit() -> bool:
    """
    Send a minimal request to Claude to check if the usage limit is still active.
    Returns True if still limited, False if we can resume.
    """
    limited, _ = _probe_usage_limit_info()
    return limited


def _probe_usage_limit_info() -> tuple[bool, float | None]:
    """
    Like _probe_usage_limit, but also returns a parsed resume time when available.
    """
    try:
        result = subprocess.run(
            ["claude", "--output-format", "stream-json", "-p", "reply with: ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        limited = any(kw in combined.lower() for kw in _USAGE_LIMIT_KEYWORDS)
        resume_at = _parse_resume_at(combined) if limited else None
        return limited, resume_at
    except Exception as exc:
        logger.debug("Usage limit probe failed: %s — assuming still limited", exc)
        return True, None


# Tools the agent is allowed to use without user confirmation.
# Covers: Firecrawl (search/crawl/scrape), custom DB MCP, Playwright,
# and WebFetch as a read-only fallback.
ALLOWED_TOOLS = ",".join([
    "mcp__firecrawl__firecrawl_search",
    "mcp__firecrawl__firecrawl_map",
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
    "Bash(firecrawl *)",
    "WebFetch",
])


def run_agent(
    prompt: str,
    mcp_config_path: str | Path = "mcp_config.json",
    cwd: str | Path | None = None,
    model: str | None = None,
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
        "--no-session-persistence",  # each worker starts clean, no prior runs loaded
        "--verbose",
        "-p", prompt,
    ]
    if model:
        cmd += ["--model", model]

    logger.debug("Spawning agent: %s", " ".join(cmd[:4]))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
    )

    output_parts: list[str] = []
    saw_output = False

    last_activity_at = [time.time()]
    next_probe_at = [last_activity_at[0] + INACTIVITY_PROBE_SECONDS]
    activity_lock = threading.Lock()
    watchdog_stop = threading.Event()
    limit_exception: list[UsageLimitError | None] = [None]

    def _mark_activity() -> None:
        now = time.time()
        with activity_lock:
            last_activity_at[0] = now
            next_probe_at[0] = now + INACTIVITY_PROBE_SECONDS

    def _terminate_process() -> None:
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass

    def _watchdog() -> None:
        while not watchdog_stop.is_set():
            time.sleep(5)
            if watchdog_stop.is_set():
                break
            with activity_lock:
                now = time.time()
                if now < next_probe_at[0]:
                    continue
                idle_seconds = now - last_activity_at[0]
            idle_mins = max(1, int(idle_seconds / 60))
            logger.info("[agent] no output for %d min — probing usage limit", idle_mins)
            limited, resume_at = _probe_usage_limit_info()
            if limited:
                limit_exception[0] = UsageLimitError("usage limit probe: denied", resume_at=resume_at)
                watchdog_stop.set()
                _terminate_process()
                return
            with activity_lock:
                next_probe_at[0] = time.time() + INACTIVITY_PROBE_SECONDS

    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()

    # Stream stdout line by line, parse stream-json events
    try:
        for raw_line in process.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            _mark_activity()
            if not saw_output:
                saw_output = True
                logger.info("[agent] output stream started")
            # Check every line (JSON or not) for usage limit hints.
            _check_usage_limit(raw_line)
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                # Non-JSON line (e.g. MCP server startup messages)
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

            elif event_type == "rate_limit_event":
                # Claude Code emits this when approaching or hitting the 5-hour limit.
                info = event.get("rate_limit_info", {})
                if info.get("status") == "denied":
                    resets_at = info.get("resetsAt")
                    resume_at = (resets_at + 60) if resets_at else None  # +60s buffer
                    raise UsageLimitError(
                        f"rate_limit_event: denied (resetsAt={resets_at})",
                        resume_at=resume_at,
                    )

            elif event_type == "result":
                # Final result event — subtype indicates success or error
                subtype = event.get("subtype", "")
                if subtype == "error":
                    error_msg = event.get("error", "unknown error")
                    logger.error("Agent reported error: %s", error_msg)
                    _check_usage_limit(error_msg)
    except UsageLimitError:
        watchdog_stop.set()
        _terminate_process()
        raise
    finally:
        watchdog_stop.set()
        watchdog_thread.join(timeout=1)

    process.wait()

    stderr_output = process.stderr.read()
    if stderr_output:
        if not saw_output:
            saw_output = True
            logger.info("[agent] stderr output received before stdout")
        logger.debug("Agent stderr: %s", stderr_output)
        _check_usage_limit(stderr_output)

    if limit_exception[0] is not None:
        raise limit_exception[0]

    if process.returncode != 0:
        no_output_note = " (no output received)" if not saw_output else ""
        raise RuntimeError(
            f"claude subprocess exited with code {process.returncode}{no_output_note}.\n"
            f"stderr: {stderr_output}"
        )

    return "".join(output_parts)


def run_agents_parallel(
    prompt_fn: Callable[[str], str],
    queue: JobQueue,
    workers: int = 1,
    mcp_config_path: str | Path = "mcp_config.json",
    cwd: str | Path | None = None,
    model: str | None = None,
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

    # Shared pause state.
    # _resume_at[0]       — epoch time after which workers may resume
    # _resume_is_exact[0] — True if the time came from the error message (probe skipped),
    #                       False if it's a fallback (probe every hour)
    # _limit_start[0]     — epoch time when the limit was first hit (fallback ceiling)
    _pause_lock = threading.Lock()
    _resume_at: list[float] = []
    _resume_is_exact: list[bool] = []
    _limit_start: list[float] = []

    # Only one worker probes at a time; others wait for the probe to finish.
    _probe_lock = threading.Lock()

    def _clear_pause() -> None:
        _resume_at.clear()
        _resume_is_exact.clear()
        _limit_start.clear()

    def _wait_if_paused() -> None:
        while True:
            with _pause_lock:
                if not _resume_at:
                    return  # no active pause
                exact = _resume_is_exact[0]
                deadline = _resume_at[0]
                started = _limit_start[0]

            now = time.time()

            if exact:
                # Exact reset time known — sleep until then, then clear.
                if now >= deadline:
                    with _pause_lock:
                        _clear_pause()
                    return
                wait_secs = deadline - now
                logger.info(
                    "[worker] usage limit pause (exact) — %d min remaining",
                    int(wait_secs / 60),
                )
                time.sleep(min(wait_secs, 60))

            else:
                # No exact time — probe Claude every hour, max 5h from limit start.
                hard_ceiling = started + USAGE_LIMIT_WAIT_SECONDS
                if now >= hard_ceiling:
                    logger.warning("[worker] usage limit: 5h ceiling reached — resuming anyway")
                    with _pause_lock:
                        _clear_pause()
                    return

                if now >= deadline:
                    # Time for a probe — only one worker at a time.
                    if _probe_lock.acquire(blocking=False):
                        try:
                            logger.info("[worker] probing Claude for usage limit status...")
                            still_limited = _probe_usage_limit()
                            if not still_limited:
                                logger.info("[worker] usage limit cleared — resuming")
                                with _pause_lock:
                                    _clear_pause()
                                return
                            # Still limited — push next probe 1h forward.
                            logger.info("[worker] still limited — next probe in 60 min")
                            with _pause_lock:
                                if _resume_at:
                                    _resume_at[0] = time.time() + 3600
                        finally:
                            _probe_lock.release()
                    else:
                        # Another worker is probing — wait a moment and re-check.
                        time.sleep(5)
                else:
                    mins_left = int((deadline - now) / 60)
                    logger.info(
                        "[worker] usage limit pause (probe in %d min)",
                        mins_left,
                    )
                    time.sleep(min(deadline - now, 60))

    def _worker() -> None:
        nonlocal done_count, failed_count
        while True:
            _wait_if_paused()

            query = queue.claim_next()
            if query is None:
                break  # queue exhausted
            logger.info("[worker] claimed query: %r", query)
            try:
                run_agent(
                    prompt=prompt_fn(query),
                    mcp_config_path=mcp_config_path,
                    cwd=cwd,
                    model=model,
                )
                queue.mark_done(query)
                done_count += 1
                logger.info("[worker] done: %r", query)
            except UsageLimitError as exc:
                # Return the job to the queue and pause all workers.
                queue.reset_to_pending(query)
                exact = exc.resume_at_is_exact
                with _pause_lock:
                    if not _resume_at:
                        _resume_at.append(exc.resume_at)
                        _resume_is_exact.append(exact)
                        _limit_start.append(time.time())
                    else:
                        _resume_at[0] = max(_resume_at[0], exc.resume_at)
                        # Once we have an exact time, keep it exact.
                        if exact:
                            _resume_is_exact[0] = True
                resume_dt = datetime.fromtimestamp(_resume_at[0], tz=timezone.utc).strftime("%H:%M UTC")
                mode = "exact" if _resume_is_exact[0] else "probe every 1h, max 5h"
                logger.warning(
                    "[worker] usage limit hit on %r — pausing until %s (%s)",
                    query, resume_dt, mode,
                )
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
