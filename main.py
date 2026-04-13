"""
SEO Competitor Analysis — main CLI orchestrator.

Usage:
    uv run python main.py --help
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.agents.competitor_prompt import build_competitor_prompt
from src.agents.job_queue import JobQueue
from src.agents.own_site_prompt import build_own_site_prompt
from src.agents.runner import run_agent, run_agents_parallel
from src.analysis.clustering import run_clustering
from src.analysis.gap_analysis import run_gap_analysis
from src.analysis.reduction import run_reduction

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
QUERIES_PATH = ROOT / "queries.yaml"
MCP_CONFIG_PATH = ROOT / "mcp_config.json"
JOBS_DB_PATH = ROOT / "data" / "jobs.db"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_queries() -> list[str]:
    with open(QUERIES_PATH) as f:
        data = yaml.safe_load(f)
    return data["queries"]


def resolve_workers(config: dict) -> int:
    """CRAWLER_WORKERS env var takes priority over config.yaml."""
    env_val = os.environ.get("CRAWLER_WORKERS")
    if env_val is not None:
        return int(env_val)
    return int(config.get("crawler_workers", 1))


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """SEO Semantic Competitor Analysis — crawl, embed, cluster, visualise."""


# ── crawl-own ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--metadata-mode", default=None, help="Override metadata_mode (llm|tfidf)")
def crawl_own(metadata_mode):
    """Crawl the client's own website and store all pages in the vector DB."""
    config = load_config()
    own_site_url = config["own_site_url"]
    max_pages = config.get("max_pages_per_domain", 100)
    mode = metadata_mode or config.get("metadata_mode", os.environ.get("METADATA_MODE", "tfidf"))

    click.echo(f"Crawling own site: {own_site_url} (max {max_pages} pages, metadata={mode})")

    prompt = build_own_site_prompt(
        own_site_url=own_site_url,
        max_pages=max_pages,
        metadata_mode=mode,
    )

    run_agent(prompt=prompt, mcp_config_path=MCP_CONFIG_PATH, cwd=ROOT)
    click.echo("Own site crawl complete.")


# ── crawl-competitors ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--workers", default=None, type=int, help="Number of parallel agents")
@click.option("--query-limit", default=None, type=int, help="Process only the first N queries (for testing)")
@click.option("--reset", is_flag=True, help="Reinitialise the job queue (re-run all queries)")
@click.option("--metadata-mode", default=None, help="Override metadata_mode (llm|tfidf)")
def crawl_competitors(workers, query_limit, reset, metadata_mode):
    """Crawl competitor websites for all search queries (supports parallel workers)."""
    config = load_config()
    n_workers = workers or resolve_workers(config)
    k = config.get("k", 5)
    max_pages = config.get("max_pages_per_domain", 100)
    max_depth = config.get("max_depth", 3)
    mode = metadata_mode or config.get("metadata_mode", os.environ.get("METADATA_MODE", "tfidf"))

    queries = load_queries()
    if query_limit:
        queries = queries[:query_limit]

    click.echo(f"Queries: {len(queries)} | Workers: {n_workers} | k={k} | metadata={mode}")

    queue = JobQueue(JOBS_DB_PATH)
    queue.initialize(queries, reset=reset)

    stats_before = queue.stats()
    pending = stats_before["pending"]
    if pending == 0:
        click.echo("No pending queries — all done. Use --reset to re-run.")
        return

    click.echo(f"Pending: {pending} | Done: {stats_before['done']} | Failed: {stats_before['failed']}")

    def prompt_fn(query: str) -> str:
        return build_competitor_prompt(
            query=query,
            k=k,
            max_pages=max_pages,
            max_depth=max_depth,
            metadata_mode=mode,
        )

    result = run_agents_parallel(
        prompt_fn=prompt_fn,
        queue=queue,
        workers=n_workers,
        mcp_config_path=MCP_CONFIG_PATH,
        cwd=ROOT,
    )

    click.echo(f"\nDone: {result['done']} | Failed: {result['failed']}")
    final = queue.stats()
    if final["failed"]:
        click.echo(f"Failed queries ({final['failed']}):")
        for item in final["failed_queries"]:
            click.echo(f"  - {item['query']}: {item['error'][:120]}")
        click.echo("Run `retry-failed` to re-queue them.")


# ── retry-failed ──────────────────────────────────────────────────────────────

@cli.command()
def retry_failed():
    """Reset all failed queries back to pending so they will be retried."""
    queue = JobQueue(JOBS_DB_PATH)
    n = queue.retry_failed()
    click.echo(f"Reset {n} failed jobs to pending.")


# ── queue-stats ───────────────────────────────────────────────────────────────

@cli.command()
def queue_stats():
    """Show current job queue status (pending / in_progress / done / failed)."""
    queue = JobQueue(JOBS_DB_PATH)
    s = queue.stats()
    click.echo(f"Total:       {s['total']}")
    click.echo(f"Pending:     {s['pending']}")
    click.echo(f"In progress: {s['in_progress']}")
    click.echo(f"Done:        {s['done']}")
    click.echo(f"Failed:      {s['failed']}")
    if s["failed_queries"]:
        click.echo("\nFailed queries:")
        for item in s["failed_queries"]:
            click.echo(f"  - {item['query']}")


# ── stats (DB) ────────────────────────────────────────────────────────────────

@cli.command()
def stats():
    """Show vector database statistics (own pages, competitor pages, domains)."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    try:
        total = client.count("pages", exact=True).count
    except Exception as e:
        click.echo(f"Cannot connect to Qdrant at {qdrant_url}: {e}")
        sys.exit(1)

    own = client.count(
        "pages",
        count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="own"))]),
        exact=True,
    ).count

    competitor = client.count(
        "pages",
        count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="competitor"))]),
        exact=True,
    ).count

    click.echo(f"Total vectors:      {total}")
    click.echo(f"Own site pages:     {own}")
    click.echo(f"Competitor pages:   {competitor}")


# ── analyze ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--n-clusters", default=None, type=int, help="Override number of clusters")
def analyze(n_clusters):
    """Run clustering, UMAP reduction and gap analysis on the vector database."""
    config = load_config()
    min_cluster_size = config.get("min_cluster_size", 5)

    click.echo("Step 1/3 — Clustering competitor pages...")
    run_clustering(min_cluster_size=min_cluster_size, n_clusters=n_clusters)

    click.echo("Step 2/3 — UMAP dimensionality reduction...")
    run_reduction()

    click.echo("Step 3/3 — Gap analysis...")
    run_gap_analysis(coverage_threshold=config.get("coverage_threshold", 0.3))

    click.echo("Analysis complete. Results saved to data/reports/.")


# ── run-all ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--workers", default=None, type=int)
@click.option("--query-limit", default=None, type=int)
@click.pass_context
def run_all(ctx, workers, query_limit):
    """Run the full pipeline: crawl-own → crawl-competitors → analyze."""
    ctx.invoke(crawl_own)
    ctx.invoke(crawl_competitors, workers=workers, query_limit=query_limit)
    ctx.invoke(analyze)


# ── ui ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8501, help="Streamlit port")
def ui(port):
    """Launch the Streamlit visualisation UI."""
    subprocess.run(
        ["uv", "run", "streamlit", "run", "src/ui/app.py", "--server.port", str(port)],
        cwd=ROOT,
        check=True,
    )


# ── token-stats ───────────────────────────────────────────────────────────────

@cli.command()
@click.option("--total-pages", default=None, type=int, help="Project cost to this many pages")
def token_stats(total_pages):
    """Show OpenAI token usage and estimated cost for the current run."""
    counter_path = ROOT / "data" / "token_stats.json"
    if not counter_path.exists():
        click.echo("No token stats recorded yet. Run a crawl first.")
        return

    data = json.loads(counter_path.read_text())

    click.echo(f"Pages processed:      {data.get('pages_processed', 0)}")
    click.echo(f"Embedding tokens:     {data.get('embedding_tokens', 0):,}")
    click.echo(f"LLM input tokens:     {data.get('llm_input_tokens', 0):,}")
    click.echo(f"LLM output tokens:    {data.get('llm_output_tokens', 0):,}")
    click.echo(f"Estimated cost:       ${data.get('estimated_cost_usd', 0):.4f}")

    if total_pages and data.get("pages_processed", 0) > 0:
        ratio = total_pages / data["pages_processed"]
        projected = data.get("estimated_cost_usd", 0) * ratio
        click.echo(f"Projected ({total_pages} pages): ${projected:.2f}")


if __name__ == "__main__":
    cli()
