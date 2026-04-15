"""
Prompt template for the own-site crawl agent.
"""

from urllib.parse import urlparse


def build_own_site_prompt(
    own_site_url: str,
    max_pages: int = 200,
    metadata_mode: str = "tfidf",
) -> str:
    """
    Build the prompt for the agent that crawls the client's own website.

    Args:
        own_site_url:  Root URL of the site to crawl.
        max_pages:     Maximum number of pages to crawl.
        metadata_mode: Metadata generation mode passed to add_page_to_db.

    Returns:
        Full prompt string ready to pass to run_agent.
    """
    domain = urlparse(own_site_url).netloc

    return f"""Your task is to crawl the client's own website and store every page in the vector database.

## Steps

1. Use `firecrawl_map` to discover all URLs on `{own_site_url}`.
   - This returns a list of URLs — do NOT use `firecrawl_crawl` (its output is too large to process).
   - Limit to {max_pages} URLs maximum.
   - Only keep URLs within the domain `{domain}`.

2. For each URL from step 1, call `firecrawl_scrape` to get its content.
   - Process URLs one at a time (do not try to batch them).
   - Skip URLs that return errors or have no meaningful text content.

3. For each successfully scraped page, call `add_page_to_db` with:
   - `url`:           the full page URL
   - `title`:         the page title (from metadata or the first H1)
   - `content`:       the full text/markdown content
   - `source`:        "own"
   - `domain`:        "{domain}"
   - `metadata_mode`: "{metadata_mode}"
   - `query`:         "" (leave empty)

4. After processing all pages, call `get_db_stats` and report:
   - How many own pages were saved
   - Any pages that were skipped and why

## Rules
- Use only the MCP tools available to you. Do not use Bash, file access, or any other tools.
- IMPORTANT: Use `firecrawl_map` to discover URLs, then `firecrawl_scrape` per page — never `firecrawl_crawl`.
- Process pages one by one — do not skip any URL from the map.
"""
