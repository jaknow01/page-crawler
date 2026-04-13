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

1. Use the Firecrawl `firecrawl_crawl` tool to crawl all subpages of `{own_site_url}`.
   - Limit: {max_pages} pages maximum.
   - Crawl only pages within the same domain (`{domain}`).
   - Retrieve clean text / markdown content for each page.

2. For every page returned, call `add_page_to_db` with:
   - `url`:           the full page URL
   - `title`:         the page title (from metadata or the first H1)
   - `content`:       the full text/markdown content
   - `source`:        "own"
   - `domain`:        "{domain}"
   - `metadata_mode`: "{metadata_mode}"
   - `query`:         "" (leave empty)

3. Skip pages that return an error or have no meaningful text content (e.g. pure image pages).

4. After processing all pages, call `get_db_stats` and report:
   - How many own pages were saved
   - Any pages that were skipped and why

## Rules
- Use only the MCP tools available to you. Do not attempt to use Bash, file access, or any other tools.
- Process pages one by one — do not batch or skip any page returned by the crawl.
- If `firecrawl_crawl` returns a partial result due to the page limit, process all returned pages before stopping.
"""
