"""
Prompt template for the competitor crawl agent.
Each agent instance handles one Google search query.
"""

from urllib.parse import urlparse


def build_competitor_prompt(
    query: str,
    k: int = 5,
    max_pages: int = 100,
    max_depth: int = 3,
    metadata_mode: str = "tfidf",
) -> str:
    """
    Build the prompt for an agent that searches Google for one query,
    then crawls every new competitor domain it finds.

    Args:
        query:         The Google search query to process.
        k:             Number of top organic results to analyse.
        max_pages:     Maximum pages to crawl per domain.
        max_depth:     Maximum crawl depth per domain.
        metadata_mode: Metadata generation mode passed to add_page_to_db.

    Returns:
        Full prompt string ready to pass to run_agent.
    """
    return f"""Your task is to find competitor websites for a Google search query and store their content in the vector database.

## Search query
{query}

## Steps

### 1 — Search Google
Use `firecrawl_search` to retrieve the top {k} organic (non-sponsored) search results for the query above.
Extract the URL and domain of each result.

### 2 — Filter known domains
For each result domain, call `check_domain_exists` with the domain name.
- If `exists` is true → skip this domain entirely (already crawled by a previous run or another worker).
- If `exists` is false → proceed to step 3.

### 3 — Crawl new domains
For each new domain, use `firecrawl_crawl` to crawl the entire site:
- Start URL: the result URL from step 1.
- Limit: {max_pages} pages maximum.
- Max depth: {max_depth}.
- Stay within the same domain — do not follow links to external sites.

If `firecrawl_crawl` fails or is unavailable for a domain, fall back to `firecrawl_scrape` on the landing page only.

### 4 — Store pages
For every page returned by the crawl, call `add_page_to_db` with:
- `url`:           the full page URL
- `title`:         the page title (from metadata or first H1)
- `content`:       the full text/markdown content
- `source`:        "competitor"
- `domain`:        the domain name (e.g. "example.com")
- `query`:         "{query}"
- `metadata_mode`: "{metadata_mode}"

Skip pages with no meaningful text content.

### 5 — Report
After processing all domains, report:
- How many domains were new vs already known
- How many pages were saved per domain
- Any domains or pages that failed

## Rules
- Use only the MCP tools available to you. Do not use Bash, file access, or any other tools.
- Never follow links outside the domain being crawled.
- Process one domain fully before moving to the next.
- The `check_domain_exists` check is critical — always perform it before crawling.
"""


def domain_from_url(url: str) -> str:
    """Extract the domain (netloc) from a URL string."""
    return urlparse(url).netloc
