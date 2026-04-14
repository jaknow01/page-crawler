"""
Custom MCP Server — SEO Vector Database

Exposes tools for the Claude Code agent:
  - add_page_to_db      — embed a page and save it to Qdrant
  - check_domain_exists — check whether a domain is already in the database
  - get_db_stats        — database statistics
  - list_own_pages      — list URLs of the client's own site
"""

import hashlib
import os
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.pl import Polish as SpacyPolish

from src.mcp_server.return_types import (
    AddPageResult,
    DbStatsResult,
    DomainExistsResult,
    OwnPagesResult,
    PageItem,
)
from src.qdrant_utils import COLLECTION as COLLECTION_NAME, QDRANT_URL, SCROLL_BATCH

load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_SIZE = 1536  # text-embedding-3-small

# Metadata generation mode: "llm" | "tfidf" — can be overridden via METADATA_MODE env var
METADATA_MODE = os.environ.get("METADATA_MODE", "tfidf")

# ── clients ───────────────────────────────────────────────────────────────────

qdrant = QdrantClient(url=QDRANT_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 381 Polish stop words from spaCy — no model download required
_POLISH_STOP_WORDS = SpacyPolish().Defaults.stop_words

# ── collection init ───────────────────────────────────────────────────────────


def ensure_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


ensure_collection()

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("seo-db")


def _embed(text: str) -> list[float]:
    """Compute an embedding vector via the OpenAI API."""
    text = text[:30_000]
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _metadata_tfidf(title: str, content: str, n_keywords: int = 8) -> dict:
    """
    Deterministic keyword extraction using TF-IDF on the full page content.
    Does not generate a summary.

    Returns {"summary": "", "keywords": [...], "metadata_mode": "tfidf"}.
    """
    # Title is repeated 3× to boost its weight in the TF-IDF score
    document = f"{title} {title} {title} {content}"

    try:
        vec = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}\b",
            stop_words=list(_POLISH_STOP_WORDS),
        )
        tfidf_matrix = vec.fit_transform([document])
        scores = zip(vec.get_feature_names_out(), tfidf_matrix.toarray()[0])
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:n_keywords]
        keywords = [kw for kw, _ in top if kw.lower() not in _POLISH_STOP_WORDS]
    except Exception:
        keywords = []

    return {"summary": "", "keywords": keywords, "metadata_mode": "tfidf"}


class _PageMetadata(BaseModel):
    summary: str
    keywords: list[str]


def _metadata_llm(title: str, content: str) -> dict:
    """
    Generate a summary and keywords via gpt-4o-mini using Structured Outputs.
    Uses up to ~15 000 characters of content.
    Falls back to empty values if the API call fails.

    Returns {"summary": str, "keywords": list[str], "metadata_mode": "llm"}.
    """
    prompt = (
        f"Strona: {title}\n\n"
        f"{content[:15_000]}\n\n"
        "Wygeneruj po polsku:\n"
        "- summary: krótkie podsumowanie (max 2 zdania) oddające główną tematykę podstrony\n"
        "- keywords: lista 6-10 słów kluczowych lub fraz (1-3 słowa) charakterystycznych dla tej strony"
    )
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            response_format=_PageMetadata,
        )
        result = response.choices[0].message.parsed
        return {
            "summary": result.summary,
            "keywords": result.keywords,
            "metadata_mode": "llm",
        }
    except Exception:
        return {"summary": "", "keywords": [], "metadata_mode": "llm"}


def _generate_metadata(title: str, content: str, mode: str) -> dict:
    """Dispatch metadata generation to the appropriate method based on mode."""
    if mode == "tfidf":
        return _metadata_tfidf(title, content)
    return _metadata_llm(title, content)


def _page_id(url: str) -> str:
    """Derive a deterministic point ID from a URL to enable upsert without duplicates."""
    return hashlib.md5(url.encode()).hexdigest()


@mcp.tool()
def add_page_to_db(
    url: str,
    title: str,
    content: str,
    source: str,
    domain: str,
    metadata_mode: Literal["tfidf", "llm"],
    query: str = "",
) -> AddPageResult:
    """
    Embed a page and save it to the Qdrant vector database.

    Args:
        url:           Full URL of the page.
        title:         Page title (from <title> tag or H1 heading).
        content:       Plain text content of the page (clean text or markdown).
        source:        Origin — "own" for the client's site, "competitor" for competitor sites.
        domain:        Domain name (e.g. "example.com").
        metadata_mode: Metadata generation mode — "tfidf" or "llm".
                       "tfidf" → keywords via TF-IDF on full content, no summary
                       "llm"   → summary + keywords via gpt-4o-mini (up to 15 000 chars)
        query:         Google search query that returned this page (empty for own site pages).

    Returns:
        AddPageResult with success flag, point id, and metadata_mode used.
    """
    mode = metadata_mode if metadata_mode in ("llm", "tfidf") else METADATA_MODE

    vector = _embed(f"{title}\n\n{content}")
    meta = _generate_metadata(title, content, mode)
    point_id = _page_id(url)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "source": source,
                    "query": query,
                    "summary": meta["summary"],
                    "keywords": meta["keywords"],
                    "metadata_mode": meta["metadata_mode"],
                    "content_snippet": content[:500],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )

    return AddPageResult(
        success=True,
        id=point_id,
        metadata_mode=meta["metadata_mode"],
    )


@mcp.tool()
def check_domain_exists(domain: str) -> DomainExistsResult:
    """
    Check whether pages from a given domain are already stored in the database (deduplication).

    Args:
        domain: Domain name (e.g. "example.com").

    Returns:
        DomainExistsResult with exists flag and page count.
    """
    results = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="domain", match=MatchValue(value=domain)),
                FieldCondition(key="source", match=MatchValue(value="competitor")),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    points, _ = results

    if not points:
        return DomainExistsResult(exists=False, page_count=0)

    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(key="domain", match=MatchValue(value=domain)),
                FieldCondition(key="source", match=MatchValue(value="competitor")),
            ]
        ),
        exact=True,
    ).count

    return DomainExistsResult(exists=True, page_count=count)


@mcp.tool()
def get_db_stats() -> DbStatsResult:
    """
    Return statistics about the vector database.

    Returns:
        DbStatsResult with total counts and list of unique competitor domains.
    """
    total = qdrant.count(collection_name=COLLECTION_NAME, exact=True).count

    own_count = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="own"))]
        ),
        exact=True,
    ).count

    competitor_count = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="competitor"))]
        ),
        exact=True,
    ).count

    domains: set[str] = set()
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="competitor"))]
            ),
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["domain"],
            with_vectors=False,
        )
        for point in batch:
            if point.payload and "domain" in point.payload:
                domains.add(point.payload["domain"])
        if offset is None:
            break

    return DbStatsResult(
        total=total,
        own_pages=own_count,
        competitor_pages=competitor_count,
        unique_domains=sorted(domains),
    )


@mcp.tool()
def list_own_pages() -> OwnPagesResult:
    """
    Return a list of the client's own site pages stored in the database.

    Returns:
        OwnPagesResult with a list of page url/title pairs.
    """
    pages: list[PageItem] = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="own"))]
            ),
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["url", "title"],
            with_vectors=False,
        )
        for point in batch:
            if point.payload:
                pages.append(PageItem(
                    url=point.payload.get("url", ""),
                    title=point.payload.get("title", ""),
                ))
        if offset is None:
            break

    return OwnPagesResult(pages=pages)


if __name__ == "__main__":
    mcp.run()
