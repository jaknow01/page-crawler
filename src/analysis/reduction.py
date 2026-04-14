"""
UMAP dimensionality reduction.

Fetches all vectors (own + competitor) from Qdrant, runs UMAP to project
them to 2D, and saves the result to data/reports/reduction.parquet.
The parquet file is the primary data source for the Streamlit UI.
"""

import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from umap import UMAP

from src.qdrant_utils import COLLECTION, QDRANT_URL, REPORTS_DIR, SCROLL_BATCH

load_dotenv()

logger = logging.getLogger(__name__)


def _fetch_all_vectors(client: QdrantClient) -> pd.DataFrame:
    """
    Scroll through all points (own + competitor) and return a DataFrame with
    columns: id, url, title, domain, source, query, summary, keywords,
             cluster_id, cluster_label, vector.
    """
    rows = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in batch:
            p = point.payload or {}
            rows.append({
                "id": str(point.id),
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "domain": p.get("domain", ""),
                "source": p.get("source", ""),
                "query": p.get("query", ""),
                "summary": p.get("summary", ""),
                "keywords": p.get("keywords", []),
                "cluster_id": p.get("cluster_id", -1),
                "cluster_label": p.get("cluster_label", ""),
                "content_snippet": p.get("content_snippet", ""),
                "timestamp": p.get("timestamp", ""),
                "vector": point.vector,
            })
        if offset is None:
            break

    return pd.DataFrame(rows)


def run_reduction(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    Run UMAP on all stored vectors and save the 2D projection to parquet.

    Args:
        n_neighbors:  UMAP n_neighbors (controls local vs global structure).
        min_dist:     UMAP min_dist (controls point spread).
        random_state: For reproducibility.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=QDRANT_URL)

    logger.info("Fetching all vectors from Qdrant...")
    df = _fetch_all_vectors(client)

    if df.empty:
        logger.error("No vectors found. Run crawl-own and crawl-competitors first.")
        return

    logger.info("Fetched %d vectors total (%d own, %d competitor)",
                len(df),
                (df["source"] == "own").sum(),
                (df["source"] == "competitor").sum())

    vectors = np.stack(df["vector"].values).astype(np.float32)
    vectors_normed = normalize(vectors, norm="l2")

    logger.info("Running UMAP (n_neighbors=%d, min_dist=%.2f)...", n_neighbors, min_dist)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    embedding = reducer.fit_transform(vectors_normed)

    df["x"] = embedding[:, 0].astype(np.float32)
    df["y"] = embedding[:, 1].astype(np.float32)

    # Drop the raw vector column before saving — saves significant disk space
    df = df.drop(columns=["vector"])

    # Serialise keywords list as comma-separated string for parquet compatibility
    df["keywords"] = df["keywords"].apply(
        lambda kw: ", ".join(kw) if isinstance(kw, list) else str(kw)
    )

    out_path = REPORTS_DIR / "reduction.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("2D reduction saved to %s (%d rows)", out_path, len(df))
