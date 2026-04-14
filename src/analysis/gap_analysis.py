"""
Gap analysis — compares the client's own pages against competitor clusters.

For each cluster, computes how well the client's own pages cover that topic
(minimum cosine distance from any own page to the cluster centroid).
Produces a ranked report from most-missing to best-covered clusters,
saved to data/reports/gap_report.json.
"""

import json
import logging

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.preprocessing import normalize

from src.qdrant_utils import COLLECTION, QDRANT_URL, REPORTS_DIR, SCROLL_BATCH

load_dotenv()

logger = logging.getLogger(__name__)


# ── data fetching ─────────────────────────────────────────────────────────────

def _fetch_vectors_by_source(
    client: QdrantClient, source: str
) -> tuple[np.ndarray, list[dict]]:
    """Return (vectors, payloads) for all points with the given source."""
    vectors, payloads = [], []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in batch:
            vectors.append(point.vector)
            payloads.append(point.payload or {})
        if offset is None:
            break

    return np.array(vectors, dtype=np.float32), payloads


# ── cluster building ──────────────────────────────────────────────────────────

def _group_by_cluster(
    comp_normed: np.ndarray,
    comp_payloads: list[dict],
    max_examples: int = 5,
) -> dict[int, dict]:
    """
    Group competitor vectors and payloads by cluster_id.
    Noise points (cluster_id == -1) are excluded.
    Returns a dict keyed by cluster_id.
    """
    clusters: dict[int, dict] = {}
    for vec, payload in zip(comp_normed, comp_payloads):
        cluster_id = payload.get("cluster_id", -1)
        if cluster_id == -1:
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "cluster_label": payload.get("cluster_label", ""),
                "vectors": [],
                "example_pages": [],
            }
        clusters[cluster_id]["vectors"].append(vec)
        if len(clusters[cluster_id]["example_pages"]) < max_examples:
            clusters[cluster_id]["example_pages"].append({
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "domain": payload.get("domain", ""),
                "summary": payload.get("summary", ""),
                "keywords": payload.get("keywords", []),
            })
    return clusters


# ── per-cluster analysis ──────────────────────────────────────────────────────

def _cluster_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute the L2-normalised centroid of a list of unit vectors."""
    stacked = np.stack(vectors)
    return normalize(stacked.mean(axis=0, keepdims=True), norm="l2")[0]


def _coverage_score(min_distance: float, threshold: float) -> float:
    """
    Map minimum cosine distance to a [0, 1] coverage score.
    0 = cluster completely absent from own site, 1 = perfectly covered.
    """
    return float(max(0.0, 1.0 - min_distance / threshold))


def _analyse_cluster(
    cluster_data: dict,
    own_normed: np.ndarray,
    own_payloads: list[dict],
    coverage_threshold: float,
) -> dict:
    """
    Compute coverage metrics for a single cluster against own-site pages.
    Returns a report dict ready for gap_report.json.
    """
    centroid = _cluster_centroid(cluster_data["vectors"])

    # Cosine distance = 1 - dot product (vectors are unit-normalised)
    distances = 1.0 - own_normed @ centroid
    best_idx = int(np.argmin(distances))
    min_distance = float(distances[best_idx])

    return {
        "cluster_id": cluster_data["cluster_id"],
        "cluster_label": cluster_data["cluster_label"],
        "page_count": len(cluster_data["vectors"]),
        "covered": min_distance <= coverage_threshold,
        "coverage_score": round(_coverage_score(min_distance, coverage_threshold), 4),
        "min_distance_to_own": round(min_distance, 4),
        "best_own_page": {
            "url": own_payloads[best_idx].get("url", ""),
            "title": own_payloads[best_idx].get("title", ""),
            "distance": round(min_distance, 4),
        },
        "example_pages": cluster_data["example_pages"],
    }


# ── summary ───────────────────────────────────────────────────────────────────

def _build_summary(
    report_clusters: list[dict],
    coverage_threshold: float,
    own_count: int,
    competitor_count: int,
) -> dict:
    """Aggregate summary statistics over all analysed clusters."""
    return {
        "total_clusters": len(report_clusters),
        "covered_clusters": sum(1 for c in report_clusters if c["covered"]),
        "missing_clusters": sum(1 for c in report_clusters if not c["covered"]),
        "coverage_threshold": coverage_threshold,
        "own_pages_count": own_count,
        "competitor_pages_count": competitor_count,
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run_gap_analysis(coverage_threshold: float = 0.3) -> None:
    """
    Compare own-site pages against competitor clusters and produce a gap report.

    A cluster is considered "covered" if at least one own-site page has a
    cosine distance ≤ coverage_threshold to the cluster centroid.

    Args:
        coverage_threshold: Distance below which a cluster counts as covered.
                            Lower = stricter (requires closer match).

    Output: data/reports/gap_report.json
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=QDRANT_URL)

    own_vectors, own_payloads = _fetch_vectors_by_source(client, "own")
    comp_vectors, comp_payloads = _fetch_vectors_by_source(client, "competitor")

    if len(own_vectors) == 0:
        logger.error("No own-site pages found. Run crawl-own first.")
        return
    if len(comp_vectors) == 0:
        logger.error("No competitor pages found. Run crawl-competitors first.")
        return

    own_normed = normalize(own_vectors, norm="l2")
    comp_normed = normalize(comp_vectors, norm="l2")

    clusters = _group_by_cluster(comp_normed, comp_payloads)
    if not clusters:
        logger.error("No clustered competitor pages found. Run analyze first.")
        return

    logger.info("Analysing %d clusters against %d own pages...", len(clusters), len(own_normed))

    report_clusters = [
        _analyse_cluster(data, own_normed, own_payloads, coverage_threshold)
        for data in clusters.values()
    ]
    report_clusters.sort(key=lambda c: c["coverage_score"])

    summary = _build_summary(report_clusters, coverage_threshold, len(own_vectors), len(comp_vectors))
    report = {"summary": summary, "clusters": report_clusters}

    out_path = REPORTS_DIR / "gap_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    logger.info(
        "Gap analysis complete — %d missing / %d covered clusters. Saved to %s",
        summary["missing_clusters"],
        summary["covered_clusters"],
        out_path,
    )
