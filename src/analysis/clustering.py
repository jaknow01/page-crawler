"""
Competitor page clustering.

Fetches all competitor vectors from Qdrant, runs HDBSCAN (auto cluster count)
labels each cluster via keyword frequency counting,
then writes cluster assignments back to Qdrant payloads and to
data/reports/clusters.json.
"""

import json
import logging
from collections import defaultdict

import numpy as np
import hdbscan
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.preprocessing import normalize

from src.qdrant_utils import COLLECTION, QDRANT_URL, REPORTS_DIR, SCROLL_BATCH

load_dotenv()

logger = logging.getLogger(__name__)


def _fetch_competitor_vectors(client: QdrantClient) -> tuple[list[str], np.ndarray, list[dict]]:
    """
    Scroll through all competitor points and return:
      - ids:      list of point id strings
      - vectors:  float32 numpy array (N, D)
      - payloads: list of payload dicts
    """
    ids, vectors, payloads = [], [], []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="competitor"))]
            ),
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in batch:
            ids.append(point.id)
            vectors.append(point.vector)
            payloads.append(point.payload or {})
        if offset is None:
            break

    return ids, np.array(vectors, dtype=np.float32), payloads


def _label_cluster(payloads: list[dict], n_terms: int = 5) -> str:
    """
    Derive a human-readable label for a cluster by counting keyword frequency
    across all pages in the cluster and returning the top N most common terms.

    Falls back to the most common words in page titles if keywords are absent.
    """
    from collections import Counter

    all_keywords = []
    for p in payloads:
        all_keywords.extend(p.get("keywords", []))

    if all_keywords:
        counter = Counter(kw.lower().strip() for kw in all_keywords if kw.strip())
        top = [kw for kw, _ in counter.most_common(n_terms)]
        return " · ".join(top) if top else "unlabelled"

    # Fallback: split titles into words and count frequency
    all_words = []
    for p in payloads:
        all_words.extend(p.get("title", "").lower().split())

    if not all_words:
        return "unlabelled"

    counter = Counter(w for w in all_words if len(w) > 3)
    top = [w for w, _ in counter.most_common(n_terms)]
    return " · ".join(top) if top else "unlabelled"


def run_clustering(
    min_cluster_size: int = 5,
) -> None:
    """
    Cluster all competitor pages and persist assignments.

    Uses HDBSCAN with automatic cluster detection. Points that don't fit
    any cluster (label == -1) are kept as "unclassified" — this is useful
    information, not a failure. Tune min_cluster_size in config.yaml after
    inspecting real data.

    Args:
        min_cluster_size: Minimum points required to form a cluster.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=QDRANT_URL)

    logger.info("Fetching competitor vectors from Qdrant...")
    ids, vectors, payloads = _fetch_competitor_vectors(client)

    if len(ids) == 0:
        logger.error("No competitor pages found in the database. Run crawl-competitors first.")
        return

    logger.info("Fetched %d competitor vectors (dim=%d)", len(ids), vectors.shape[1])

    # Normalise to unit length — improves cosine-based clustering
    vectors_normed = normalize(vectors, norm="l2")

    logger.info("Running HDBSCAN (min_cluster_size=%d)...", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # on normalised vectors ≈ cosine
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vectors_normed)

    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).mean()
    logger.info("HDBSCAN: %d clusters, %.1f%% unclassified", n_found, noise_ratio * 100)

    unique_labels = sorted(set(labels))
    logger.info("Final cluster count: %d", len([l for l in unique_labels if l != -1]))

    # Group payloads by cluster to derive labels
    cluster_payloads: dict[int, list[dict]] = defaultdict(list)
    for label, payload in zip(labels, payloads):
        cluster_payloads[int(label)].append(payload)

    cluster_labels: dict[int, str] = {
        label: ("noise" if label == -1 else _label_cluster(pls))
        for label, pls in cluster_payloads.items()
    }

    # Write cluster assignments JSON (used by reduction.py and gap_analysis.py)
    assignments = [
        {
            "id": str(point_id),
            "cluster_id": int(label),
            "cluster_label": cluster_labels[int(label)],
        }
        for point_id, label in zip(ids, labels)
    ]
    out_path = REPORTS_DIR / "clusters.json"
    out_path.write_text(json.dumps({"clusters": assignments, "labels": {str(k): v for k, v in cluster_labels.items()}}, ensure_ascii=False, indent=2))
    logger.info("Cluster assignments saved to %s", out_path)

    # Persist cluster_id and cluster_label back into Qdrant payloads.
    # Group point IDs by cluster so each set_payload call covers one cluster —
    # one request per cluster instead of one request per point.
    logger.info("Writing cluster labels back to Qdrant...")
    points_by_cluster: dict[int, list] = defaultdict(list)
    for point_id, label in zip(ids, labels):
        points_by_cluster[int(label)].append(point_id)

    for label, point_ids in points_by_cluster.items():
        client.set_payload(
            collection_name=COLLECTION,
            payload={
                "cluster_id": label,
                "cluster_label": cluster_labels[label],
            },
            points=point_ids,
        )

    logger.info("Done — cluster labels written to Qdrant (%d calls).", len(points_by_cluster))
