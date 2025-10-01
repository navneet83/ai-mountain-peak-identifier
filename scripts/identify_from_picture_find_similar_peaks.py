#!/usr/bin/env python3
# scripts/identify_and_similar.py
"""
Identify the most likely peak in an uploaded photo, then fetch similar photos
from photo folder.

Pipeline (high level)
---------------------
1) Embed the uploaded image with SigLIP-2 (IMAGE encoder).
2) Predict top-K peak *names* by running a kNN search against `peaks_catalog.text_embed`
   (text vectors). This gives human-readable candidates (e.g., "Ama Dablam").
3) Choose a query vector for "similar photos":
     - default: a TEXT vector built from the best peak name (robust to variations).
     - or, if --use-image-query: the IMAGE vector of the uploaded photo (more literal).
4) Search `photos.clip_image` with kNN to retrieve similar photos, optionally
   filtered to the predicted peak via a term filter on `predicted_peaks`.
5) Pretty-print results (path, predicted names, GPS, time, ES score).

Why TEXT query after identification?
------------------------------------
Using text("best peak") for the neighbor search tends to stabilize results across
lighting/cropping; the IMAGE vector is great too—use --use-image-query to compare.

Auth (API key)
--------------
Use one of:
  ES_CLOUD_ID + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)
  ES_URL      + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)

Examples
--------
# Identify + neighbors (default: TEXT query)
python scripts/identify_and_similar.py --image data/images/IMG_0001.jpg --topk 3 --neighbors 30

# Same but with IMAGE query (literal visual similarity)
python scripts/identify_and_similar.py --image data/images/IMG_0001.jpg --use-image-query

# Disable the post-id term filter on predicted_peaks (broader results)
python scripts/identify_and_similar.py --image data/images/IMG_0001.jpg --no-filter
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from elasticsearch import Elasticsearch
from PIL import Image, UnidentifiedImageError

# Optional HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

# --- repo path shim so we can import src/ai_mpi/embeddings.py ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ai_mpi.embeddings import Siglip2  # returns L2-normalized text/image vectors


# =============================================================================
# Elasticsearch client (API key)
# =============================================================================
def es_client() -> Elasticsearch:
    cloud_id = os.getenv("ES_CLOUD_ID")
    url = os.getenv("ES_URL", "http://localhost:9200")
    api_key_b64 = os.getenv("ES_API_KEY_B64")
    api_key_id = os.getenv("ES_API_KEY_ID")
    api_key = os.getenv("ES_API_KEY")

    if cloud_id:
        if api_key_b64:
            return Elasticsearch(cloud_id=cloud_id, api_key=api_key_b64)
        if api_key_id and api_key:
            return Elasticsearch(cloud_id=cloud_id, api_key=(api_key_id, api_key))
        raise SystemExit("Elastic Cloud: set ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY.")
    if api_key_b64:
        return Elasticsearch(url, api_key=api_key_b64)
    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    raise SystemExit("Set ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY (and ES_URL/ES_CLOUD_ID).")


# =============================================================================
# Helpers
# =============================================================================
def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Both vectors are unit-length → cosine == dot product
    return float(np.dot(a, b))

def peak_text_query_vec(emb: Siglip2, peak_name: str) -> List[float]:
    prompts = [
        f"a natural photo of the mountain peak {peak_name} in the Himalayas, Nepal",
        f"{peak_name} landmark peak in the Khumbu region, alpine landscape",
        f"{peak_name} mountain summit, snow, rocky ridgeline",
    ]
    proto = np.mean([emb.text_vec(p) for p in prompts], axis=0)
    anti = emb.text_vec("painting, illustration, poster, map, logo")  # downweight non-photos
    q = l2norm(proto - 0.25 * anti)
    return q.astype("float32").tolist()


# =============================================================================
# Peak identification (image -> peaks_catalog.text_embed)
# =============================================================================
def predict_peaks(
        es: Elasticsearch,
        image_vec: List[float],
        *,
        index: str = "peaks_catalog",
        k: int = 3,
        num_candidates: int = 500,
) -> List[Dict[str, Any]]:
    """
    Return the top-K peak candidates. Also include cosine & confidence by fetching
    the peak's text_embed and comparing it to the image_vec.
    """
    body = {
        "knn": {
            "field": "text_embed",
            "query_vector": image_vec,
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        "_source": ["id", "names", "latlon", "text_embed"],  # fetch vector to compute cosine/confidence
    }
    resp = es.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)

    img = np.asarray(image_vec, dtype=np.float32)
    out = []
    for h in hits:
        src = h.get("_source", {})
        name = (src.get("names") or [src.get("id")])[0]
        score = float(h.get("_score", 0.0))
        vec = np.asarray(src.get("text_embed", []), dtype=np.float32)
        cos = float(cosine(img, vec)) if vec.size else None
        conf = ((cos + 1.0) / 2.0) if cos is not None else None
        out.append({
            "id": src.get("id"),
            "name": name,
            "score": score,          # ES kNN score
            "cosine": cos,           # [-1, 1]
            "confidence": conf,      # [0, 1] approx
            "latlon": src.get("latlon"),
        })
    return out


# =============================================================================
# Neighbor search (photos.clip_image) with optional term filter
# =============================================================================
def similar_photos(
        es: Elasticsearch,
        query_vec: List[float],
        *,
        photos_index: str = "photos",
        k: int = 30,
        num_candidates: int = 2000,
        filter_peak: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch similar photos. Include the photo vector to compute cosine/confidence
    against the chosen query_vec (TEXT or IMAGE).
    """
    body: Dict[str, Any] = {
        "knn": {
            "field": "clip_image",
            "query_vector": query_vec,
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        "_source": ["path", "gps", "shot_time", "predicted_peaks", "clip_image"],  # include vector
    }
    if filter_peak:
        body["query"] = {"bool": {"filter": [{"term": {"predicted_peaks": filter_peak}}]}}

    resp = es.search(index=photos_index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)
    return [{**h.get("_source", {}), "_score": float(h.get("_score", 0.0))} for h in hits]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Identify the peak in a photo and fetch similar photos from your library."
    )
    ap.add_argument("--image", required=True, help="Path to an image file (jpg/png/heic)")
    ap.add_argument("--peaks-index", default="peaks_catalog", help="Index with peak TEXT embeddings")
    ap.add_argument("--photos-index", default="photos", help="Index with photo IMAGE embeddings")
    ap.add_argument("--topk", type=int, default=3, help="How many peak guesses to print")
    ap.add_argument("--neighbors", type=int, default=30, help="How many similar photos to fetch")
    ap.add_argument("--num-candidates-photos", type=int, default=2000, help="HNSW candidate pool for photos kNN")
    ap.add_argument("--num-candidates-peaks", type=int, default=500, help="HNSW candidate pool for peaks kNN")
    ap.add_argument("--no-filter", action="store_true",
                    help="Do NOT restrict neighbors to best peak via term filter on predicted_peaks.")
    ap.add_argument("--use-image-query", action="store_true",
                    help="Use uploaded IMAGE vector for neighbor search (default uses TEXT vector of best peak).")
    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    es = es_client()
    emb = Siglip2()

    # ---- Load & embed the uploaded image
    try:
        with Image.open(args.image) as im:
            ivec = emb.image_vec(im.convert("RGB")).astype("float32")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        raise SystemExit(f"Failed to open/embed image '{args.image}': {e}")
    ivec_list = ivec.tolist()

    # ---- Identify: image -> peak names
    top_peaks = predict_peaks(
        es,
        ivec_list,
        index=args.peaks_index,
        k=args.topk,
        num_candidates=args.num_candidates_peaks,
    )
    if not top_peaks:
        print("No peak prediction found.")
        return

    print("Top peak guesses:")
    for t in top_peaks:
        conf_str = f"{t['confidence']*100:5.1f}%" if t.get("confidence") is not None else " n/a "
        cos_str  = f"{t['cosine']:.4f}" if t.get("cosine") is not None else " n/a "
        print(f"  {t['name']} (id={t['id']}) | knn_score={t['score']:.4f} | cosine={cos_str} | confidence≈ {conf_str}")

    # ---- Choose neighbor query: TEXT(best) or IMAGE(uploaded)
    best_peak = top_peaks[0]["name"]
    if args.use_image_query:
        qvec = ivec_list
        print("\nUsing IMAGE vector for neighbors.")
    else:
        qvec = peak_text_query_vec(emb, best_peak)
        print(f"\nUsing TEXT vector for neighbors (peak='{best_peak}').")
    qvec_np = np.asarray(qvec, dtype=np.float32)

    # ---- Neighbor search (optionally filter by predicted_peaks)
    peak_filter = None if args.no_filter else best_peak
    sims = similar_photos(
        es,
        qvec,
        photos_index=args.photos_index,
        k=args.neighbors,
        num_candidates=args.num_candidates_photos,
        filter_peak=peak_filter,
    )

    if not sims and not args.no_filter:
        print("No results with filter; retrying without filter…")
        sims = similar_photos(
            es,
            qvec,
            photos_index=args.photos_index,
            k=args.neighbors,
            num_candidates=args.num_candidates_photos,
            filter_peak=None,
        )

    # ---- Pretty print neighbors with knn_score + cosine + confidence
    print(f"\nSimilar library photos (top {min(len(sims), args.neighbors)}):")
    for s in sims[:50]:
        path = s.get("path", "<no path>")
        preds = s.get("predicted_peaks")
        gps = s.get("gps")
        ts = s.get("shot_time")
        knn_score = float(s.get("_score", 0.0))

        cos = conf = None
        vec = s.get("clip_image")
        if isinstance(vec, list) and vec:
            vec_np = np.asarray(vec, dtype=np.float32)
            cos = cosine(qvec_np, vec_np)     # [-1, 1]
            conf = (cos + 1.0) / 2.0          # [0, 1]

        cos_str  = f"{cos:.4f}" if cos is not None else " n/a "
        conf_str = f"{conf*100:5.1f}%" if conf is not None else " n/a "

        print(f"{path}")
        print(f"  knn_score={knn_score:.4f} | cosine={cos_str} | confidence≈ {conf_str}")
        if preds: print(f"  predicted_peaks: {', '.join(preds)}")
        if gps:   print(f"  gps: ({gps.get('lat'):.5f}, {gps.get('lon'):.5f})")
        if ts:    print(f"  shot_time: {ts}")
        print()


if __name__ == "__main__":
    main()
