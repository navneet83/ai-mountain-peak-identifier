#!/usr/bin/env python3
# scripts/query_by_peak.py
"""
Find photos matching a given *peak name* using SigLIP-2 **text→image** kNN in Elasticsearch.

What this does (high level)
---------------------------
1) Builds a **text query vector** for the peak by encoding a tiny prompt-ensemble
   with the SigLIP-2 TEXT encoder (plus a light "anti-concept" to downweight maps/logos).
2) Runs **kNN** against the `photos` index's `clip_image` (IMAGE) vectors.
3) (Optional) Applies a **geo-fence** filter if --lat/--lon/--distance is provided
4) Prints the top matches (path, GPS, shot_time).

Auth & Endpoint (env)
---------------------
- ES_URL or ES_CLOUD_ID
- ES_API_KEY_B64  or
- ES_API_KEY_ID + ES_API_KEY

Example
-------
python scripts/query_by_peak.py --peak "Ama Dablam" --k 30 --num-candidates 4000 \
  --lat 27.93 --lon 86.90 --distance 120km
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import sys
from typing import List

import numpy as np
from elasticsearch import Elasticsearch

# --- repo path shim so we can `from ai_mpi.embeddings import Siglip2` ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ai_mpi.embeddings import Siglip2  # SigLIP-2 wrapper (returns L2-normalized vectors)


# -----------------------------------------------------------------------------
# Elasticsearch client (API key)
# -----------------------------------------------------------------------------
def es_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from env vars.

    Supported combos:
      - ES_CLOUD_ID + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)
      - ES_URL      + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)
    """
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
    # self-hosted URL path
    if api_key_b64:
        return Elasticsearch(url, api_key=api_key_b64)
    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    raise SystemExit("Set ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY (and ES_URL/ES_CLOUD_ID).")


# -----------------------------------------------------------------------------
# Embedding helpers
# -----------------------------------------------------------------------------
def l2norm(v: np.ndarray) -> np.ndarray:
    """Safe L2 normalization (cosine ≡ dot product when vectors are unit-norm)."""
    return v / (np.linalg.norm(v) + 1e-12)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are already unit-length

def build_query_vec(emb: Siglip2, peak_name: str) -> List[float]:
    """
    Turn a *name* into a robust text embedding for retrieval by:
      - encoding a few short, factual prompts about the mountain
      - averaging them (prompt ensemble)
      - subtracting a small "anti-concept" vector to push away maps/posters/logos
      - L2-normalizing the result
    """
    prompts = [
        f"a natural photo of the mountain peak {peak_name} in the Himalayas, Nepal",
        f"{peak_name} landmark peak in the Khumbu region, alpine landscape",
        f"{peak_name} mountain summit, snow, rocky ridgeline",
    ]
    proto = np.mean([emb.text_vec(p) for p in prompts], axis=0)

    # Downweight non-photo interpretations (logos, posters, maps, illustrations)
    neg = emb.text_vec("painting, illustration, poster, map, logo")
    vec = l2norm(proto - 0.25 * neg)

    return vec.astype("float32").tolist()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Text→image kNN over photos to find matches for a peak name.")
    ap.add_argument("--peak", default="Ama Dablam", help='Peak name (e.g., "Ama Dablam")')
    ap.add_argument("--index", default="photos", help='Target index (default: "photos")')
    ap.add_argument("--k", type=int, default=200, help="Number of results (also k in kNN)")
    ap.add_argument("--num-candidates", type=int, default=2000, help="HNSW candidate pool for kNN")
    ap.add_argument("--lat", type=float, help="Geo filter center latitude (optional)")
    ap.add_argument("--lon", type=float, help="Geo filter center longitude (optional)")
    ap.add_argument("--distance", default="120km", help='Geo distance like "50km" / "120km"')
    ap.add_argument("--source", default="path,predicted_peaks,gps,shot_time",
                    help="Comma-separated fields to return (default shown)")
    args = ap.parse_args()

    es = es_client()
    emb = Siglip2()

    # 1) Build the text query vector from the peak name
    qvec = build_query_vec(emb, args.peak)
    qvec_np = np.asarray(qvec, dtype=np.float32)

    # include clip_image in _source so we can compute cosine
    default_fields = ["path", "predicted_peaks", "gps", "shot_time", "clip_image"]

    # 2) Assemble kNN body (optional geo filter)
    src_fields = [f.strip() for f in args.source.split(",") if f.strip()] if args.source else default_fields
    if "clip_image" not in src_fields:
        src_fields.append("clip_image")

    body = {
        "knn": {
            "field": "clip_image",
            "query_vector": qvec,
            "k": int(args.k),
            "num_candidates": int(args.num_candidates),
        },
        "_source": src_fields or True,
    }
    if args.lat is not None and args.lon is not None:
        body["query"] = {
            "bool": {
                "filter": [
                    {"geo_distance": {"distance": args.distance, "gps": {"lat": args.lat, "lon": args.lon}}}
                ]
            }
        }

    # 3) Run the search
    try:
        resp = es.search(index=args.index, body=body)
    except Exception as e:
        raise SystemExit(f"Elasticsearch search failed: {e}")

    hits = resp.get("hits", {}).get("hits", [])
    print(f"\nFound {len(hits)} results for peak='{args.peak}' (index='{args.index}')")
    if args.lat is not None and args.lon is not None:
        print(f"Geo filter: {args.distance} around ({args.lat}, {args.lon})")
    print("-" * 80)

    for h in hits[:50]:
        s = h.get("_source", {})
        path = s.get("path")
        preds = s.get("predicted_peaks") or []
        gps   = s.get("gps")
        ts    = s.get("shot_time")
        es_score = float(h.get("_score", 0.0))

        cos_sim = None
        conf = None
        vec = s.get("clip_image")
        if isinstance(vec, list) and vec:
            vec_np = np.asarray(vec, dtype=np.float32)
            cos_sim = cosine(qvec_np, vec_np)                  # [-1, 1]
            conf    = (cos_sim + 1.0) / 2.0                    # [0, 1]

        print(path or "<no path>")
        if preds:
            print(f"  predicted_peaks: {', '.join(preds)}")
        if gps:
            print(f"  gps: ({gps.get('lat'):.5f}, {gps.get('lon'):.5f})")
        if ts:
            print(f"  shot_time: {ts}")
        # show both the ES kNN _score and our cosine/“confidence”
        if cos_sim is not None:
            print(f"  knn_score: {es_score:.4f} | cosine_sim: {cos_sim:.4f} | confidence≈ {conf*100:5.1f}%")
        else:
            print(f"  knn_score: {es_score:.4f}")
        print()
    if not hits:
        print("No matches. Double-check the index name, embeddings, or try a different peak.")


if __name__ == "__main__":
    main()
