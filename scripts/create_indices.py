#!/usr/bin/env python3
"""
Create (or re-create) the two Elasticsearch indices for the demo.

Indices
-------
1) peaks_catalog
   - One document per mountain peak (Everest, Ama Dablam, etc.)
   - Fields:
       id (keyword)           : stable identifier / slug
       names (keyword[])      : primary name + aliases (exact matching)
       latlon (geo_point)     : peak location (useful in UI or sanity checks)
       text_embed (dense_vec) : SigLIP-2 TEXT embedding (dims must match model)

2) photos
   - One document per photo in library (photos that I clicked during my trek)
   - Fields:
       path (keyword)          : relative path used by the UI to open a thumbnail
       clip_image (dense_vec)  : SigLIP-2 IMAGE embedding (same dims as text)
       predicted_peaks (kw[])  : top-k peak guesses assigned at index time
       gps (geo_point)         : parsed EXIF GPS if present
       shot_time (date)        : parsed EXIF DateTimeOriginal if present

Why dense_vector + HNSW?
------------------------
We want approximate nearest-neighbor (ANN) search over embeddings. The can be done
in Elasticsearch using `dense_vector` with `index=true` + HNSW index_options,
and a similarity (cosine here).

Usage
-----
# Create indices if missing (non-destructive)
python scripts/create_indices.py

# Force re-create (deletes & recreates both indices)
python scripts/create_indices.py --recreate

# Customize names / dims / HNSW params
python scripts/create_indices.py --photos-index photos --peaks-index peaks_catalog --dims 768 --hnsw-m 16 --hnsw-ef 128
"""

from __future__ import annotations
import os
import argparse
from elasticsearch import Elasticsearch


# -----------------------------------------------------------------------------
# Client construction (API key via env vars)
# -----------------------------------------------------------------------------
def es_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from environment variables.

    Supported (pick one auth path):
      - ES_CLOUD_ID + (ES_API_KEY_B64  |  ES_API_KEY_ID + ES_API_KEY)
      - ES_URL      + (ES_API_KEY_B64  |  ES_API_KEY_ID + ES_API_KEY)
      - ES_URL only (unauthenticated local dev)

    """
    cloud_id    = os.getenv("ES_CLOUD_ID")
    url         = os.getenv("ES_URL", "http://localhost:9200")
    api_key_b64 = os.getenv("ES_API_KEY_B64")
    api_key_id  = os.getenv("ES_API_KEY_ID")
    api_key     = os.getenv("ES_API_KEY")

    if cloud_id:
        if api_key_b64:
            return Elasticsearch(cloud_id=cloud_id, api_key=api_key_b64)
        if api_key_id and api_key:
            return Elasticsearch(cloud_id=cloud_id, api_key=(api_key_id, api_key))
        raise SystemExit("Elastic Cloud detected: set ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY.")
    # self-hosted URL path
    if api_key_b64:
        return Elasticsearch(url, api_key=api_key_b64)
    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    return Elasticsearch(url)  # local dev w/o auth


# -----------------------------------------------------------------------------
# Mappings
# -----------------------------------------------------------------------------
def peaks_catalog_mapping(dims: int, hnsw_m: int, hnsw_ef: int) -> dict:
    """
    Mapping for the 'peaks_catalog' index.

    Notes:
      * 'names' is keyword (exact terms) because we use it in filters / display.
      * 'text_embed' is searchable ANN vector space using cosine similarity.
    """
    return {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "names": {"type": "keyword"},
                "latlon": {"type": "geo_point"},
                "text_embed": {
                    "type": "dense_vector",
                    "dims": dims,                 # must match SigLIP-2 output size
                    "index": True,                # enable ANN
                    "similarity": "cosine",       # CLIP/SigLIP uses cosine typically
                    "index_options": {            # HNSW configuration
                        "type": "hnsw",
                        "m": hnsw_m,
                        "ef_construction": hnsw_ef
                    }
                }
            }
        }
    }


def photos_mapping(dims: int, hnsw_m: int, hnsw_ef: int) -> dict:
    """
    Mapping for the 'photos' index.

    Notes:
      * 'predicted_peaks' is keyword[] so we can do exact term matches and use it
        as a signal in fusion (RRF).
      * 'shot_time' uses a permissive date format so ISO8601 strings parse cleanly.
    """
    return {
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "clip_image": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": hnsw_m,
                        "ef_construction": hnsw_ef
                    }
                },
                "predicted_peaks": {"type": "keyword"},
                "gps": {"type": "geo_point"},
                "shot_time": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                }
            }
        }
    }


# -----------------------------------------------------------------------------
# Create / recreate helpers
# -----------------------------------------------------------------------------
def create_index_if_missing(es: Elasticsearch, name: str, body: dict) -> None:
    """Create an index if it doesn't exist; otherwise leave it as-is."""
    if es.indices.exists(index=name):
        print(f"[ok] index exists: {name}")
        return
    es.indices.create(index=name, body=body)
    print(f"[new] created index: {name}")


def recreate_index(es: Elasticsearch, name: str, body: dict) -> None:
    """Delete and re-create an index (DANGEROUS: wipes data)."""
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
        print(f"[drop] deleted index: {name}")
    es.indices.create(index=name, body=body)
    print(f"[new] created index: {name}")


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create (or re-create) Elasticsearch indices for the AI Mountain Peak Identifier demo."
    )
    p.add_argument("--peaks-index", default="peaks_catalog", help="Index name for the peaks catalog")
    p.add_argument("--photos-index", default="photos", help="Index name for the photos")
    p.add_argument("--dims", type=int, default=768,
                   help="Embedding dimension (SigLIP-2 so400m@384 outputs 768)")
    p.add_argument("--hnsw-m", type=int, default=16, help="HNSW 'm' (graph degree)")
    p.add_argument("--hnsw-ef", type=int, default=128, help="HNSW ef_construction (build-time)")
    p.add_argument("--recreate", action="store_true",
                   help="Delete and recreate both indices (DANGEROUS: wipes data)")
    return p.parse_args()


def main():
    args = parse_args()
    es = es_client()

    # Build mappings. IMPORTANT: --dims must match the model (SigLIP-2 so400m@384) output size.
    peaks_body  = peaks_catalog_mapping(args.dims, args.hnsw_m, args.hnsw_ef)
    photos_body = photos_mapping(args.dims, args.hnsw_m, args.hnsw_ef)

    # Friendly cluster banner (works across client versions)
    try:
        info = es.info()
        cluster = info.get("cluster_name")
        version = (info.get("version") or {}).get("number")
        print(f"Cluster: {cluster} (v{version})")
    except Exception:
        print("Cluster: <unavailable>")

    print(f"Peaks index : {args.peaks_index}")
    print(f"Photos index: {args.photos_index}")
    print(f"Vector dims : {args.dims} | HNSW m={args.hnsw_m}, ef_construction={args.hnsw_ef}")

    if args.recreate:
        recreate_index(es, args.peaks_index,  peaks_body)
        recreate_index(es, args.photos_index, photos_body)
    else:
        create_index_if_missing(es, args.peaks_index,  peaks_body)
        create_index_if_missing(es, args.photos_index, photos_body)

    print("[done] indices are ready.")


if __name__ == "__main__":
    main()

