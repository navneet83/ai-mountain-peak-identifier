#!/usr/bin/env python3
"""
Create (or recreate) the two Elasticsearch indices:

1) peaks_catalog
   - One document per mountain peak (Everest, Ama Dablam, etc.)
   - Stores: names/aliases, lat/lon, and a text embedding ("text_embed")
     created with the SigLIP-2 TEXT encoder (dims usually 768)

2) photos
   - One document per photo
   - Stores: relative path, image embedding ("clip_image") created with the
     SigLIP-2 IMAGE encoder (same dims as text), optional GPS + shot_time,
     and a small array of predicted peak names ("predicted_peaks")

USAGE
-----
# Create indices if missing (no destructive changes)
python scripts/create_indices.py

# Force re-create (DANGEROUS: deletes + creates fresh)
python scripts/create_indices.py --recreate

# Customize names / dims / HNSW params
python scripts/create_indices.py --photos-index photos --peaks-index peaks_catalog --dims 768 --hnsw-m 16 --hnsw-ef 128
"""

from __future__ import annotations
import os
import argparse
from pprint import pformat
from elasticsearch import Elasticsearch

# --------------------------------------------------------------------------------------
# 1) Client construction (API key)
# --------------------------------------------------------------------------------------
def es_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from environment variables.

    Supported env vars:
      - ES_CLOUD_ID:    for Elastic Cloud (preferred for cloud)
      - ES_URL:         for self-hosted cluster (default http://localhost:9200)
      - ES_API_KEY_B64: base64 "id:key" form (works for both cloud and self-hosted)
      - or ES_API_KEY_ID + ES_API_KEY as a pair

    Priority is:
      Cloud (ES_CLOUD_ID) -> API key
      Else self-hosted ES_URL -> API key

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
    else:
        if api_key_b64:
            return Elasticsearch(url, api_key=api_key_b64)
        if api_key_id and api_key:
            return Elasticsearch(url, api_key=(api_key_id, api_key))
        return Elasticsearch(url)  # last resort: unauthenticated local dev


# --------------------------------------------------------------------------------------
# 2) Mappings for both indices
# --------------------------------------------------------------------------------------
def peaks_catalog_mapping(dims: int, hnsw_m: int, hnsw_ef: int) -> dict:
    """
    Mapping for the 'peaks_catalog' index.

    Fields:
      id        : keyword    - stable identifier (slug)
      names     : keyword[]  - primary name + aliases (exact terms)
      latlon    : geo_point  - peak location (for UI / sanity checks)
      text_embed: dense_vector - SigLIP-2 TEXT embedding (dims must match your model)

    Vector options:
      - index=true, similarity=cosine, index_options=HNSW with m/ef_construction
    """
    return {
        "mappings": {
            "properties": {
                "id":       {"type": "keyword"},
                "names":    {"type": "keyword"},
                "latlon":   {"type": "geo_point"},
                "text_embed": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "hnsw", "m": hnsw_m, "ef_construction": hnsw_ef}
                }
            }
        }
    }


def photos_mapping(dims: int, hnsw_m: int, hnsw_ef: int) -> dict:
    """
    Mapping for the 'photos' index.

    Fields:
      path           : keyword     - relative path to the image (UI uses this to resolve a file)
      clip_image     : dense_vector - SigLIP-2 IMAGE embedding (same dims as text)
      predicted_peaks: keyword[]   - top-k peak guesses (stored at index time)
      gps            : geo_point   - parsed EXIF GPS if available
      shot_time      : date        - parsed EXIF date if available

    Notes:
      - 'predicted_peaks' is a keyword array so we can do exact-term matches
      - todo for later version: add captions and extend with a text field and a standard retriever.
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
                    "index_options": {"type": "hnsw", "m": hnsw_m, "ef_construction": hnsw_ef}
                },
                "predicted_peaks": {"type": "keyword"},
                "gps": {"type": "geo_point"},
                # We use a permissive date format so ISO8601 strings from EXIF parse cleanly.
                "shot_time": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            }
        }
    }


# --------------------------------------------------------------------------------------
# 3) Create / recreate helpers
# --------------------------------------------------------------------------------------
def create_index_if_missing(es: Elasticsearch, name: str, body: dict) -> None:
    """
    Create an index if it doesn't exist. If it exists, we print the mapping and keep going.
    """
    if es.indices.exists(index=name):
        print(f"[ok] index exists: {name}")
        existing = es.indices.get_mapping(index=name)
        # Show only the mapping for this index (clusters can return multi-index dicts)
        if isinstance(existing, dict) and name in existing:
            print(f"      existing mapping (short): {list(existing[name].get('mappings', {}).get('properties', {}).keys())}")
        return
    es.indices.create(index=name, body=body)
    print(f"[new] created index: {name}")


def recreate_index(es: Elasticsearch, name: str, body: dict) -> None:
    """
    Caution: Delete and create the index. This wipes data.
    """
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
        print(f"[drop] deleted index: {name}")
    es.indices.create(index=name, body=body)
    print(f"[new] created index: {name}")


# --------------------------------------------------------------------------------------
# 4) CLI and main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create (or re-create) Elasticsearch indices for the AI Mountain Peak Identifier demo."
    )
    p.add_argument("--peaks-index", default="peaks_catalog", help="Index name for the peaks catalog")
    p.add_argument("--photos-index", default="photos", help="Index name for photos")
    p.add_argument("--dims", type=int, default=768,
                   help="Vector dimension for SigLIP-2 embeddings (google/siglip-so400m-patch14-384 = 768)")
    p.add_argument("--hnsw-m", type=int, default=16, help="HNSW 'm' graph degree (good default: 16)")
    p.add_argument("--hnsw-ef", type=int, default=128, help="HNSW ef_construction (good default: 128)")
    p.add_argument("--recreate", action="store_true",
                   help="Delete and recreate the indices (DANGEROUS: wipes data)")
    return p.parse_args()


def main():
    args = parse_args()
    es = es_client()

    # Build mappings with your chosen dims/HNSW params.
    # IMPORTANT: 'dims' must match the output size of your SigLIP-2 model.
    # For google/siglip-so400m-patch14-384 the embedding size is 768.
    peaks_body  = peaks_catalog_mapping(args.dims, args.hnsw_m, args.hnsw_ef)
    photos_body = photos_mapping(args.dims, args.hnsw_m, args.hnsw_ef)

    try:
      info = es.info()
      cluster = info.get("cluster_name")
      version = (info.get("version") or {}).get("number")
      print(f"Cluster: {cluster} (v{version})")
    except Exception:
      print("Cluster: <unavailable>")
    print(f"Peaks index  : {args.peaks_index}")
    print(f"Photos index : {args.photos_index}")
    print(f"Vector dims  : {args.dims} | HNSW m={args.hnsw_m}, ef_construction={args.hnsw_ef}")

    if args.recreate:
        recreate_index(es, args.peaks_index,  peaks_body)
        recreate_index(es, args.photos_index, photos_body)
    else:
        create_index_if_missing(es, args.peaks_index,  peaks_body)
        create_index_if_missing(es, args.photos_index, photos_body)

    # Optional: print the full mapping for readers (uncomment if you want the verbosity)
    # print(pformat(es.indices.get_mapping(index=args.peaks_index)))
    # print(pformat(es.indices.get_mapping(index=args.photos_index)))

    print("[done] indices are ready.")


if __name__ == "__main__":
    main()

