#!/usr/bin/env python3
"""
Embed + index photos, and (optionally) build blended peak vectors for the peaks catalog.

This script powers two workflows:

A) Re/Index the *peaks catalog* with a blended vector per peak
   - Reads `data/peaks.yaml`
   - Optionally loads 1–3 reference images per peak from `data/peaks/<id>/`
   - Blends SigLIP-2 TEXT (prompts) with IMAGE vectors (refs) → `text_embed`
   - Upserts documents into the `peaks_catalog` index

B) Index your *photo library*
   - Walks `data/images/**` (JPG/PNG/HEIC)
   - Computes SigLIP-2 IMAGE embeddings → `clip_image`
   - Extracts EXIF `gps` and `shot_time` when present
   - Predicts top-k `predicted_peaks` by kNN against `peaks_catalog.text_embed`
   - Indexes one doc per photo into the `photos` index

Why blended vectors?
--------------------
Peak names alone (text) can drift; a few reference photos anchor the concept to how
the mountain actually *looks*. We average TEXT+IMAGE vectors (then L2-normalize).

Auth (API key)
--------------
Use one of:
  ES_CLOUD_ID + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)
  ES_URL      + (ES_API_KEY_B64 | ES_API_KEY_ID + ES_API_KEY)

Examples
--------
# 1) Re-index peaks catalog (blended vectors)
python scripts/embed_and_index_photos.py \
  --index-peaks \
  --peaks-yaml data/peaks.yaml \
  --peaks-images-root data/peaks \
  --blend-alpha-text 0.55 --blend-max-images 3

# 2) Index photos with vectors + GPS/time + predicted_peaks

python scripts/embed_and_index_photos.py \
  --index-photos \
  --images data/images \
  --photos-index photos \
  --peaks-index  peaks_catalog \
  --topk-predicted 5 \
  --batch-size 200
"""

from __future__ import annotations

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from elasticsearch import Elasticsearch, helpers
from PIL import Image, ExifTags, UnidentifiedImageError

# --- Optional HEIC support (so .HEIC photos open like JPG/PNG) ---
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

# --- Repo-local import so `from ai_mpi.embeddings import Siglip2` works ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_mpi.embeddings import Siglip2  # wraps SigLIP-2 image/text encoders (returns L2-normalized vectors)


# =============================================================================
# Elasticsearch client (API key auth)
# =============================================================================
def es_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from environment variables.

    Supported env vars:
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
    raise SystemExit("Set ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY (and ES_URL).")


# =============================================================================
# Utility: L2 normalization (cosine ≡ dot when L2-normalized)
# =============================================================================
def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


# =============================================================================
# EXIF helpers (robust GPS + shot_time parsing)
# =============================================================================
_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPSTAGS = ExifTags.GPSTAGS


def _ratio_to_float(x) -> float:
    """PIL can return a Rational or a (num, den) tuple; normalize to float."""
    try:
        return float(x.numerator) / float(x.denominator)
    except Exception:
        try:
            return float(x[0]) / float(x[1])
        except Exception:
            return float(x)


def _dms_to_deg(d, m, s, ref) -> float:
    deg = _ratio_to_float(d) + _ratio_to_float(m) / 60.0 + _ratio_to_float(s) / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if (ref or "").upper() in ("S", "W"):
        deg = -deg
    return deg


def get_exif(image_path: str) -> Dict[str, Any]:
    """Return EXIF dict (safe on errors)."""
    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            return dict(exif) if exif else {}
    except Exception:
        return {}


def get_gps_from_exif(exif: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Parse EXIF GPSInfo into {'lat': float, 'lon': float} or None.

    GPSInfo key is often 34853. Some images store an *int* here (odd but real).
    Guard carefully so we never crash on malformed EXIF.
    """
    gps_ifd = exif.get(_TAGS.get("GPSInfo", 34853))
    if not gps_ifd or isinstance(gps_ifd, int):
        return None

    try:
        gps = {_GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat_vals = gps.get("GPSLatitude")
        lat_ref = gps.get("GPSLatitudeRef", "N")
        lon_vals = gps.get("GPSLongitude")
        lon_ref = gps.get("GPSLongitudeRef", "E")
        if not lat_vals or not lon_vals:
            return None
        lat = _dms_to_deg(lat_vals[0], lat_vals[1], lat_vals[2], lat_ref)
        lon = _dms_to_deg(lon_vals[0], lon_vals[1], lon_vals[2], lon_ref)
        return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        return None


def get_shot_time_from_exif(exif: Dict[str, Any]) -> Optional[str]:
    """
    Return ISO8601 (YYYY-MM-DDTHH:MM:SS) from DateTimeOriginal or DateTime, if present.
    """
    dto_key = _TAGS.get("DateTimeOriginal", 36867)
    dt_key = _TAGS.get("DateTime", 306)
    val = exif.get(dto_key) or exif.get(dt_key)
    if not val:
        return None
    try:
        if isinstance(val, bytes):
            val = val.decode(errors="ignore")
        # EXIF format "YYYY:MM:DD HH:MM:SS" → "YYYY-MM-DDTHH:MM:SS"
        return val.strip().replace(":", "-", 2).replace(" ", "T")
    except Exception:
        return None


def get_gps(image_path: str) -> Optional[Dict[str, float]]:
    return get_gps_from_exif(get_exif(image_path))


def get_shot_time(image_path: str) -> Optional[str]:
    return get_shot_time_from_exif(get_exif(image_path))


# =============================================================================
# File iteration + IDs
# =============================================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".HEIC", ".JPG", ".JPEG", ".PNG", ".WEBP")


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image files recursively under `root`."""
    for p in sorted(root.rglob("*")):
        if p.suffix in IMG_EXTS and p.is_file():
            yield p


def relpath_within(base: Path, path: Path) -> str:
    """Relative path (portable slashes) so the UI can resolve thumbnails locally."""
    base = base.resolve()
    try:
        rel = path.resolve().relative_to(base)
    except Exception:
        rel = path.name
    return str(rel).replace("\\", "/")


def id_for_path(rel_path: str) -> str:
    """Stable doc id from relative path."""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


# =============================================================================
# Peak text prompts + blended vectors (TEXT + reference IMAGES)
# =============================================================================
def peak_text_prompts(primary_name: str) -> List[str]:
    """Simple, factual prompts; avoid poetic language to keep the embedding tight."""
    return [
        f"{primary_name} mountain peak in the Himalayas, Nepal",
        f"{primary_name} landmark peak in the Khumbu region, alpine landscape",
        f"{primary_name} mountain summit, snow and rocky ridgelines",
    ]


def slugify(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append(ch if ch.isalnum() else "_")
    return "_".join(filter(None, "".join(out).split("_")))


def list_ref_images(peak_dir: Path) -> List[Path]:
    """All reference images found under a given directory (JPG/PNG/HEIC)."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.heic", "*.HEIC", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")
    paths: List[Path] = []
    for e in exts:
        paths.extend(sorted(peak_dir.glob(e)))
    return paths


def embed_text_blend(emb: Siglip2, names: List[str]) -> np.ndarray:
    """
    Average a few text prompts for the primary name, subtract a light "anti-concept"
    to downweight maps/posters/icons, then L2-normalize.
    """
    main = names[0] if names else ""
    tv = np.mean([emb.text_vec(p) for p in peak_text_prompts(main)], axis=0)
    tv = l2norm(tv - 0.25 * emb.text_vec("painting, illustration, poster, map, logo"))
    return tv.astype("float32")


def embed_image_mean(emb: Siglip2, paths: List[Path]) -> Optional[np.ndarray]:
    """Average up to N reference image vectors (skip unreadable files)."""
    vecs = []
    for p in paths:
        try:
            with Image.open(p) as im:
                vecs.append(emb.image_vec(im.convert("RGB")))
        except Exception:
            continue
    if not vecs:
        return None
    return l2norm(np.mean(np.stack(vecs, 0), 0).astype("float32"))


def compute_blended_peak_vec(
        emb: Siglip2,
        names: List[str],
        peak_id: str,
        peaks_images_root: str,
        alpha_text: float = 0.5,
        max_images: int = 3,
) -> Tuple[np.ndarray, int, int, List[str]]:
    """
    Build blended vector for a single peak.

    Returns:
      vec           : np.ndarray (L2-normalized)
      found_count   : number of reference images discovered
      used_count    : number of references used (<= max_images)
      used_filenames: list of filenames used (for logging)
    """
    # 1) TEXT vector
    tv = embed_text_blend(emb, names)

    # 2) IMAGE refs: prefer folder by id; fallback to slug of the primary name
    root = Path(peaks_images_root)
    candidates = [root / peak_id]
    if names:
        candidates.append(root / slugify(names[0]))

    all_refs: List[Path] = []
    for c in candidates:
        if c.exists() and c.is_dir():
            all_refs = list_ref_images(c)
            if all_refs:
                break

    found = len(all_refs)
    used_list = all_refs[:max_images] if (max_images and found > max_images) else all_refs
    used = len(used_list)

    img_v = embed_image_mean(emb, used_list) if used_list else None

    # 3) Blend TEXT and IMAGE vectors, clamp alpha to [0,1]
    a = max(0.0, min(1.0, float(alpha_text)))
    vec = l2norm(tv if img_v is None else (a * tv + (1.0 - a) * img_v)).astype("float32")
    return vec, found, used, [p.name for p in used_list]


def index_peaks_with_blend(
        es: Elasticsearch,
        emb: Siglip2,
        peaks_yaml: str,
        peaks_images_root: str,
        alpha_text: float,
        max_images: int,
        index_name: str = "peaks_catalog",
        refresh: str = "wait_for",
) -> None:
    """
    Read peaks from YAML, compute blended vectors, and index into `index_name`.
    Accepts YAML shaped as:
      - list of {id, names, latlon, ...}
      - or { "peaks": [ ... ] }
    """
    with open(peaks_yaml, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and isinstance(data.get("peaks"), list):
        peak_list = data["peaks"]
    elif isinstance(data, list):
        peak_list = data
    else:
        raise SystemExit(f"Invalid YAML structure in {peaks_yaml} (expected list or dict with 'peaks').")

    for item in peak_list:
        pid = item.get("id") or slugify((item.get("names", [None])[0] or "peak"))
        names = item.get("names") or [item.get("name") or pid]
        latlon = item.get("latlon")
        if not latlon and ("lat" in item and "lon" in item):
            latlon = {"lat": item["lat"], "lon": item["lon"]}

        vec, found_cnt, used_cnt, used_files = compute_blended_peak_vec(
            emb=emb,
            names=names,
            peak_id=pid,
            peaks_images_root=peaks_images_root,
            alpha_text=alpha_text,
            max_images=max_images,
        )

        es.index(
            index=index_name,
            id=pid,
            document={"id": pid, "names": names, "latlon": latlon, "text_embed": vec.tolist()},
            refresh=refresh,
        )

        msg = (
            f"[peaks_catalog] upserted id={pid} names={names[:1]} "
            f"α_text={alpha_text:.2f} dim={len(vec)} refs_found={found_cnt} refs_used={used_cnt}"
        )
        print(msg)
        if os.getenv("LOG_REF_FILES", "0").lower() in ("1", "true"):
            if used_files:
                print("  used files:", ", ".join(used_files))


# =============================================================================
# Predict peaks for a photo (top-k) using ES kNN on `peaks_catalog.text_embed`
# =============================================================================
def predict_peaks(
        es: Elasticsearch,
        image_vec: List[float],
        peaks_index: str = "peaks_catalog",
        k: int = 5,
        num_candidates: int = 1000,
) -> List[str]:
    body = {
        "knn": {
            "field": "text_embed",
            "query_vector": image_vec,
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        "_source": ["id", "names"],
    }
    resp = es.search(index=peaks_index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    # Use the primary display name (names[0]) for each hit
    names = []
    for h in hits:
        src = h.get("_source", {})
        nm = (src.get("names") or [src.get("id")])[0]
        if nm:
            names.append(nm)
    return names


# =============================================================================
# Photo indexing (bulk)
# =============================================================================
def bulk_index_photos(
        es: Elasticsearch,
        images_root: str,
        photos_index: str = "photos",
        peaks_index: str = "peaks_catalog",
        topk_predicted: int = 5,
        batch_size: int = 200,
        refresh: str = "false",
) -> None:
    """Walk a folder of images, embed + enrich, and bulk index to Elasticsearch."""
    root = Path(images_root)
    if not root.exists():
        raise SystemExit(f"Images root not found: {images_root}")

    emb = Siglip2()
    batch: List[Dict[str, Any]] = []
    n_indexed = 0

    for p in iter_images(root):
        rel = relpath_within(root, p)
        _id = id_for_path(rel)

        # 1) Image embedding (and reuse it for predicted_peaks)
        try:
            with Image.open(p) as im:
                ivec = emb.image_vec(im.convert("RGB")).astype("float32")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[skip] {rel} — cannot embed: {e}")
            continue

        # 2) Predict top-k peak names
        try:
            top_names = predict_peaks(es, ivec.tolist(), peaks_index=peaks_index, k=topk_predicted)
        except Exception as e:
            print(f"[warn] predict_peaks failed for {rel}: {e}")
            top_names = []

        # 3) EXIF enrichment (safe)
        gps = get_gps(str(p))
        shot = get_shot_time(str(p))

        # 4) Build doc and stage for bulk
        doc = {"path": rel, "clip_image": ivec.tolist(), "predicted_peaks": top_names}
        if gps:
            doc["gps"] = gps
        if shot:
            doc["shot_time"] = shot

        batch.append(
            {"_op_type": "index", "_index": photos_index, "_id": _id, "_source": doc}
        )

        # 5) Periodic flush
        if len(batch) >= batch_size:
            helpers.bulk(es, batch, refresh=refresh)
            n_indexed += len(batch)
            print(f"[photos] indexed {n_indexed} (last: {rel})")
            batch.clear()

    # Final flush
    if batch:
        helpers.bulk(es, batch, refresh=refresh)
        n_indexed += len(batch)
        print(f"[photos] indexed {n_indexed} total.")

    print("[done] photos indexing")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Embed and index photos; optionally blend+index the peaks_catalog."
    )
    # Peaks (blend) options
    ap.add_argument(
        "--index-peaks", action="store_true", help="(Re)index peaks_catalog with blended vectors"
    )
    ap.add_argument("--peaks-yaml", default="data/peaks.yaml", help="YAML with peak entries")
    ap.add_argument(
        "--peaks-images-root",
        default="data/peaks",
        help="Folder containing reference images under subfolders named by peak id (or primary name slug)",
    )
    ap.add_argument("--peaks-index", default="peaks_catalog", help="Index name for peaks catalog")
    ap.add_argument(
        "--blend-alpha-text",
        type=float,
        default=0.5,
        help="Weight for TEXT in the blend (0..1); 1.0=text-only, 0.0=image-only",
    )
    ap.add_argument(
        "--blend-max-images",
        type=int,
        default=3,
        help="Max reference images to average per peak (0 = text-only)",
    )

    # Photos options
    ap.add_argument(
        "--index-photos", action="store_true", help="Index photos with vectors + metadata + predicted_peaks"
    )
    ap.add_argument("--images", default="data/images", help="Root folder of your photo library")
    ap.add_argument("--photos-index", default="photos", help="Index name for photos")
    ap.add_argument(
        "--topk-predicted", type=int, default=5, help="Top-k peak names to store in predicted_peaks"
    )
    ap.add_argument("--batch-size", type=int, default=200, help="Bulk indexing batch size")
    ap.add_argument(
        "--refresh", default="false", help="Elasticsearch refresh policy (false|true|wait_for)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    es = es_client()

    # Build one model instance for the "blend peaks" path
    emb_for_peaks = Siglip2()

    if args.index_peaks:
        index_peaks_with_blend(
            es=es,
            emb=emb_for_peaks,
            peaks_yaml=args.peaks_yaml,
            peaks_images_root=args.peaks_images_root,
            alpha_text=args.blend_alpha_text,
            max_images=args.blend_max_images,
            index_name=args.peaks_index,
            refresh="wait_for",
        )

    if args.index_photos:
        bulk_index_photos(
            es=es,
            images_root=args.images,
            photos_index=args.photos_index,
            peaks_index=args.peaks_index,
            topk_predicted=args.topk_predicted,
            batch_size=args.batch_size,
            refresh=args.refresh,
        )

    if not args.index_peaks and not args.index_photos:
        print("No action specified. Use --index-peaks and/or --index-photos.")


if __name__ == "__main__":
    main()
