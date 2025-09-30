#!/usr/bin/env python3
"""
Index photos into Elasticsearch with:
  - path (relative to BASE_IMAGE_DIR or --images)
  - clip_image (SigLIP-2 IMAGE embedding)
  - gps (geo_point) when EXIF has GPS
  - shot_time (date) when EXIF has DateTimeOriginal
  - predicted_peaks (keyword[]) = top-K names scored against blended text prompts

This script indexes photos into Elasticsearch:
 - computes SigLIP-2 image embeddings
 - extracts EXIF GPS/time, and writes one doc per image.
 - builds text prototypes from data/peaks.yaml and assigns top-K predicted_peaks to each photo.
 - stores path, clip_image (vector), predicted_peaks, optional gps, and shot_time in the photos index.

USAGE
-----
# Minimal (uses env ES_URL / API key)
python scripts/index_peaks.py --images data/images --yaml data/peaks.yaml --index photos

# Index only the first 200 images (quick test)
python scripts/index_peaks.py --images data/images --yaml data/peaks.yaml --index photos --limit 200

ENV VARS (auth)
---------------
ES_CLOUD_ID      : Elastic Cloud id (optional)
ES_URL           : http://localhost:9200 (default) or your self-hosted URL
ES_API_KEY_B64   : base64 "id:key" (easiest)
ES_API_KEY_ID    : explicit id  (if not using ES_API_KEY_B64)
ES_API_KEY       : explicit key (if not using ES_API_KEY_B64)

BASE_IMAGE_DIR   : Used only to compute nice relative paths for "path" field (thumbnails in UI)
"""

from __future__ import annotations
import os, sys, glob, yaml, argparse, numpy as np
from typing import Optional, List, Iterable

# --- repo path shim (so we can import src/ai_mpi/embeddings.py) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from elasticsearch import Elasticsearch, helpers
from PIL import Image, UnidentifiedImageError, ExifTags

# Optional HEIC support (pillow-heif)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

from ai_mpi.embeddings import Siglip2  # wraps SigLIP-2 (image_vec/text_vec, already L2-normalized)


# -----------------------------------------------------------------------------
# Elasticsearch client via API key
# -----------------------------------------------------------------------------
def es_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from env vars.
    Prefers API key auth; works for Elastic Cloud and self-hosted.
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
    return Elasticsearch(url)


# -----------------------------------------------------------------------------
# Robust EXIF helpers (GPS + DateTimeOriginal)
# -----------------------------------------------------------------------------
_TAGS    = {v: k for k, v in ExifTags.TAGS.items()}
_GPSTAGS = ExifTags.GPSTAGS

def _open_exif(path: str):
    try:
        with Image.open(path) as im:
            return im.getexif() or None
    except Exception:
        return None

def _ratio_to_float(x):
    """Handle PIL's Rational or (num, den) tuples."""
    try:
        return float(x.numerator) / float(x.denominator)
    except Exception:
        try:
            num, den = x
            return float(num) / float(den)
        except Exception:
            return float(x)

def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d, m, s = (_ratio_to_float(v) for v in dms)
        deg = d + m / 60.0 + s / 3600.0
        if isinstance(ref, (bytes, bytearray)):
            ref = ref.decode(errors="ignore")
        if str(ref).upper() in ("S", "W"):
            deg = -deg
        return deg
    except Exception:
        return None

def get_gps(path: str) -> Optional[dict]:
    """
    Parse GPSInfo into {'lat': float, 'lon': float} or None.
    Guards against odd EXIF variants where GPSInfo can be missing or non-dict.
    """
    exif = _open_exif(path)
    if not exif:
        return None
    # GPSInfo tag id = 34853
    gps_ifd = None
    # Pillow >= 9: get_ifd may exist
    try:
        gps_ifd = exif.get_ifd(34853)
    except Exception:
        pass
    if not isinstance(gps_ifd, dict):
        gps_ifd = exif.get(34853) if exif else None
        if not isinstance(gps_ifd, dict):
            return None

    gps = {_GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    lat = _dms_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = _dms_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    if lat is None or lon is None:
        return None
    return {"lat": float(lat), "lon": float(lon)}

def _parse_dt_str(s: str) -> Optional[str]:
    # EXIF DateTimeOriginal is usually "YYYY:MM:DD HH:MM:SS"
    from datetime import datetime
    try:
        return datetime.strptime(s.strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:
        return None

def get_shot_time(path: str) -> Optional[str]:
    """
    Return ISO8601 string for DateTimeOriginal (or DateTime), with TZ offset if present.
    """
    exif = _open_exif(path)
    if not exif:
        return None
    # DateTimeOriginal=36867, Digitized=36868, DateTime=306
    dt = exif.get(36867) or exif.get(36868) or exif.get(306)
    if isinstance(dt, bytes):
        dt = dt.decode(errors="ignore")
    iso = _parse_dt_str(dt) if isinstance(dt, str) else None

    # OffsetTimeOriginal=36881, OffsetTime=36880   (e.g., "+05:45")
    off = exif.get(36881) or exif.get(36880)
    if isinstance(off, bytes):
        off = off.decode(errors="ignore")
    if iso and isinstance(off, str) and off.strip():
        s = off.strip()
        # Normalize common variants to "+HH:MM"
        if len(s) == 5 and s[3] == ":":
            tz = s
        elif s[0] in "+-" and (len(s) in (3, 4, 5)):
            hh = s[1:3]
            mm = (s[3:] if len(s) > 3 else "00").rjust(2, "0")
            tz = f"{s[0]}{hh}:{mm}"
        else:
            tz = None
        if tz:
            return iso + tz
    return iso


# -----------------------------------------------------------------------------
# Peak text prototypes (blended prompts → a single vector per peak)
# -----------------------------------------------------------------------------
def build_text_proto(emb: Siglip2, names: List[str]) -> np.ndarray:
    """
    Create a single representative text vector for a peak:
      - a few descriptive prompts for the PRIMARY name (names[0])
      - one short prompt per alias
      - a light "anti-concept" subtraction to downweight maps/posters
    """
    primary = names[0]
    prompts = [
        f"{primary} mountain peak in the Himalayas, Nepal",
        f"{primary} alpine landscape, snow, rocky ridgeline",
        f"{primary} landmark peak in the Khumbu region",
    ]
    prompts += [f"{alias} mountain peak in Nepal" for alias in names[1:]]

    vecs = np.stack([emb.text_vec(p) for p in prompts], axis=0)  # (N,D), already L2-normalized
    v = vecs.mean(axis=0)
    # anti-concept to de-emphasize drawings/maps/logos
    v = v - 0.25 * emb.text_vec("painting, illustration, poster, map, logo")
    # final L2 normalize (keeps cosine = dot product)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype("float32")


# -----------------------------------------------------------------------------
# Image discovery
# -----------------------------------------------------------------------------
IMG_GLOBS = ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG", "**/*.JPEG", "**/*.PNG",
             "**/*.heic", "**/*.HEIC")

def iter_image_paths(folder: str, patterns: Iterable[str] = IMG_GLOBS) -> Iterable[str]:
    """
    Recursively yield image paths under 'folder' matching 'patterns'.
    Uses glob with recursive ** so nested trip folders are picked up.
    """
    folder = os.path.abspath(folder)
    for pat in patterns:
        yield from glob.iglob(os.path.join(folder, pat), recursive=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Index photos with embeddings + GPS/time + top-K predicted_peaks.")
    ap.add_argument("--images", default=os.getenv("PHOTOS_DIR", "data/images"),
                    help="Folder with photos (recursively scanned)")
    ap.add_argument("--yaml",   default=os.getenv("PEAKS_YAML", "data/peaks.yaml"),
                    help="Peaks YAML (list with 'names' per peak)")
    ap.add_argument("--index",  default=os.getenv("PHOTOS_INDEX", "photos"),
                    help="Target Elasticsearch index for photos")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="Bulk batch size")
    ap.add_argument("--limit",  type=int, default=0,
                    help="Index only the first N images (0 = all)")
    ap.add_argument("--topk-predicted", type=int, default=5,
                    help="How many peak names to store in predicted_peaks[]")
    args = ap.parse_args()

    # For nice relative "path" values (used by the UI to resolve thumbnails)
    BASE_IMAGE_DIR = os.getenv("BASE_IMAGE_DIR", args.images)
    BASE_IMAGE_DIR = os.path.abspath(BASE_IMAGE_DIR)

    es  = es_client()
    emb = Siglip2()

    # ---- Load peak definitions from YAML and build text prototypes ----
    with open(args.yaml, "r") as f:
        peaks = yaml.safe_load(f) or []
    if not isinstance(peaks, list) or not peaks:
        raise SystemExit(f"No peaks found in {args.yaml}. Expected a list of {{id, names, ...}}.")

    # Canonical display name = first alias; store vector per peak
    peak_display_names = [p["names"][0] for p in peaks]
    print(f"Loaded {len(peak_display_names)} peaks; computing text prototypes…")
    P = np.stack([build_text_proto(emb, p["names"]) for p in peaks], axis=0)  # (P,D)

    # ---- Walk images recursively ----
    all_paths = list(iter_image_paths(args.images))
    if args.limit > 0:
        all_paths = all_paths[:args.limit]
    if not all_paths:
        print(f"No images found under {args.images}")
        return

    print(f"Indexing {len(all_paths)} images into '{args.index}' with top-{args.topk_predicted} predicted_peaks…")

    actions, ok, skip = [], 0, 0
    for i, p in enumerate(all_paths, 1):
        # 1) Image embedding (SigLIP-2 image encoder)
        try:
            with Image.open(p) as im:
                v_img = emb.image_vec(im.convert("RGB")).astype("float32")  # already L2-normalized
        except (UnidentifiedImageError, OSError) as e:
            skip += 1
            print(f"Skip [{i}]: {p} ({e})")
            continue

        # 2) Predict peak names by cosine sim between image vec and text prototypes
        #    (cosine == dot product since both are normalized)
        sims = (P @ v_img).astype("float32")             # (P,)
        order = np.argsort(-sims)[:args.topk_predicted]  # top-K indices
        pred_names = [peak_display_names[j] for j in order]

        # 3) Build relative path for UI (keeps repo-portable paths like data/images/IMG_1234.jpg)
        abs_path = os.path.abspath(p)
        try:
            rel_path = os.path.relpath(abs_path, BASE_IMAGE_DIR)
        except ValueError:
            rel_path = os.path.basename(abs_path)  # fallback if different drive

        # 4) Assemble document
        doc = {
            "path": rel_path,
            "clip_image": v_img.tolist(),
            "predicted_peaks": pred_names,
        }
        gps = get_gps(p)
        if gps:
            doc["gps"] = gps
        shot = get_shot_time(p)
        if shot:
            doc["shot_time"] = shot

        actions.append({
            "_op_type": "index",
            "_index": args.index,
            "_id": doc["path"],     # use relative path as stable id
            "_source": doc
        })

        # 5) Bulk flush periodically
        if len(actions) >= args.batch_size:
            helpers.bulk(es, actions)
            ok += len(actions)
            actions.clear()
            print(f"Indexed {ok}/{len(all_paths)}…")

    # Final flush
    if actions:
        helpers.bulk(es, actions)
        ok += len(actions)

    print(f"Done. Indexed={ok}, Skipped={skip}, Total={len(all_paths)}")


if __name__ == "__main__":
    main()
