#!/usr/bin/env python3
# scripts/app.py
# Streamlit UI for multimodal mountain peak search:
#   - Left column: peak list + geo filter
#   - Right column: search + (optional) upload + top full-resolution preview
#   - Results grid: thumbnails in 3 columns, each with a "View Image" button
#
# NOTE ON EDITING:
#   This file has been annotated with detailed comments for clarity.
#   Comments explain design decisions, environment variables, caching, ES queries, and UI flow.

from __future__ import annotations

# ------------------------------
# Standard library imports
# ------------------------------
import os, sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------
# Third-party imports
# ------------------------------
import numpy as np
import streamlit as st
from elasticsearch import Elasticsearch
from PIL import Image, UnidentifiedImageError  # Pillow (PIL) for image I/O

# Optional HEIC support (so .HEIC opens like JPEG/PNG)
# If pillow-heif isn't installed, the app still works for common formats.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    # Silently ignore if HEIC support isn't available.
    pass

# Optional Folium map for interactive geo filter
# If not installed, we fall back to a simple center+radius numeric input.
_HAS_FOLIUM = False
try:
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    _HAS_FOLIUM = True
except Exception:
    _HAS_FOLIUM = False

# ------------------------------
# Local import path setup
# ------------------------------
# The embeddings wrapper (SigLIP-2) lives in src/, so we add it to sys.path.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ai_mpi.embeddings import Siglip2  # noqa: E402  # local dependency (kept after path tweak)


# ==============================================================================
# Page config & constants
# ==============================================================================
# Wide layout is better for side-by-side columns (left filters, right results).
st.set_page_config(page_title="Multimodal Mountain Peak Search", layout="wide")

# Paths / index names can be configured via environment variables to avoid code edits.
BASE_IMAGE_DIR = os.getenv("BASE_IMAGE_DIR", "data/images")  # on-disk image root (relative or absolute)
PHOTOS_INDEX   = os.getenv("PHOTOS_INDEX", "photos")         # ES index for photo documents
PEAKS_INDEX    = os.getenv("PEAKS_INDEX", "peaks_catalog")   # ES index for reference peak vectors

# Default kNN settings (tunable via env without code changes).
DEFAULT_K = int(os.getenv("KNN_K", "320"))
DEFAULT_NUM_CANDIDATES = int(os.getenv("KNN_NUM_CANDIDATES", "5000"))

# Grid layout / thumbnail sizing for the results area.
THUMB_WIDTH = 260  # px for thumbnail width in the grid
GRID_COLS   = 3    # number of columns in the result grid


# ==============================================================================
# ES client & embeddings (cached singletons)
# ==============================================================================
def es_client() -> Elasticsearch:
    """Create an Elasticsearch client.
    Supports Elastic Cloud and self-hosted via env vars:
      - Elastic Cloud: ES_CLOUD_ID + (ES_API_KEY_B64 or ES_API_KEY_ID/ES_API_KEY)
      - Self-hosted:   ES_URL + (optional API key variants)
    If credentials are incomplete for a chosen mode, the app will stop to avoid confusing failures.
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
        # Cloud ID present but no API credentials; fail early for clarity.
        st.stop()

    # Self-hosted modes (URL with or without API key formats).
    if api_key_b64:
        return Elasticsearch(url, api_key=api_key_b64)
    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    return Elasticsearch(url)


@st.cache_resource(show_spinner=False)
def get_es() -> Elasticsearch:
    """Singleton ES client to avoid reconnect cost on every rerun."""
    return es_client()


@st.cache_resource(show_spinner=False)
def get_model() -> Siglip2:
    """Singleton embeddings model wrapper (SigLIP-2)."""
    # If SIGLIP_MODEL_ID env var is set, embeddings.py will respect it.
    return Siglip2()


# ==============================================================================
# Embedding helpers
# ==============================================================================
def l2norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector with numerical stability (adds tiny epsilon)."""
    return v / (np.linalg.norm(v) + 1e-12)


def prompt_vec(emb: Siglip2, peak_name: str) -> np.ndarray:
    """Build a text prompt ensemble for a peak name and return an L2-normalized vector.
    We nudge the vector away from an 'anti-concept' (non-photo content) for robustness.
    """
    prompts = [
        f"a natural photo of the mountain peak {peak_name} in the Himalayas, Nepal",
        f"{peak_name} landmark peak in the Khumbu region, alpine landscape",
        f"{peak_name} mountain summit, snow, rocky ridgeline",
    ]
    proto = np.mean([emb.text_vec(p) for p in prompts], axis=0)
    anti  = emb.text_vec("painting, illustration, poster, map, logo")
    return l2norm(proto - 0.25 * anti).astype("float32")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for unit vectors (equivalent to dot-product)."""
    return float(np.dot(a, b))


# ==============================================================================
# Elasticsearch queries
# ==============================================================================
def agg_peaks_from_photos(es: Elasticsearch, photos_index: str, size: int = 300) -> List[Tuple[str, int]]:
    """Aggregate unique predicted peak names from the photos index and count their occurrences.
    Returns a list of (peak_name, count) sorted by frequency descending (via ES terms agg).
    """
    body = {"size": 0, "aggs": {"peaks": {"terms": {"field": "predicted_peaks", "size": size,
                                                    "order": {"_count": "desc"}}}}}
    resp = es.search(index=photos_index, body=body)
    buckets = resp.get("aggregations", {}).get("peaks", {}).get("buckets", [])
    return [(b["key"], b["doc_count"]) for b in buckets]


def geo_filter_from_bounds(bounds: Optional[Dict[str, Dict[str, float]]]) -> Optional[Dict[str, Any]]:
    """Convert Folium-style bounds to an Elasticsearch geo_bounding_box filter.
    Expected shape:
      bounds = {"southWest": {"lat": .., "lng": ..}, "northEast": {"lat": .., "lng": ..}}
    Returns None if bounds are missing or incomplete.
    """
    if not bounds: return None
    sw, ne = bounds.get("southWest"), bounds.get("northEast")
    if not sw or not ne: return None
    return {"geo_bounding_box": {"gps": {
        "top_left":     {"lat": ne["lat"], "lon": sw["lng"]},
        "bottom_right": {"lat": sw["lat"], "lon": ne["lng"]},
    }}}


def search_by_text(
        es: Elasticsearch,
        qvec: np.ndarray,
        *,
        geo_filter: Optional[Dict[str, Any]],
        index: str = PHOTOS_INDEX,
        k: int = DEFAULT_K,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run a kNN query over photos.clip_image (vector field) with an optional geo filter.
    Returns each hit's _source enriched with its _score for downstream sorting/labels.
    """
    body: Dict[str, Any] = {
        "knn": {
            "field": "clip_image",
            "query_vector": qvec.tolist(),
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        "_source": fields or ["path", "predicted_peaks", "gps", "shot_time", "clip_image"],
    }
    if geo_filter:
        # Combine with a simple bool filter to restrict the vector search by geography.
        body["query"] = {"bool": {"filter": [geo_filter]}}

    resp = es.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {}) or {}
        src["_score"] = float(h.get("_score", 0.0))
        out.append(src)
    return out


def identify_peaks(
        es: Elasticsearch,
        image_vec: np.ndarray,
        *,
        peaks_index: str = PEAKS_INDEX,
        k: int = 3,
        num_candidates: int = 500,
) -> List[Dict[str, Any]]:
    """Given an image embedding, find top-k peak records and return name + scores.
    The cosine/confidence is computed client-side for transparency.
    """
    body = {
        "knn": {
            "field": "text_embed",
            "query_vector": image_vec.tolist(),
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        "_source": ["id", "names", "latlon", "text_embed"],
    }
    resp = es.search(index=peaks_index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    img = image_vec.astype("float32")
    out = []
    for h in hits:
        src = h.get("_source", {})
        # Prefer the first alias in 'names'; fallback to 'id' when empty.
        name = (src.get("names") or [src.get("id")])[0]
        peak_vec = np.asarray(src.get("text_embed", []), dtype=np.float32)
        cos  = float(np.dot(img, peak_vec)) if peak_vec.size else None
        conf = (cos + 1.0) / 2.0 if cos is not None else None
        out.append({"name": name, "id": src.get("id"),
                    "knn_score": float(h.get("_score", 0.0)),
                    "cosine": cos, "confidence": conf, "latlon": src.get("latlon")})
    return out


# ==============================================================================
# Image & rendering helpers
# ==============================================================================
def open_local_image(rel_path: str) -> Optional[Image.Image]:
    """Open an image from BASE_IMAGE_DIR/rel_path and convert to RGB.
    Returns None if the file is missing or unreadable (bad/corrupt format).
    """
    abs_path = os.path.join(BASE_IMAGE_DIR, rel_path)
    try:
        return Image.open(abs_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def caption_for_hit(hit: Dict[str, Any], *, cos: Optional[float] = None, conf: Optional[float] = None) -> str:
    """Assemble a compact caption showing predicted peaks, ES score, cosine, confidence, GPS, and shot date."""
    preds = hit.get("predicted_peaks") or []
    gps   = hit.get("gps")
    ts    = hit.get("shot_time")
    knn   = hit.get("_score")
    bits = []
    if preds: bits.append(f"pred: {', '.join(preds[:2])}")
    if knn is not None: bits.append(f"knn {knn:.3f}")
    if cos is not None: bits.append(f"cos {cos:.3f}")
    if conf is not None: bits.append(f"~{conf*100:.1f}%")
    if gps: bits.append(f"{gps.get('lat'):.3f},{gps.get('lon'):.3f}")
    if ts:  bits.append(ts.split('T')[0])
    return " | ".join(bits)


def attach_cosine_conf(hits: List[Dict[str, Any]], qvec: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    """Attach _cosine and _confidence to each hit when a query vector is available.
    This allows client-side sorting by semantic similarity (cosine) in addition to ES score.
    """
    if qvec is None: return hits
    out = []
    for h in hits:
        v = h.get("clip_image")
        cos = conf = None
        if isinstance(v, list) and v:
            vec = np.asarray(v, dtype=np.float32)
            cos  = float(np.dot(qvec, vec))
            conf = (cos + 1.0) / 2.0
        h2 = {**h, "_cosine": cos, "_confidence": conf}
        out.append(h2)
    return out


def render_hits_grid(
        hits: List[Dict[str, Any]],
        *,
        qvec_for_cosine: Optional[np.ndarray] = None,
        cols: int = GRID_COLS,
        thumb_width: int = THUMB_WIDTH,
) -> None:
    """Render results as a grid of thumbnails with a 'View' button under each.
    The 'View Image' button records which image was selected and triggers a rerun;
    the top preview panel then reads st.session_state['full_image'] to display it.
    """
    if not hits:
        st.info("No results."); return

    rows = (len(hits) + cols - 1) // cols
    idx_global = 0
    for _ in range(rows):
        col_objs = st.columns(cols, gap="small")
        for col in col_objs:
            if idx_global >= len(hits): break
            h = hits[idx_global]; idx_global += 1
            rel = h.get("path")
            im  = open_local_image(rel) if rel else None
            with col:
                if im:
                    # Display the image thumbnail
                    st.image(im, caption=None, width=thumb_width)
                else:
                    st.write("*(image unavailable)*")

                # Optional similarity scores (if a text query vector was used)
                cos = conf = None
                if qvec_for_cosine is not None and isinstance(h.get("clip_image"), list):
                    v = np.asarray(h["clip_image"], dtype=np.float32)
                    cos  = float(np.dot(qvec_for_cosine, v))
                    conf = (cos + 1.0) / 2.0
                st.caption(caption_for_hit(h, cos=cos, conf=conf))

                # Robust full-image viewer: store selection in session state, then rerun.
                # The rerun is needed because Streamlit executes top-to-bottom per interaction.
                if st.button("View Image", key=f"view_{idx_global}_{rel}", use_container_width=False):
                    st.session_state["full_image"] = rel
                    st.rerun()


def show_full_image_panel() -> None:
    """Full-resolution preview panel, with 'Close preview' at the top.
    Displays when st.session_state['full_image'] holds a relative path (set by 'View Image').
    """
    rel = st.session_state.get("full_image")
    if not rel:
        return

    # Resolve absolute path to the selected image (supports both relative and absolute).
    abs_path = rel if os.path.isabs(rel) else os.path.join(BASE_IMAGE_DIR, rel)

    # Open as bytes first (for a clean download payload), then decode for display.
    try:
        with open(abs_path, "rb") as f:
            raw = f.read()
        im = Image.open(BytesIO(raw)).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        st.warning(f"Could not open image: {e}")
        st.session_state["full_image"] = None
        return

    # --- TOP BAR: Close (left) | Download (right) ---
    # Keeping controls above the image avoids scrolling to find the close button during demos.
    top_left, top_right = st.columns([0.75, 0.25])
    with top_left:
        if st.button("Close preview", type="primary"):
            st.session_state["full_image"] = None
            st.rerun()
    with top_right:
        st.download_button(
            "Download",
            data=raw,
            file_name=os.path.basename(abs_path),
            mime="image/jpeg",
            use_container_width=True,
        )

    # Show the full-resolution image below the control bar.
    st.image(im, use_container_width=True)


# ==============================================================================
# UI (layout + interactions)
# ==============================================================================
def main() -> None:
    """Top-level UI composition:
      - Left column: peak list + (optional) interactive map to create a geo filter
      - Right column: search or upload → results grid → click 'View Image' to open preview
    """
    es  = get_es()     # cached ES client
    emb = get_model()  # cached embeddings model

    # Two-column layout: filters (left) and main search/results (right).
    left, main = st.columns([0.28, 0.72], gap="large")

    # ---- LEFT: peak list + geo filter
    with left:
        st.subheader("Peaks in your library")

        # Quick glance at available peaks (terms aggregation on predicted_peaks).
        with st.spinner("Loading peaks…"):
            peak_counts = agg_peaks_from_photos(es, PHOTOS_INDEX, size=300)
        if not peak_counts:
            st.warning("No peaks found in the photos index. Did you run the indexing scripts?")

        # Filterable checkbox list (selecting a peak triggers a kNN search using its text vector).
        peak_filter = st.text_input("Filter peak list", "", placeholder="Type to filter…")
        filtered = [(p, n) for (p, n) in peak_counts if peak_filter.lower() in p.lower()]
        st.caption("Select any to show those photos in the center (uses the same kNN as the search box).")

        selected_peaks = []
        for name, count in filtered[:200]:
            if st.checkbox(f"{name} ({count})", key=f"chk_{name}"):
                selected_peaks.append(name)

        st.markdown("---")
        st.subheader("Geo filter")

        # Use Folium map if available; otherwise present a simple numeric geo radius.
        bounds = None
        if _HAS_FOLIUM:
            st.caption("Pan/zoom the map — results update automatically.")
            fmap = folium.Map(location=[27.95, 86.83], zoom_start=9, tiles="OpenStreetMap")
            m = st_folium(fmap, height=260, width=None, returned_objects=["bounds", "zoom"])
            bounds = m.get("bounds") if isinstance(m, dict) else None
        else:
            st.caption("Install `streamlit-folium` for interactive map. Using center+radius instead.")
            lat = st.number_input("Lat", value=27.95, format="%.5f")
            lon = st.number_input("Lon", value=86.83, format="%.5f")
            radius_km = st.slider("Radius (km)", 10, 250, 120, step=10)
            # crude bbox: latitude ≈ 111 km/deg; longitude ≈ 111 km * cos(lat)
            dlat = radius_km / 111.0
            dlon = radius_km / (111.0 * max(np.cos(np.deg2rad(lat)), 1e-3))
            bounds = {"southWest": {"lat": lat - dlat, "lng": lon - dlon},
                      "northEast": {"lat": lat + dlat, "lng": lon + dlon}}

        # Construct an ES geo filter (or None if bounds are missing).
        geo_filter = geo_filter_from_bounds(bounds)

    # ---- MAIN: top search bar + full-size viewer + results
    with main:
        st.subheader("Search")

        # Two inputs on the top row:
        #   - Left:  text search (peak name)
        #   - Right: uploader inside a collapsed expander (saves vertical space in videos)
        qcol1, qcol2 = st.columns([0.65, 0.35], gap="large")
        with qcol1:
            peak_query = st.text_input("Search by peak name", "", placeholder="e.g., Ama Dablam")
        with qcol2:
            uploaded = None  # ensure variable exists when expander is closed
            with st.expander("Drag and drop file here", expanded=False):
                uploaded = st.file_uploader(
                    label="",
                    type=["jpg", "jpeg", "png", "heic"],
                    label_visibility="collapsed",
                )

        # The full-size viewer panel appears here (above results) when a thumbnail was "Viewed".
        show_full_image_panel()

        st.markdown("---")

        # ------------------------------
        # Build results list (one of three paths):
        #   1) Selected peaks: single-peak → direct kNN; multi-peak → merge top hits
        #   2) Text query: prompt ensemble → kNN
        #   3) Uploaded photo: embed → identify likely peak → kNN using peak's text vector
        # ------------------------------
        results: List[Dict[str, Any]] = []
        qvec_for_scores: Optional[np.ndarray] = None  # for client-side cosine/confidence display
        results_title = None

        if selected_peaks:
            if len(selected_peaks) == 1:
                # Treat checkbox like the search box: run kNN with that peak’s text vector
                nm = selected_peaks[0]
                try:
                    qvec = prompt_vec(emb, nm)
                    hits = search_by_text(
                        es, qvec, geo_filter=geo_filter, index=PHOTOS_INDEX,
                        k=DEFAULT_K, num_candidates=DEFAULT_NUM_CANDIDATES,
                        fields=["path", "predicted_peaks", "gps", "shot_time", "clip_image"]
                    )
                    qvec_for_scores = qvec
                    results = hits
                    results_title = f"Results for “{nm}” ({len(hits)})"
                except Exception as e:
                    st.error(f"Search failed: {e}")
            else:
                # Multiple peaks selected: small kNN per peak, then merge by max score.
                merged: Dict[str, Dict[str, Any]] = {}
                per_peak_k = max(40, DEFAULT_K // max(1, len(selected_peaks)))
                for nm in selected_peaks:
                    try:
                        qvec = prompt_vec(emb, nm)
                        hits = search_by_text(
                            es, qvec, geo_filter=geo_filter, index=PHOTOS_INDEX,
                            k=per_peak_k, num_candidates=max(500, DEFAULT_NUM_CANDIDATES // 2),
                            fields=["path", "predicted_peaks", "gps", "shot_time", "clip_image"]
                        )
                        for h in hits:
                            key = h.get("path")
                            # Deduplicate by max ES score across all selected peaks.
                            if key not in merged or h["_score"] > merged[key]["_score"]:
                                merged[key] = h
                    except Exception as e:
                        st.warning(f"Query failed for {nm}: {e}")
                results = sorted(merged.values(), key=lambda d: d["_score"], reverse=True)
                qvec_for_scores = None  # no single cosine reference
                results_title = f"Results for {', '.join(selected_peaks[:3])}{'…' if len(selected_peaks) > 3 else ''}"

        elif peak_query.strip():
            # Free-form text query: embed prompt → kNN search
            try:
                qvec = prompt_vec(emb, peak_query.strip())
                hits = search_by_text(
                    es, qvec, geo_filter=geo_filter, index=PHOTOS_INDEX,
                    k=DEFAULT_K, num_candidates=DEFAULT_NUM_CANDIDATES,
                    fields=["path", "predicted_peaks", "gps", "shot_time", "clip_image"]
                )
                qvec_for_scores = qvec
                results = hits
                results_title = f"Results for “{peak_query.strip()}” ({len(hits)})"
            except Exception as e:
                st.error(f"Search failed: {e}")

        elif uploaded is not None:
            # Identify the most likely peak from the uploaded photo, then search similar photos.
            try:
                with Image.open(uploaded) as im:
                    ivec = emb.image_vec(im.convert("RGB")).astype("float32")
                guesses = identify_peaks(es, ivec, peaks_index=PEAKS_INDEX, k=3, num_candidates=500)
                if not guesses:
                    st.warning("Couldn’t identify a likely peak.")
                else:
                    best = guesses[0]
                    st.write(f"**Top guess:** {best['name']} — "
                             f"knn {best['knn_score']:.3f}, cos {best['cosine']:.3f}, "
                             f"~{best['confidence']*100:.1f}%")
                    # Use the guessed peak name as the query term for a stable text-based search.
                    qvec = prompt_vec(emb, best["name"])
                    hits = search_by_text(
                        es, qvec, geo_filter=geo_filter, index=PHOTOS_INDEX,
                        k=DEFAULT_K, num_candidates=DEFAULT_NUM_CANDIDATES,
                        fields=["path", "predicted_peaks", "gps", "shot_time", "clip_image"]
                    )
                    qvec_for_scores = qvec
                    results = hits
                    results_title = f"Similar photos for “{best['name']}”"
                    st.caption("Using the best peak’s text vector for stability.")
            except Exception as e:
                st.error(f"Identification failed: {e}")

        else:
            # Helpful hint when the user hasn't selected peaks, typed text, or uploaded a photo.
            st.info("Tip: select peaks on the left, type a name above, or upload a photo to identify.")

        # ----- SORTING -----
        if results:
            # Attach cosine/confidence (if we have a query vector), then sort as requested.
            results = attach_cosine_conf(results, qvec_for_scores)
            sort_choice = st.selectbox(
                "Sort by",
                ["Confidence (desc)", "kNN score (desc)", "Newest first"],
                index=0,
                help="Confidence is cosine mapped to [0,1]; kNN score is ES score; Newest uses shot_time."
            )
            if sort_choice == "Confidence (desc)":
                results.sort(key=lambda d: (d.get("_confidence") is not None, d.get("_confidence") or -1), reverse=True)
            elif sort_choice == "kNN score (desc)":
                results.sort(key=lambda d: (d.get("_score") is not None, d.get("_score") or -1), reverse=True)
            else:  # Newest first
                results.sort(key=lambda d: d.get("shot_time", ""), reverse=True)

        # ----- RESULTS GRID -----
        if results_title: st.write(f"### {results_title}")
        if results:
            render_hits_grid(results, qvec_for_cosine=qvec_for_scores, cols=GRID_COLS, thumb_width=THUMB_WIDTH)


# Entrypoint wrapper so exceptions surface in the Streamlit UI.
if __name__ == "__main__":
    try:
        # Launch with: streamlit run scripts/app.py
        main()
    except Exception as e:
        # Show startup exceptions in the UI instead of a blank page.
        st.exception(e)
