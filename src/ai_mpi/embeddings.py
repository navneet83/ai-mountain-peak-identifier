# src/ai_mpi/embeddings.py
"""
Tiny wrapper around a SigLIP-2 model to produce L2-normalized embeddings for:
  - IMAGEs  → used for photo vectors (kNN over photos.clip_image)
  - TEXT    → used for peak name prototypes / queries (kNN over peaks_catalog.text_embed)

Why this wrapper exists
-----------------------
- Keeps scripts simple (one place to choose model/device and preprocessing).
- Always returns **float32, unit-length** vectors so **cosine = dot product**,
  which is what Elasticsearch's cosine similarity expects for stable scoring.

Model selection
---------------
Default model id can be overridden at runtime:

  - Environment variable:  SIGLIP_MODEL_ID
  - Constructor argument:  Siglip2(model_id="google/…")

The default below is a light model that runs on CPU easily. If you switch to a
larger/hi-res model, make sure your Elasticsearch `dense_vector.dims` matches.

Note: Both image and text encoders come from the same checkpoint, exposed by HF
via `get_image_features` and `get_text_features`.
"""

from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class Siglip2:
    """
    Minimal, friendly wrapper around a SigLIP-2 checkpoint.

    Public methods:
      - image_vec(PIL.Image) -> np.ndarray   # shape: (D,)
      - text_vec(str)        -> np.ndarray   # shape: (D,)

    All outputs are float32 and L2-normalized.
    """

    def __init__(
            self,
            model_id: str | None = None,
            device: str | None = None,
            max_text_len: int = 64,
    ) -> None:
        """
        Args:
          model_id: HF model id; if None, uses env SIGLIP_MODEL_ID or a sensible default.
          device:  "cuda" / "cpu" / "mps"; if None, auto-detect CUDA → else CPU.
          max_text_len: tokenizer max length for text prompts (keep short & factual).
        """
        # Keep the existing default model; override if needed
        default_model = "google/siglip2-base-patch16-224"
        self.model_id = model_id or os.getenv("SIGLIP_MODEL_ID", default_model)

        # Device selection: prefer CUDA if available; otherwise CPU works fine for small batches.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Processor handles both image + text preprocessing for SigLIP-2.
        # (Some processors print a warning about "slow" vs "fast"—safe to ignore.)
        self.proc = AutoProcessor.from_pretrained(self.model_id)

        # The model exposes get_image_features / get_text_features on forward.
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

        # Store config
        self.max_text_len = int(max_text_len)

    # ---------------------------------------------------------------------
    # Internal helper: L2 normalization with numerical safety
    # ---------------------------------------------------------------------
    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length along last dimension so cosine == dot.

        Works for shape (D,) or (N,D). Adds tiny epsilon to avoid div by zero.
        """
        denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x / denom

    # ---------------------------------------------------------------------
    # Public API: single image → vector
    # ---------------------------------------------------------------------
    def image_vec(self, pil_image) -> np.ndarray:
        """
        Compute a single IMAGE embedding (float32, unit-norm).

        Args:
          pil_image: a PIL Image (RGB/other modes accepted; we convert to RGB)

        Returns:
          np.ndarray of shape (D,), dtype float32, L2-normalized
        """
        batch = self.proc(images=pil_image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            vec = self.model.get_image_features(**batch)  # shape: (1, D) tensor
        v = vec.detach().cpu().numpy().astype("float32")  # → numpy
        return self._norm(v)[0]  # → (D,)

    # ---------------------------------------------------------------------
    # Public API: single text → vector
    # ---------------------------------------------------------------------
    def text_vec(self, text: str) -> np.ndarray:
        """
        Compute a single TEXT embedding (float32, unit-norm).

        Tip: Keep prompts short and factual (avoid flowery language).
             Example: "Ama Dablam mountain peak in the Himalayas, Nepal"

        Returns:
          np.ndarray of shape (D,), dtype float32, L2-normalized
        """
        toks = self.proc(
            text=[text.lower()],                # lowercasing keeps things simple/consistent
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            vec = self.model.get_text_features(**toks)  # shape: (1, D) tensor
        v = vec.detach().cpu().numpy().astype("float32")
        return self._norm(v)[0]

    # ---------------------------------------------------------------------
    # (Nice-to-have) batched helpers — handy in notebooks / bulk indexing
    # ---------------------------------------------------------------------
    def images_vec(self, images: Iterable) -> np.ndarray:
        """
        Batch version of image_vec. Accepts an iterable of PIL Images.
        Returns (N, D) float32 unit-norm array.
        """
        # The HF processor handles batching if you pass a list.
        batch = self.proc(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        with torch.no_grad():
            vec = self.model.get_image_features(**batch)  # (N, D)
        v = vec.detach().cpu().numpy().astype("float32")
        return self._norm(v)

    def texts_vec(self, texts: List[str]) -> np.ndarray:
        """
        Batch version of text_vec. Returns (N, D) float32 unit-norm array.
        """
        toks = self.proc(
            text=[t.lower() for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            vec = self.model.get_text_features(**toks)  # (N, D)
        v = vec.detach().cpu().numpy().astype("float32")
        return self._norm(v)
