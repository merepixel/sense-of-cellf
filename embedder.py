"""
embedder.py
Load Cell-DINO (DINOv2 ViT fine-tuned on single-cell microscopy) from HuggingFace
and compute CLS-token embeddings for cell crops.

Model: recursionpharma/OpenPhenom  (DINOv2 ViT-S/8 fine-tuned on RxRx datasets)
Fallback: facebook/dinov2-base     (general DINOv2, works well on microscopy)

The embedder is used:
  - During training  (with gradient) — Cell-DINO weights are updated
  - During evaluation (no gradient)  — frozen after loading checkpoint
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Silence HuggingFace Hub authentication warnings — we only use public models
# and don't need a token.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*", category=UserWarning)

# --------------------------------------------------------------------------- #
# HuggingFace model IDs
# --------------------------------------------------------------------------- #

# Primary attempt: single-cell microscopy fine-tuned DINOv2 (falls back if incompatible)
CELL_DINO_MODEL_ID = "recursionpharma/OpenPhenom"
# Fallback: DINOv2-small — half the memory of base, still 384-dim, good on microscopy
FALLBACK_MODEL_ID = "facebook/dinov2-small"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def build_transform(img_size: int = 224) -> transforms.Compose:
    """
    Standard DINOv2 preprocessing:
      - Resize to img_size x img_size
      - Normalize with ImageNet mean/std (DINOv2 was trained with these)
    Input: float32 numpy array (H, W, 3) in [0, 1]
    """
    return transforms.Compose([
        transforms.ToTensor(),                        # (H, W, 3) → (3, H, W), keeps [0,1]
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# --------------------------------------------------------------------------- #
# CellDINOEmbedder
# --------------------------------------------------------------------------- #

class CellDINOEmbedder(nn.Module):
    """
    Wraps a DINOv2 ViT model and exposes a forward pass that returns the
    CLS-token embedding (768-dim for ViT-B, 384-dim for ViT-S).

    Parameters
    ----------
    model_id  : HuggingFace model identifier
    img_size  : input resolution expected by the model (default 224)
    device    : 'cuda', 'mps', or 'cpu'
    """

    def __init__(
        self,
        model_id: str = CELL_DINO_MODEL_ID,
        img_size: int = 224,
        device: Optional[str] = None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.img_size = img_size
        self.transform = build_transform(img_size)

        self.backbone = self._load_backbone(model_id)
        self.backbone.to(self.device)

        # Enable gradient checkpointing to trade compute for memory.
        # Only if the model explicitly supports it (MAEModel does not).
        if (hasattr(self.backbone, 'gradient_checkpointing_enable')
                and getattr(self.backbone, 'supports_gradient_checkpointing', False)):
            self.backbone.gradient_checkpointing_enable()

        # Infer embedding dimension from a dummy forward pass.
        # If it fails (e.g. model expects different channels/resolution),
        # fall back to dinov2-base automatically.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
            try:
                out = self._forward_backbone(dummy)
            except Exception as e:
                if model_id != FALLBACK_MODEL_ID:
                    print(f"[embedder] {model_id} incompatible with 3-ch {img_size}px input "
                          f"({e.__class__.__name__}: {e})")
                    print(f"[embedder] Falling back to {FALLBACK_MODEL_ID} …")
                    self.backbone = self._load_backbone(FALLBACK_MODEL_ID)
                    self.backbone.to(self.device)
                    out = self._forward_backbone(dummy)
                else:
                    raise
        self.embed_dim: int = out.shape[-1]

    # ------------------------------------------------------------------ #

    def _load_backbone(self, model_id: str) -> nn.Module:
        """
        Attempt to load model_id; fall back to FALLBACK_MODEL_ID if unavailable.
        After loading, probes the forward signature to determine calling convention.
        Sets self._call_convention to one of:
          'pixel_values_cls'  — transformers DINOv2 style (last_hidden_state[:, 0])
          'pixel_values_pool' — transformers with pooler_output
          'positional_cls'    — MAE / other models taking positional x arg
          'positional_direct' — torch.hub DINOv2 returning tensor directly
        """
        try:
            from transformers import AutoModel
            print(f"[embedder] Loading {model_id} via transformers …")
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self._use_transformers = True
            # Probe calling convention with a tiny dummy (on CPU before .to(device))
            self._call_convention = self._probe_convention(model)
            print(f"[embedder] Call convention: {self._call_convention}")
            return model
        except Exception as e:
            print(f"[embedder] Could not load {model_id}: {e}")
            if model_id != FALLBACK_MODEL_ID:
                print(f"[embedder] Falling back to {FALLBACK_MODEL_ID} …")
                return self._load_backbone(FALLBACK_MODEL_ID)
            raise RuntimeError(
                "Could not load any backbone. "
                "pip install transformers torch torchvision"
            ) from e

    @staticmethod
    def _probe_convention(model: nn.Module) -> str:
        """Inspect the forward signature to determine calling convention."""
        import inspect
        try:
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            params = []

        if 'pixel_values' in params:
            # Standard transformers DINOv2 — check whether it has pooler_output
            # by looking at the config (no forward pass needed)
            cfg = getattr(model, 'config', None)
            if cfg is not None and getattr(cfg, 'model_type', '') in ('vit', 'dinov2'):
                return 'pixel_values_cls'
            return 'pixel_values_cls'
        else:
            # MAE / custom models (OpenPhenom) — positional x arg
            # Run a tiny forward to see if output is a plain tensor or has last_hidden_state
            model.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                try:
                    out = model(dummy)
                    if isinstance(out, torch.Tensor):
                        return 'positional_direct'
                    if hasattr(out, 'last_hidden_state'):
                        return 'positional_cls'
                except Exception:
                    pass
            return 'positional_direct'

    def _forward_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return a (B, D) embedding tensor using the probed calling convention.

        Grad context is managed entirely by the caller (embed_crops / training loop).
        Do NOT use set_grad_enabled here — it overrides explicit torch.no_grad()
        contexts in the training loop and causes hard-negative gradients to leak
        into the anchor backward pass, producing NaN loss.
        """
        conv = self._call_convention
        if conv == 'pixel_values_cls':
            out = self.backbone(pixel_values=pixel_values)
            return out.last_hidden_state[:, 0, :]
        elif conv == 'pixel_values_pool':
            out = self.backbone(pixel_values=pixel_values)
            return out.pooler_output
        elif conv == 'positional_cls':
            out = self.backbone(pixel_values)
            return out.last_hidden_state[:, 0, :]
        else:  # positional_direct
            out = self.backbone(pixel_values)
            if isinstance(out, torch.Tensor):
                # If shape is (B, D) use as-is; if (B, N, D) take CLS (index 0)
                return out[:, 0] if out.dim() == 3 else out
            return out.last_hidden_state[:, 0, :]

    # ------------------------------------------------------------------ #
    # nn.Module forward
    # ------------------------------------------------------------------ #

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : (B, 3, H, W) tensor, already normalised

        Returns
        -------
        embeddings : (B, D) L2-normalised CLS token embeddings
        """
        embeddings = self._forward_backbone(pixel_values)
        return nn.functional.normalize(embeddings, dim=-1)

    # ------------------------------------------------------------------ #
    # Convenience: embed a batch of numpy crops
    # ------------------------------------------------------------------ #

    def embed_crops(
        self,
        crops: List[np.ndarray],
        no_grad: bool = False,
    ) -> torch.Tensor:
        """
        Embed a list of float32 numpy crops (H, W, 3) in [0,1].

        Parameters
        ----------
        crops   : list of ndarray, each (H, W, 3) float32 [0,1]
        no_grad : if True, run under torch.no_grad()

        Returns
        -------
        embeddings : (N, D) float32 tensor on self.device
        """
        if not crops:
            return torch.zeros(0, self.embed_dim, device=self.device)

        tensors = [self.transform(crop) for crop in crops]          # list of (3, H, W)
        batch = torch.stack(tensors).to(self.device)               # (N, 3, H, W)

        if no_grad:
            with torch.no_grad():
                return self.forward(batch)
        else:
            return self.forward(batch)

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model weights only (used by evaluate.py)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), str(path))
        print(f"[embedder] Checkpoint saved → {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model weights.  Handles both plain state-dicts (old format) and
        the full training checkpoint dicts written by train.py (new format)."""
        path = Path(path)
        ckpt = torch.load(str(path), map_location=self.device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.backbone.load_state_dict(state)
        print(f"[embedder] Checkpoint loaded ← {path}")
