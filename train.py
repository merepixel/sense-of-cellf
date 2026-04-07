"""
train.py
Self-supervised contrastive fine-tuning of Cell-DINO on sequence 01.

No GT labels are used.  Positive pairs are built by spatial matching of
CellPose detections across consecutive frames.  Within-frame hard negatives
are mined with cosine similarity before loss computation.

Usage
-----
python train.py --seq_dir /data/DIC-C2DH-HeLa/01 \
                --output_dir ./checkpoints \
                --epochs 20 \
                --batch_size 32
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import silhouette_score

from data_loader import SequenceData
from detector import CellDetector, DetectedCell, match_cells_across_frames
from embedder import CellDINOEmbedder
from logger import get_logger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, D) x (M, D) → (N, M) cosine similarity matrix."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T


# --------------------------------------------------------------------------- #
# NT-Xent / InfoNCE loss (manual implementation, no extra deps required)
# --------------------------------------------------------------------------- #

class NTXentLoss(torch.nn.Module):
    """
    NT-Xent loss for a batch of (anchor, positive) pairs.

    For each anchor i, the positive is pairs[i] and all other samples
    (including cross-pair) are negatives.

    Optionally adds hard-negative weighting by replacing the uniform
    denominator with an importance-weighted version.

    Parameters
    ----------
    temperature : InfoNCE temperature τ
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,          # (N, D) — embeddings from frame t
        positives: torch.Tensor,        # (N, D) — matched embeddings from frame t+1
        hard_negatives: Optional[torch.Tensor] = None,  # (N, K, D) within-frame negatives
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Each row i in anchors is matched to row i in positives.
        Hard negatives (if provided) are concatenated into the denominator pool.
        """
        N = anchors.shape[0]
        if N < 2:
            return torch.tensor(0.0, requires_grad=True, device=anchors.device)

        # All embeddings: [anchors | positives]  shape (2N, D)
        all_emb = torch.cat([anchors, positives], dim=0)              # (2N, D)
        all_emb = F.normalize(all_emb, dim=-1)

        # Similarity matrix (2N, 2N) / τ
        sim = (all_emb @ all_emb.T) / self.temperature

        # Mask out self-similarity on diagonal
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=anchors.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Positive indices: for anchor i (row i) → positive is row i+N
        #                   for positive i (row i+N) → anchor is row i
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=anchors.device),
            torch.arange(0, N, device=anchors.device),
        ])                                                             # (2N,)

        # If hard negatives provided, append them to the similarity rows of anchors
        if hard_negatives is not None:
            # hard_negatives: (N, K, D)
            hn = F.normalize(hard_negatives, dim=-1)                  # (N, K, D)
            # Anchor-to-hard-neg sims: (N, K)
            hn_sim = torch.einsum("nd,nkd->nk", F.normalize(anchors, dim=-1), hn) / self.temperature
            # Build augmented rows for anchor half only
            # Row i of sim is (2N,); we append K hard-neg sims → (2N+K,)
            # This requires padding the positive rows too — simpler to just add
            # hard-neg loss as auxiliary InfoNCE on anchor rows only.
            loss_hn = self._infonce_rows(
                sim[:N],                        # anchor rows, shape (N, 2N)
                pos_idx[:N],                    # positive column for each anchor
                extra_neg_cols=hn_sim,          # (N, K) additional negative similarities
            )
            loss_base = self._infonce_rows(sim, pos_idx)
            return (loss_base + loss_hn) * 0.5

        return self._infonce_rows(sim, pos_idx)

    @staticmethod
    def _infonce_rows(
        sim: torch.Tensor,
        pos_idx: torch.Tensor,
        extra_neg_cols: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-entropy loss where for each row i the correct column is pos_idx[i].

        extra_neg_cols: (N, K) — appended as additional negative logits.
        """
        if extra_neg_cols is not None:
            N = extra_neg_cols.shape[0]
            # Only modify the first N rows (anchor rows)
            sim_aug = torch.cat([sim[:N], extra_neg_cols], dim=1)     # (N, 2N+K)
            loss_aug = F.cross_entropy(sim_aug, pos_idx[:N])
            loss_rest = F.cross_entropy(sim[N:], pos_idx[N:])
            return (loss_aug * N + loss_rest * (sim.shape[0] - N)) / sim.shape[0]

        return F.cross_entropy(sim, pos_idx)


# --------------------------------------------------------------------------- #
# Within-frame hard negative mining
# --------------------------------------------------------------------------- #

def mine_hard_negatives(
    embeddings: torch.Tensor,        # (N, D) embeddings for one frame
    anchor_indices: List[int],       # which rows are anchors
    k: int = 4,
) -> torch.Tensor:
    """
    For each anchor, find the top-k most cosine-similar cells in the same
    frame (excluding itself).

    Returns
    -------
    hard_negs : (len(anchor_indices), k, D) tensor
                (padded with zeros if fewer than k negatives available)
    """
    N = embeddings.shape[0]
    emb_norm = F.normalize(embeddings, dim=-1)
    sim_mat = emb_norm @ emb_norm.T                                    # (N, N)

    hard_negs = []
    for i in anchor_indices:
        row = sim_mat[i].clone()
        row[i] = -2.0                                                  # exclude self
        topk_vals, topk_idx = torch.topk(row, min(k, N - 1))
        selected = embeddings[topk_idx]                                # (≤k, D)
        if selected.shape[0] < k:
            pad = torch.zeros(k - selected.shape[0], embeddings.shape[1],
                              device=embeddings.device)
            selected = torch.cat([selected, pad], dim=0)
        hard_negs.append(selected)

    return torch.stack(hard_negs, dim=0)                               # (|anchors|, k, D)


# --------------------------------------------------------------------------- #
# Silhouette score proxy (no labels needed — uses CellPose track IDs)
# --------------------------------------------------------------------------- #

def compute_proxy_silhouette(
    embedder: CellDINOEmbedder,
    detections: Dict[int, List[DetectedCell]],
    held_out_frames: List[int],
    min_tracks: int = 5,
) -> float:
    """
    Compute silhouette score on held-out frames using spatial-proximity
    pseudo-labels as class assignments (not GT).

    Pseudo-labelling: we use a simple connected-components approach — greedily
    link detections across the held-out frames by centroid proximity and assign
    a unique pseudo-ID to each chain.  This is a rough proxy, but good enough
    to select the best checkpoint.

    Returns silhouette score in [-1, 1], or -1.0 on failure.
    """
    embedder.eval()

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    pseudo_id = 0
    prev_cells: List[DetectedCell] = []
    prev_ids: List[int] = []

    for fidx in sorted(held_out_frames):
        if fidx not in detections:
            continue
        cells = detections[fidx]
        if not cells:
            continue

        crops = [c.crop for c in cells]
        with torch.no_grad():
            embs = embedder.embed_crops(crops, no_grad=True).cpu().numpy()

        if not prev_cells:
            # First frame: assign new IDs to all
            curr_ids = list(range(pseudo_id, pseudo_id + len(cells)))
            pseudo_id += len(cells)
        else:
            from detector import match_cells_across_frames
            pairs = match_cells_across_frames(prev_cells, cells)
            matched_curr = {j: prev_ids[i] for i, j in pairs}
            curr_ids = []
            for j in range(len(cells)):
                if j in matched_curr:
                    curr_ids.append(matched_curr[j])
                else:
                    curr_ids.append(pseudo_id)
                    pseudo_id += 1

        all_embeddings.extend(embs)
        all_labels.extend(curr_ids)
        prev_cells = cells
        prev_ids = curr_ids

    if len(set(all_labels)) < 2 or len(all_embeddings) < 4:
        return -1.0

    try:
        score = silhouette_score(
            np.array(all_embeddings),
            np.array(all_labels),
            metric="cosine",
        )
    except Exception:
        score = -1.0

    embedder.train()
    return float(score)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train(
    seq_dir: Path,
    output_dir: Path,
    run_name: Optional[str] = None,
    epochs: int = 20,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    hard_neg_k: int = 4,
    silhouette_every: int = 2,
    held_out_frac: float = 0.2,
    max_dist_px: float = 50.0,
    iou_threshold: float = 0.2,
    crop_size: int = 96,
    use_gpu: bool = False,
    seed: int = 42,
    resume_from: Optional[Path] = None,
    start_epoch: int = 1,
) -> None:
    set_seed(seed)
    output_dir = Path(output_dir)
    if run_name:
        output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger("train", output_dir, "train.log")

    # ------------------------------------------------------------------ #
    # 1. Load frames
    # ------------------------------------------------------------------ #
    log.info(f"Loading sequence from {seq_dir} …")
    seq_data = SequenceData(seq_dir, gt_tra_dir=None)
    n_frames = seq_data.num_frames()
    log.info(f"{n_frames} frames loaded.")

    n_held = max(2, int(n_frames * held_out_frac))
    all_fidx = seq_data.frame_indices
    train_fidx = all_fidx[:-n_held]
    held_fidx = all_fidx[-n_held:]
    log.info(f"Train frames: {len(train_fidx)}, held-out: {len(held_fidx)}")

    # ------------------------------------------------------------------ #
    # 2. Detect cells in all training frames (CellPose, no labels)
    # ------------------------------------------------------------------ #
    log.info("Running CellPose segmentation …")
    detector = CellDetector(crop_size=crop_size, use_gpu=use_gpu)

    train_detections: Dict[int, List[DetectedCell]] = {}
    for fidx in train_fidx:
        rgb = seq_data.get_frame(fidx)
        train_detections[fidx] = detector.detect_frame(fidx, rgb, gt_mask=None)

    held_detections: Dict[int, List[DetectedCell]] = {}
    for fidx in held_fidx:
        rgb = seq_data.get_frame(fidx)
        held_detections[fidx] = detector.detect_frame(fidx, rgb, gt_mask=None)

    # ------------------------------------------------------------------ #
    # 3. Build positive pairs across consecutive training frames
    # ------------------------------------------------------------------ #
    log.info("Building positive pairs …")
    pos_pair_data = []
    sorted_train = sorted(train_fidx)
    for i in range(len(sorted_train) - 1):
        t, t1 = sorted_train[i], sorted_train[i + 1]
        cells_t = train_detections.get(t, [])
        cells_t1 = train_detections.get(t1, [])
        pairs = match_cells_across_frames(
            cells_t, cells_t1,
            iou_threshold=iou_threshold,
            max_dist_px=max_dist_px,
        )
        if pairs:
            pos_pair_data.append((cells_t, cells_t1, pairs))

    total_pairs = sum(len(p) for _, _, p in pos_pair_data)
    log.info(f"Found {total_pairs} positive pairs across {len(pos_pair_data)} consecutive frame-pairs.")

    # ------------------------------------------------------------------ #
    # 4. Load embedder
    # ------------------------------------------------------------------ #
    log.info("Loading Cell-DINO embedder …")
    embedder = CellDINOEmbedder(device="cuda" if use_gpu else "cpu")

    if resume_from is not None:
        resume_from = Path(resume_from)
        embedder.load_checkpoint(resume_from)
        log.info(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

    embedder.train()

    optimizer = AdamW(embedder.backbone.parameters(), lr=lr, weight_decay=weight_decay)
    remaining = max(1, epochs - start_epoch + 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=lr * 0.1)
    criterion = NTXentLoss(temperature=temperature)

    # Log config
    log.info(
        f"Config: epochs={epochs} start={start_epoch} lr={lr} wd={weight_decay} "
        f"temp={temperature} hard_neg_k={hard_neg_k} crop={crop_size}"
    )

    # ------------------------------------------------------------------ #
    # 5. Epoch loop
    # ------------------------------------------------------------------ #
    best_silhouette = -2.0
    best_ckpt_path = output_dir / "best_checkpoint.pt"

    for epoch in range(start_epoch, epochs + 1):
        embedder.train()
        epoch_loss = 0.0
        n_batches = 0

        random.shuffle(pos_pair_data)

        for cells_t, cells_t1, matched in pos_pair_data:
            if not matched:
                continue

            anchor_idx   = [i for i, _ in matched]
            pos_idx_list = [j for _, j in matched]

            anchor_crops = [cells_t[i].crop for i in anchor_idx]
            pos_crops    = [cells_t1[j].crop for j in pos_idx_list]

            if not anchor_crops:
                continue

            anchors_emb = embedder.embed_crops(anchor_crops, no_grad=False)

            with torch.no_grad():
                pos_emb     = embedder.embed_crops(pos_crops, no_grad=True)
                all_crops_t = [c.crop for c in cells_t]
                all_emb_t   = embedder.embed_crops(all_crops_t, no_grad=True)

            hn   = mine_hard_negatives(all_emb_t, anchor_idx, k=hard_neg_k)
            loss = criterion(anchors_emb, pos_emb, hard_negatives=hn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.backbone.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            del anchors_emb, pos_emb, all_emb_t, hn, loss
            if use_gpu:
                torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        msg = f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f}"

        if epoch % silhouette_every == 0 or epoch == epochs:
            sil = compute_proxy_silhouette(embedder, held_detections, held_fidx)
            msg += f" | silhouette={sil:.4f}"

            if sil > best_silhouette:
                best_silhouette = sil
                embedder.save_checkpoint(best_ckpt_path)
                msg += " ← best"

        log.info(msg)

    log.info(f"Training complete. Best silhouette: {best_silhouette:.4f}")
    log.info(f"Best checkpoint: {best_ckpt_path}")
    embedder.save_checkpoint(output_dir / "final_checkpoint.pt")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train Cell-DINO on CTC sequence (no labels)")
    p.add_argument("--seq_dir", type=Path, required=True,
                   help="Path to raw sequence folder, e.g. /data/DIC-C2DH-HeLa/01")
    p.add_argument("--output_dir", type=Path, default=Path("./checkpoints"),
                   help="Directory to save checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--hard_neg_k", type=int, default=4)
    p.add_argument("--silhouette_every", type=int, default=2)
    p.add_argument("--crop_size", type=int, default=96)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        seq_dir=args.seq_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        hard_neg_k=args.hard_neg_k,
        silhouette_every=args.silhouette_every,
        crop_size=args.crop_size,
        use_gpu=args.use_gpu,
        seed=args.seed,
    )
