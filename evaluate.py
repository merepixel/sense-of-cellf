"""
evaluate.py
Evaluate Cell-DINO on sequence 02 using GT tracklet labels.

Protocol
--------
1. Parse man_track.txt → filter tracklets (≥10 frames, exclude ±2 div frames)
2. For each qualifying tracklet: first 50% → gallery, second 50% → query
3. Embed gallery and query crops with fine-tuned Cell-DINO (best checkpoint)
4. Compute CMC @ rank-1 / 5 / 10 and silhouette score
5. Repeat with FROZEN Cell-DINO (no fine-tuning) as baseline

Usage
-----
python evaluate.py \
    --seq_dir        /data/DIC-C2DH-HeLa/02 \
    --gt_tra_dir     /data/DIC-C2DH-HeLa/02_GT/TRA \
    --checkpoint     ./checkpoints/best_checkpoint.pt \
    --seq01_dir      /data/DIC-C2DH-HeLa/01 \
    --seq01_gt_dir   /data/DIC-C2DH-HeLa/01_GT/TRA
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

from data_loader import SequenceData, TrackletDict, parse_man_track_txt, load_gt_mask
from detector import CellDetector, DetectedCell
from embedder import CellDINOEmbedder


# --------------------------------------------------------------------------- #
# Tracklet filtering
# --------------------------------------------------------------------------- #

MIN_TRACKLET_LEN = 10
DIV_EXCLUSION_RADIUS = 2       # exclude ±2 frames around each division event


def get_excluded_frames(tracklets: TrackletDict) -> Dict[int, set]:
    """
    For each cell label, compute the set of frames to exclude near division events.

    Divisions: whenever a daughter cell starts (parent != 0), exclude frames
    [begin-2, begin+2] from the PARENT's tracklet and the start of the daughter's.
    """
    excluded: Dict[int, set] = defaultdict(set)

    for cell_id, (begin, end, parent) in tracklets.items():
        if parent != 0:
            div_frame = begin
            r = DIV_EXCLUSION_RADIUS
            # Exclude around division in daughter
            excluded[cell_id].update(range(div_frame, min(div_frame + r + 1, end + 1)))
            # Exclude end of parent
            if parent in tracklets:
                p_end = tracklets[parent][1]
                excluded[parent].update(range(max(tracklets[parent][0], p_end - r), p_end + 1))

    return excluded


def filter_tracklets(
    tracklets: TrackletDict,
    min_len: int = MIN_TRACKLET_LEN,
) -> List[int]:
    """Return cell IDs whose tracklet length (end-begin+1) >= min_len."""
    return [
        cell_id
        for cell_id, (begin, end, _) in tracklets.items()
        if (end - begin + 1) >= min_len
    ]


def split_tracklet(
    cell_id: int,
    tracklets: TrackletDict,
    excluded_frames: Dict[int, set],
) -> Tuple[List[int], List[int]]:
    """
    Split a tracklet 50/50 temporally into gallery (first half) and query
    (second half), excluding frames near division events.

    Returns (gallery_frames, query_frames)
    """
    begin, end, _ = tracklets[cell_id]
    all_frames = [
        f for f in range(begin, end + 1)
        if f not in excluded_frames.get(cell_id, set())
    ]
    if not all_frames:
        return [], []
    mid = len(all_frames) // 2
    return all_frames[:mid], all_frames[mid:]


# --------------------------------------------------------------------------- #
# Embed crops for a set of (cell_id, frame) instances
# --------------------------------------------------------------------------- #

def embed_instances(
    instances: List[Tuple[int, int]],   # [(cell_id, frame_idx), ...]
    frame_detections: Dict[int, List[DetectedCell]],
    embedder: CellDINOEmbedder,
    no_grad: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each (cell_id, frame_idx) pair, find the matching DetectedCell
    (by gt_cell_id) and embed it.

    Returns
    -------
    embeddings : (N, D) float32 numpy array
    labels     : (N,)  int array of cell_id
    """
    crops: List[np.ndarray] = []
    labels: List[int] = []

    for cell_id, frame_idx in instances:
        cells_in_frame = frame_detections.get(frame_idx, [])
        # Find the detection whose gt_cell_id matches cell_id
        match = next((c for c in cells_in_frame if c.gt_cell_id == cell_id), None)
        if match is None:
            continue
        crops.append(match.crop)
        labels.append(cell_id)

    if not crops:
        return np.zeros((0, embedder.embed_dim)), np.array([], dtype=int)

    embs = embedder.embed_crops(crops, no_grad=no_grad).cpu().numpy()
    return embs, np.array(labels, dtype=int)


# --------------------------------------------------------------------------- #
# CMC curve
# --------------------------------------------------------------------------- #

def compute_cmc(
    query_emb: np.ndarray,      # (Nq, D)
    query_labels: np.ndarray,   # (Nq,)
    gallery_emb: np.ndarray,    # (Ng, D)
    gallery_labels: np.ndarray, # (Ng,)
    ranks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    """
    Compute CMC accuracy at each rank.

    For each query, rank all gallery samples by cosine similarity.
    A query is "correct at rank k" if at least one gallery sample with the
    same label appears in the top-k results.
    """
    if query_emb.shape[0] == 0 or gallery_emb.shape[0] == 0:
        return {r: 0.0 for r in ranks}

    # Cosine similarity matrix (Nq, Ng)
    q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8)
    sim = q @ g.T                                                      # (Nq, Ng)

    # Sort gallery by descending similarity for each query
    sorted_idx = np.argsort(-sim, axis=1)                              # (Nq, Ng)
    sorted_labels = gallery_labels[sorted_idx]                         # (Nq, Ng)

    cmc: Dict[int, float] = {}
    for r in ranks:
        correct = 0
        for i in range(query_emb.shape[0]):
            top_r = sorted_labels[i, :r]
            if query_labels[i] in top_r:
                correct += 1
        cmc[r] = correct / query_emb.shape[0]

    return cmc


# --------------------------------------------------------------------------- #
# Main evaluation function
# --------------------------------------------------------------------------- #

def evaluate(
    seq_dir: Path,
    gt_tra_dir: Path,
    checkpoint: Optional[Path],
    crop_size: int = 96,
    use_gpu: bool = False,
    ranks: Tuple[int, ...] = (1, 5, 10),
) -> None:
    seq_dir = Path(seq_dir)
    gt_tra_dir = Path(gt_tra_dir)

    # ------------------------------------------------------------------ #
    # 1. Load sequence and GT annotations
    # ------------------------------------------------------------------ #
    print(f"[eval] Loading sequence from {seq_dir} …")
    seq_data = SequenceData(seq_dir, gt_tra_dir=gt_tra_dir)

    if seq_data.tracklets is None:
        raise FileNotFoundError(f"man_track.txt not found in {gt_tra_dir}")

    tracklets = seq_data.tracklets
    print(f"[eval] Total tracklets: {len(tracklets)}")

    # ------------------------------------------------------------------ #
    # 2. Filter tracklets
    # ------------------------------------------------------------------ #
    valid_ids = filter_tracklets(tracklets, min_len=MIN_TRACKLET_LEN)
    print(f"[eval] Qualifying tracklets (≥{MIN_TRACKLET_LEN} frames): {len(valid_ids)}")

    excluded_frames = get_excluded_frames(tracklets)

    # ------------------------------------------------------------------ #
    # 3. Detect cells with GT labels (eval mode)
    # ------------------------------------------------------------------ #
    print("[eval] Running CellPose in eval mode (with GT mask lookup) …")
    detector = CellDetector(crop_size=crop_size, use_gpu=use_gpu)
    frame_detections: Dict[int, List[DetectedCell]] = {}

    for frame_idx, rgb in seq_data.frames:
        gt_mask = load_gt_mask(gt_tra_dir, frame_idx)
        cells = detector.detect_frame(frame_idx, rgb, gt_mask=gt_mask)
        frame_detections[frame_idx] = cells

    # ------------------------------------------------------------------ #
    # 4. Build gallery / query splits
    # ------------------------------------------------------------------ #
    gallery_instances: List[Tuple[int, int]] = []   # (cell_id, frame_idx)
    query_instances:   List[Tuple[int, int]] = []

    for cell_id in valid_ids:
        gal_frames, qry_frames = split_tracklet(cell_id, tracklets, excluded_frames)
        gallery_instances.extend((cell_id, f) for f in gal_frames)
        query_instances.extend((cell_id, f) for f in qry_frames)

    print(f"[eval] Gallery instances: {len(gallery_instances)}, "
          f"Query instances: {len(query_instances)}")

    # ------------------------------------------------------------------ #
    # 5. Load embedder and run evaluation (fine-tuned + frozen baseline)
    # ------------------------------------------------------------------ #
    device = "cuda" if use_gpu else "cpu"
    embedder = CellDINOEmbedder(device=device)
    embedder.eval()

    results = {}

    for label, ckpt in [("frozen_baseline", None), ("fine_tuned", checkpoint)]:
        if ckpt is not None:
            if not Path(ckpt).exists():
                print(f"[eval] Checkpoint {ckpt} not found — skipping fine-tuned eval.")
                continue
            embedder.load_checkpoint(ckpt)
        else:
            # Re-instantiate to get fresh frozen weights
            embedder = CellDINOEmbedder(device=device)
            embedder.eval()

        print(f"\n[eval] === {label.upper()} ===")

        with torch.no_grad():
            gal_emb, gal_labels = embed_instances(
                gallery_instances, frame_detections, embedder, no_grad=True
            )
            qry_emb, qry_labels = embed_instances(
                query_instances, frame_detections, embedder, no_grad=True
            )

        print(f"[eval] Embedded gallery: {gal_emb.shape[0]}, query: {qry_emb.shape[0]}")

        # CMC
        cmc = compute_cmc(qry_emb, qry_labels, gal_emb, gal_labels, ranks=ranks)
        for r in ranks:
            print(f"[eval]   Rank-{r:2d} accuracy: {cmc[r]*100:.1f}%")

        # Silhouette (on all instances combined)
        all_emb = np.concatenate([gal_emb, qry_emb], axis=0)
        all_lbl = np.concatenate([gal_labels, qry_labels], axis=0)

        if len(set(all_lbl.tolist())) >= 2 and len(all_emb) >= 4:
            sil = silhouette_score(all_emb, all_lbl, metric="cosine")
            print(f"[eval]   Silhouette score: {sil:.4f}")
        else:
            sil = float("nan")
            print("[eval]   Silhouette: not enough samples")

        results[label] = {"cmc": cmc, "silhouette": sil}

    # ------------------------------------------------------------------ #
    # 6. Summary comparison
    # ------------------------------------------------------------------ #
    print("\n[eval] ====== SUMMARY ======")
    print(f"{'Metric':<22} {'Frozen':>10} {'Fine-tuned':>12}")
    print("-" * 46)
    for r in ranks:
        frozen_v = results.get("frozen_baseline", {}).get("cmc", {}).get(r, float("nan"))
        ft_v     = results.get("fine_tuned",      {}).get("cmc", {}).get(r, float("nan"))
        print(f"Rank-{r:<17} {frozen_v*100:>9.1f}% {ft_v*100:>11.1f}%")
    frozen_sil = results.get("frozen_baseline", {}).get("silhouette", float("nan"))
    ft_sil     = results.get("fine_tuned",      {}).get("silhouette", float("nan"))
    print(f"{'Silhouette':<22} {frozen_sil:>10.4f} {ft_sil:>12.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Cell-DINO on CTC sequence 02")
    p.add_argument("--seq_dir",    type=Path, required=True,
                   help="Path to raw sequence folder, e.g. /data/DIC-C2DH-HeLa/02")
    p.add_argument("--gt_tra_dir", type=Path, required=True,
                   help="Path to GT/TRA folder, e.g. /data/DIC-C2DH-HeLa/02_GT/TRA")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to fine-tuned checkpoint (.pt); omit to skip fine-tuned eval")
    p.add_argument("--crop_size",  type=int, default=96)
    p.add_argument("--use_gpu",    action="store_true")
    p.add_argument("--ranks",      type=int, nargs="+", default=[1, 5, 10])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        seq_dir=args.seq_dir,
        gt_tra_dir=args.gt_tra_dir,
        checkpoint=args.checkpoint,
        crop_size=args.crop_size,
        use_gpu=args.use_gpu,
        ranks=tuple(args.ranks),
    )
