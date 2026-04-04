"""
data_loader.py
Load and preprocess DIC-C2DH-HeLa frames and ground truth annotations.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
# {cell_id: (begin_frame, end_frame, parent_id)}
TrackletDict = Dict[int, Tuple[int, int, int]]
# {frame_idx: {cell_id: (min_row, min_col, max_row, max_col)}}
FrameBBoxDict = Dict[int, Dict[int, Tuple[int, int, int, int]]]


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def load_frame(frame_path: Path) -> np.ndarray:
    """Load a single 16-bit TIF frame, normalize to float32 [0,1]."""
    img = tifffile.imread(str(frame_path)).astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert a 2-D float32 grayscale image to (H, W, 3) by repeating channels."""
    return np.stack([gray, gray, gray], axis=-1)


def load_sequence_frames(seq_dir: Path) -> List[Tuple[int, np.ndarray]]:
    """
    Load all frames for a sequence in sorted order.

    Returns
    -------
    list of (frame_idx, rgb_array) where rgb_array is float32 (H, W, 3) in [0,1]
    """
    tif_files = sorted(seq_dir.glob("t*.tif"))
    frames = []
    for tif_path in tif_files:
        idx = int(tif_path.stem[1:])          # e.g. "t007" → 7
        gray = load_frame(tif_path)
        rgb = to_rgb(gray)
        frames.append((idx, rgb))
    return frames


# --------------------------------------------------------------------------- #
# man_track.txt parsing
# --------------------------------------------------------------------------- #

def parse_man_track_txt(txt_path: Path) -> TrackletDict:
    """
    Parse man_track.txt into a dict.

    Format: L B E P
        L  = cell label (integer)
        B  = first frame the label appears
        E  = last  frame the label appears
        P  = parent label (0 = no parent / root)

    Returns
    -------
    {cell_id: (begin_frame, end_frame, parent_id)}
    """
    tracklets: TrackletDict = {}
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            label, begin, end, parent = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            tracklets[label] = (begin, end, parent)
    return tracklets


def get_division_frames(tracklets: TrackletDict) -> Dict[int, List[int]]:
    """
    Return a mapping {parent_cell_id: [daughter_begin_frame, ...]} for all
    division events (entries with parent != 0).
    """
    divisions: Dict[int, List[int]] = {}
    for cell_id, (begin, end, parent) in tracklets.items():
        if parent != 0:
            divisions.setdefault(parent, []).append(begin)
    return divisions


# --------------------------------------------------------------------------- #
# GT mask (man_trackT.tif) parsing
# --------------------------------------------------------------------------- #

def load_gt_mask(gt_tra_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load the GT tracking mask for a specific frame.
    Returns uint16 array (H, W), or None if the file does not exist.
    """
    mask_path = gt_tra_dir / f"man_track{frame_idx:03d}.tif"
    if not mask_path.exists():
        return None
    return tifffile.imread(str(mask_path)).astype(np.uint16)


def build_frame_bboxes(gt_tra_dir: Path, frame_indices: List[int]) -> FrameBBoxDict:
    """
    Parse GT tracking masks for the given frames and build a bbox lookup.

    Uses skimage regionprops to extract bounding boxes for each labelled cell.

    Returns
    -------
    {frame_idx: {cell_id: (min_row, min_col, max_row, max_col)}}
    """
    from skimage.measure import regionprops

    frame_bboxes: FrameBBoxDict = {}
    for idx in frame_indices:
        mask = load_gt_mask(gt_tra_dir, idx)
        if mask is None:
            continue
        props = regionprops(mask)
        cell_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        for prop in props:
            if prop.label == 0:
                continue
            # regionprops bbox = (min_row, min_col, max_row, max_col)
            cell_bboxes[prop.label] = prop.bbox
        frame_bboxes[idx] = cell_bboxes
    return frame_bboxes


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #

class SequenceData:
    """
    High-level container for a single CTC sequence.

    Parameters
    ----------
    seq_dir : path to the raw image folder (e.g. .../01/)
    gt_tra_dir : path to GT/TRA folder (e.g. .../01_GT/TRA/), or None
    """

    def __init__(self, seq_dir: Path, gt_tra_dir: Optional[Path] = None):
        self.seq_dir = Path(seq_dir)
        self.gt_tra_dir = Path(gt_tra_dir) if gt_tra_dir is not None else None

        self.frames: List[Tuple[int, np.ndarray]] = load_sequence_frames(self.seq_dir)
        self.frame_indices: List[int] = [idx for idx, _ in self.frames]

        # GT annotations (optional — used only in evaluate.py)
        self.tracklets: Optional[TrackletDict] = None
        self.frame_bboxes: Optional[FrameBBoxDict] = None

        if self.gt_tra_dir is not None:
            txt_path = self.gt_tra_dir / "man_track.txt"
            if txt_path.exists():
                self.tracklets = parse_man_track_txt(txt_path)
                self.frame_bboxes = build_frame_bboxes(self.gt_tra_dir, self.frame_indices)

    # ------------------------------------------------------------------ #
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Return the float32 RGB array for a given frame index."""
        for idx, arr in self.frames:
            if idx == frame_idx:
                return arr
        raise KeyError(f"Frame {frame_idx} not found in {self.seq_dir}")

    def num_frames(self) -> int:
        return len(self.frames)

    def get_division_frames(self) -> Dict[int, List[int]]:
        if self.tracklets is None:
            return {}
        return get_division_frames(self.tracklets)
