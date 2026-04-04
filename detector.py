"""
detector.py
Run CellPose on each frame and extract cell crops.

Each detected cell is represented as a DetectedCell named-tuple containing:
    frame_idx    : int
    cell_pos     : (row_centroid, col_centroid) in pixels
    bbox         : (min_row, min_col, max_row, max_col)
    crop         : float32 ndarray of shape (H, W, 3), normalised [0,1]
    mask_label   : int — CellPose instance label in the segmentation mask
    gt_cell_id   : int or None — GT label from man_trackT.tif (eval mode only)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

from skimage.measure import regionprops

# data_loader utilities
from data_loader import SequenceData, load_gt_mask


# --------------------------------------------------------------------------- #
# Data structure
# --------------------------------------------------------------------------- #

@dataclass
class DetectedCell:
    frame_idx: int
    cell_pos: Tuple[float, float]          # (row, col) centroid
    bbox: Tuple[int, int, int, int]        # (min_row, min_col, max_row, max_col)
    crop: np.ndarray                       # float32 (H, W, 3)
    mask_label: int                        # CellPose instance ID
    gt_cell_id: Optional[int] = None      # set in eval mode


# --------------------------------------------------------------------------- #
# CellPose wrapper
# --------------------------------------------------------------------------- #

class CellDetector:
    """
    Wraps CellPose (cyto2 model) to segment frames and extract crops.

    Parameters
    ----------
    crop_size : output crop size in pixels (square), applied via padding/cropping
    min_area  : minimum cell area in pixels; detections below this are discarded
    use_gpu   : pass to CellPose
    """

    def __init__(
        self,
        crop_size: int = 96,
        min_area: int = 500,
        use_gpu: bool = False,
    ):
        self.crop_size = crop_size
        self.min_area = min_area

        import cellpose
        from cellpose import models
        from importlib.metadata import version as pkg_version

        try:
            major = int(pkg_version('cellpose').split('.')[0])
        except Exception:
            major = 3  # safe fallback

        if major >= 4:
            # cellpose 4.x: CellposeModel() with no model_type IS CellposeSAM
            self.model = models.CellposeModel(gpu=use_gpu)
            self._backend = 'sam'
        elif hasattr(models, 'CellposeModel'):
            self.model = models.CellposeModel(model_type="cyto3", gpu=use_gpu)
            self._backend = 'cp3'
        else:
            self.model = models.Cellpose(model_type="cyto2", gpu=use_gpu)
            self._backend = 'cp2'
        print(f'[detector] cellpose {cellpose.__version__} → backend: {self._backend}')

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #

    def segment(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Run CellPose/CellposeSAM on a single (H, W, 3) float32 frame.

        CellposeSAM accepts a 3-channel RGB image directly.
        Older backends receive the grayscale channel only.

        Returns a uint32 label mask (H, W) where 0 = background.
        """
        if self._backend == 'sam':
            # CellposeSAM: pass the full RGB image, returns (masks, flows, styles)
            # Convert float32 [0,1] → uint8 [0,255] as SAM expects uint8
            img_uint8 = (rgb_frame * 255).clip(0, 255).astype(np.uint8)
            result = self.model.eval(
                [img_uint8],
                diameter=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
        elif self._backend == 'cp3':
            gray = rgb_frame[:, :, 0]
            result = self.model.eval(
                [gray],
                diameter=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
        else:
            # cellpose 2.x: returns (masks, flows, styles, diams)
            gray = rgb_frame[:, :, 0]
            result = self.model.eval(
                [gray],
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
        masks = result[0]
        return masks[0].astype(np.uint32)

    # ------------------------------------------------------------------ #
    # Crop extraction
    # ------------------------------------------------------------------ #

    def _extract_crop(self, rgb_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and resize a crop from an RGB frame given a bounding box.

        The crop is padded to a square of `crop_size x crop_size` pixels,
        preserving aspect ratio.
        """
        min_r, min_c, max_r, max_c = bbox
        h = max_r - min_r
        w = max_c - min_c

        # Expand to square around centroid
        side = max(h, w, self.crop_size)
        cr = (min_r + max_r) // 2
        cc = (min_c + max_c) // 2
        half = side // 2

        H, W = rgb_frame.shape[:2]
        r0 = max(0, cr - half)
        r1 = min(H, cr + half)
        c0 = max(0, cc - half)
        c1 = min(W, cc + half)

        crop = rgb_frame[r0:r1, c0:c1].copy()

        # Resize to crop_size x crop_size using simple interpolation
        from skimage.transform import resize as sk_resize
        crop = sk_resize(
            crop,
            (self.crop_size, self.crop_size, 3),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

        return crop

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect_frame(
        self,
        frame_idx: int,
        rgb_frame: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> List[DetectedCell]:
        """
        Segment one frame and return a list of DetectedCell objects.

        Parameters
        ----------
        frame_idx : int index of the frame
        rgb_frame : float32 (H, W, 3) normalised to [0,1]
        gt_mask   : optional uint16 (H, W) GT tracking mask — used to assign
                    gt_cell_id to each detection (eval mode only)
        """
        seg_mask = self.segment(rgb_frame)
        props = regionprops(seg_mask)

        cells: List[DetectedCell] = []
        for prop in props:
            if prop.label == 0:
                continue
            if prop.area < self.min_area:
                continue

            bbox = prop.bbox                             # (min_r, min_c, max_r, max_c)
            centroid = (prop.centroid[0], prop.centroid[1])
            crop = self._extract_crop(rgb_frame, bbox)

            gt_id: Optional[int] = None
            if gt_mask is not None:
                cr = int(round(centroid[0]))
                cc = int(round(centroid[1]))
                cr = np.clip(cr, 0, gt_mask.shape[0] - 1)
                cc = np.clip(cc, 0, gt_mask.shape[1] - 1)
                gt_id_at_center = int(gt_mask[cr, cc])
                gt_id = gt_id_at_center if gt_id_at_center > 0 else None

            cells.append(DetectedCell(
                frame_idx=frame_idx,
                cell_pos=centroid,
                bbox=bbox,
                crop=crop,
                mask_label=prop.label,
                gt_cell_id=gt_id,
            ))

        return cells

    def detect_sequence(
        self,
        seq_data: SequenceData,
        eval_mode: bool = False,
    ) -> Dict[int, List[DetectedCell]]:
        """
        Run detection on every frame in a SequenceData object.

        Parameters
        ----------
        seq_data   : SequenceData container
        eval_mode  : if True and seq_data has a gt_tra_dir, load GT masks to
                     populate gt_cell_id on each DetectedCell

        Returns
        -------
        {frame_idx: [DetectedCell, ...]}
        """
        results: Dict[int, List[DetectedCell]] = {}

        for frame_idx, rgb_frame in seq_data.frames:
            gt_mask: Optional[np.ndarray] = None
            if eval_mode and seq_data.gt_tra_dir is not None:
                gt_mask = load_gt_mask(seq_data.gt_tra_dir, frame_idx)

            cells = self.detect_frame(frame_idx, rgb_frame, gt_mask=gt_mask)
            results[frame_idx] = cells

        return results


# --------------------------------------------------------------------------- #
# Spatial matching utilities (used by train.py to build positive pairs)
# --------------------------------------------------------------------------- #

def compute_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two (min_r, min_c, max_r, max_c) bounding boxes."""
    r0 = max(bbox_a[0], bbox_b[0])
    c0 = max(bbox_a[1], bbox_b[1])
    r1 = min(bbox_a[2], bbox_b[2])
    c1 = min(bbox_a[3], bbox_b[3])

    inter_h = max(0, r1 - r0)
    inter_w = max(0, c1 - c0)
    inter = inter_h * inter_w

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def centroid_distance(cell_a: DetectedCell, cell_b: DetectedCell) -> float:
    """Euclidean distance between two cell centroids (in pixels)."""
    dr = cell_a.cell_pos[0] - cell_b.cell_pos[0]
    dc = cell_a.cell_pos[1] - cell_b.cell_pos[1]
    return float(np.sqrt(dr ** 2 + dc ** 2))


def match_cells_across_frames(
    cells_t: List[DetectedCell],
    cells_t1: List[DetectedCell],
    iou_threshold: float = 0.2,
    max_dist_px: float = 50.0,
) -> List[Tuple[int, int]]:
    """
    Build positive pairs between consecutive frames t and t+1 without GT labels.

    Strategy (greedy nearest-neighbour on centroid distance, filtered by IoU):
      1. For each cell in t, find the closest cell in t+1 by centroid distance.
      2. Accept the match if distance < max_dist_px AND IoU > iou_threshold.
      3. Each cell in t+1 can only be matched once (greedy assignment).

    Returns
    -------
    list of (index_in_cells_t, index_in_cells_t1) matched pairs
    """
    if not cells_t or not cells_t1:
        return []

    # Build distance matrix
    n, m = len(cells_t), len(cells_t1)
    dist_mat = np.full((n, m), np.inf)
    iou_mat = np.zeros((n, m))

    for i, ca in enumerate(cells_t):
        for j, cb in enumerate(cells_t1):
            d = centroid_distance(ca, cb)
            dist_mat[i, j] = d
            iou_mat[i, j] = compute_iou(ca.bbox, cb.bbox)

    matched: List[Tuple[int, int]] = []
    used_j: set = set()

    # Sort anchors by their best available distance
    order = sorted(range(n), key=lambda i: dist_mat[i].min())

    for i in order:
        # Find best j not yet used
        candidates = sorted(
            [(dist_mat[i, j], j) for j in range(m) if j not in used_j],
            key=lambda x: x[0],
        )
        if not candidates:
            break
        best_dist, best_j = candidates[0]

        if best_dist > max_dist_px:
            continue
        if iou_mat[i, best_j] < iou_threshold and best_dist > max_dist_px * 0.5:
            # Allow pure-distance match for cells that moved but didn't overlap
            # (e.g. cells that shifted at frame boundary)
            if best_dist > max_dist_px * 0.5:
                continue

        matched.append((i, best_j))
        used_j.add(best_j)

    return matched
