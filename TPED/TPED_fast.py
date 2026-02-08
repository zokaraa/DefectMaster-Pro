# -*- coding: utf-8 -*-
"""
TPED.py
========
Topology-Preserving Elastic Deformation (TPED) as a reusable augmentation module.
(Fixed: Corrected Jacobian determinant calculation for inverse warping)

This version focuses on SPEED:
- Replaces scipy.interpolate.Rbf evaluation with an explicit Thin-Plate Spline (TPS) solve + fast grid evaluation.
- Avoids materializing a full det(J) map when only min det(J) is needed.
- Optional Numba JIT (if installed) for C-like speed on the hot loops (TPS evaluation + min det(J)).
  Install: pip install numba

Usage:
    import TPED as TP
    img_aug = TP.TPED(img, bboxes=bboxes)  # default: fold-free, sigma=0.1, delta_max=0.1 (relative)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, distance_transform_edt
except Exception as e:  # pragma: no cover
    raise ImportError("TPED requires SciPy. Install: pip install scipy") from e

# Optional: faster hot loops
try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    njit = None  # type: ignore
    prange = range  # type: ignore

# Optional: PIL / torch round-trip support
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None
    _HAS_PIL = False

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

# -------- speed cache --------
_GRID_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_BORDER_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _get_rng(rng: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.Generator):
        return rng
    raise TypeError("rng must be None, int, or np.random.Generator")


def _precompute_grid(H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rows(y), cols(x), coords(x,y)"""
    key = (H, W)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    rows, cols = np.indices((H, W), dtype=np.float32)
    coords = np.column_stack([cols.ravel(), rows.ravel()]).astype(np.float32)  # (x,y)
    _GRID_CACHE[key] = (rows, cols, coords)
    return rows, cols, coords


def _as_bbox_list(bboxes: Optional[Union[BBox, Sequence[BBox], np.ndarray]], H: int, W: int) -> List[BBox]:
    """Normalize bbox input to a list of clipped, valid boxes in pixel coordinates."""
    if bboxes is None:
        return []
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()

    # single bbox
    if isinstance(bboxes, (list, tuple)) and len(bboxes) == 4 and not isinstance(bboxes[0], (list, tuple, np.ndarray)):
        bboxes_list = [tuple(map(float, bboxes))]  # type: ignore
    else:
        bboxes_list = [tuple(map(float, b)) for b in (bboxes or [])]  # type: ignore

    out: List[BBox] = []
    for (x1, y1, x2, y2) in bboxes_list:
        x1 = float(np.clip(x1, 0, W - 1))
        x2 = float(np.clip(x2, 0, W - 1))
        y1 = float(np.clip(y1, 0, H - 1))
        y2 = float(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2))
    return out


def _base_length(bboxes: List[BBox], H: int, W: int) -> float:
    """base_length = min bbox side; if no bbox -> min(H,W)"""
    if not bboxes:
        return float(min(H, W))
    mins = [min(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes]
    base = float(min(mins)) if mins else float(min(H, W))
    return max(1.0, base)


def _make_border_points(W: int, H: int, edge_samples: int = 5) -> np.ndarray:
    """Border anchor points (x,y) with zero displacement."""
    key = (W, H, edge_samples)
    if key in _BORDER_CACHE:
        return _BORDER_CACHE[key]
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    xs = np.linspace(0, W - 1, edge_samples)[1:-1]
    ys = np.linspace(0, H - 1, edge_samples)[1:-1]
    top = np.array([[x, 0] for x in xs], dtype=np.float32)
    bottom = np.array([[x, H - 1] for x in xs], dtype=np.float32)
    left = np.array([[0, y] for y in ys], dtype=np.float32)
    right = np.array([[W - 1, y] for y in ys], dtype=np.float32)
    pts = np.concatenate([corners, top, bottom, left, right], axis=0)
    _BORDER_CACHE[key] = pts
    return pts


def _sample_points_in_box(rng: np.random.Generator, bbox: BBox, n: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    xs = rng.uniform(x1, x2, n).astype(np.float32)
    ys = rng.uniform(y1, y2, n).astype(np.float32)
    return np.stack([xs, ys], axis=1)  # (n,2) (x,y)


def _sample_points_in_image(rng: np.random.Generator, H: int, W: int, n: int) -> np.ndarray:
    xs = rng.uniform(0, W - 1, n).astype(np.float32)
    ys = rng.uniform(0, H - 1, n).astype(np.float32)
    return np.stack([xs, ys], axis=1)


def _sample_displacements_keep_inside(
    rng: np.random.Generator,
    pts_xy: np.ndarray,  # (n,2)
    delta_abs: float,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample displacements while guaranteeing each control point stays inside the image:
        x_i + dx_i in [0, W-1], y_i + dy_i in [0, H-1].
    """
    x = pts_xy[:, 0].astype(np.float32)
    y = pts_xy[:, 1].astype(np.float32)

    dx_low = np.maximum(-delta_abs, -x)
    dx_high = np.minimum(delta_abs, (W - 1) - x)
    dy_low = np.maximum(-delta_abs, -y)
    dy_high = np.minimum(delta_abs, (H - 1) - y)

    # fall back to 0 if a point has no feasible range (should be rare)
    bad_dx = dx_low > dx_high
    bad_dy = dy_low > dy_high

    dx = rng.uniform(dx_low, dx_high).astype(np.float32)
    dy = rng.uniform(dy_low, dy_high).astype(np.float32)

    dx[bad_dx] = 0.0
    dy[bad_dy] = 0.0
    return dx, dy


# ============================================================
# TPS core (explicit solve + evaluation)
# ============================================================
_TPS_EPS = 1e-6


def _tps_kernel_r2(r2: np.ndarray, eps: float = _TPS_EPS) -> np.ndarray:
    """Thin-plate spline kernel using squared distance: phi(r) = r^2 log(r^2 + eps)."""
    return r2 * np.log(r2 + eps)


def _tps_fit(ctrl_xy: np.ndarray, values: np.ndarray, smooth: float = 0.0, eps: float = _TPS_EPS):
    """
    Fit TPS coefficients for one scalar field.
    u(x,y) = a0 + a1*x + a2*y + sum_i w_i * phi(||(x,y)-(x_i,y_i)||)
    """
    ctrl = np.asarray(ctrl_xy, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    N = int(ctrl.shape[0])
    if N < 3:
        # degenerate: no meaningful TPS; return zero field
        w = np.zeros((N,), dtype=np.float64)
        a = np.zeros((3,), dtype=np.float64)
        return w, a

    x = ctrl[:, 0]
    y = ctrl[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    r2 = dx * dx + dy * dy
    K = _tps_kernel_r2(r2, eps=eps)
    if smooth and smooth > 0:
        K = K + float(smooth) * np.eye(N, dtype=np.float64)

    P = np.stack([np.ones(N, dtype=np.float64), x, y], axis=1)  # (N,3)

    L = np.zeros((N + 3, N + 3), dtype=np.float64)
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.T

    rhs = np.zeros((N + 3,), dtype=np.float64)
    rhs[:N] = v

    params = np.linalg.solve(L, rhs)
    w = params[:N]
    a = params[N:]  # a0, a1, a2
    return w, a


if _HAS_NUMBA:
    import math

    @njit(parallel=True, fastmath=True)
    def _tps_eval_numba(cols: np.ndarray, rows: np.ndarray,
                       ctrl_x: np.ndarray, ctrl_y: np.ndarray,
                       w: np.ndarray, a: np.ndarray, eps: float) -> np.ndarray:
        """
        Evaluate TPS scalar field on a full grid (C-like speed).
        """
        H, W = cols.shape
        out = np.empty((H, W), dtype=np.float32)
        N = ctrl_x.shape[0]
        a0 = a[0]; a1 = a[1]; a2 = a[2]
        for yy in prange(H):
            for xx in range(W):
                x = cols[yy, xx]
                y = rows[yy, xx]
                s = a0 + a1 * x + a2 * y
                for i in range(N):
                    dx = x - ctrl_x[i]
                    dy = y - ctrl_y[i]
                    r2 = dx * dx + dy * dy
                    s += w[i] * (r2 * math.log(r2 + eps))
                out[yy, xx] = np.float32(s)
        return out

    @njit(parallel=True, fastmath=True)
    def _min_detJ_numba(u_x: np.ndarray, u_y: np.ndarray) -> float:
        """
        Compute min det(J) for phi(x,y)=(x-u_x, y-u_y) -- Inverse Warp.
        J = I - grad(u).
        """
        H, W = u_x.shape
        min_det = 1e30

        for y in prange(H):
            ym1 = y - 1
            yp1 = y + 1
            if ym1 < 0:
                ym1 = 0
            if yp1 >= H:
                yp1 = H - 1

            for x in range(W):
                xm1 = x - 1
                xp1 = x + 1
                if xm1 < 0:
                    xm1 = 0
                if xp1 >= W:
                    xp1 = W - 1

                # dux/dx, dux/dy
                dux_dx = 0.5 * (u_x[y, xp1] - u_x[y, xm1]) if xp1 != xm1 else 0.0
                dux_dy = 0.5 * (u_x[yp1, x] - u_x[ym1, x]) if yp1 != ym1 else 0.0

                # duy/dx, duy/dy
                duy_dx = 0.5 * (u_y[y, xp1] - u_y[y, xm1]) if xp1 != xm1 else 0.0
                duy_dy = 0.5 * (u_y[yp1, x] - u_y[ym1, x]) if yp1 != ym1 else 0.0

                # FIXED: Since warp is x_src = x - u, Jacobian is I - grad(u)
                det = (1.0 - dux_dx) * (1.0 - duy_dy) - (dux_dy * duy_dx)
                
                if det < min_det:
                    min_det = det

        return float(min_det)


def _tps_field(
    H: int,
    W: int,
    ctrl_xy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    grid_cache,
    smooth: float = 0.0,
    post_smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build TPS displacement field u_x/u_y.
    - Uses explicit TPS solve.
    - Optional Numba acceleration for grid evaluation (if installed).
    """
    rows, cols, _coords = grid_cache

    w_x, a_x = _tps_fit(ctrl_xy, dx, smooth=smooth, eps=_TPS_EPS)
    w_y, a_y = _tps_fit(ctrl_xy, dy, smooth=smooth, eps=_TPS_EPS)

    ctrl_x = np.asarray(ctrl_xy[:, 0], dtype=np.float32)
    ctrl_y = np.asarray(ctrl_xy[:, 1], dtype=np.float32)

    if _HAS_NUMBA:
        u_x = _tps_eval_numba(cols, rows, ctrl_x, ctrl_y, w_x, a_x, _TPS_EPS)
        u_y = _tps_eval_numba(cols, rows, ctrl_x, ctrl_y, w_y, a_y, _TPS_EPS)
    else:
        # Vectorized fallback (slower, but no extra dependency)
        X = cols.astype(np.float64)
        Y = rows.astype(np.float64)

        u_x = (a_x[0] + a_x[1] * X + a_x[2] * Y).astype(np.float64)
        u_y = (a_y[0] + a_y[1] * X + a_y[2] * Y).astype(np.float64)

        for i in range(ctrl_xy.shape[0]):
            dxg = X - float(ctrl_x[i])
            dyg = Y - float(ctrl_y[i])
            r2 = dxg * dxg + dyg * dyg
            k = _tps_kernel_r2(r2, eps=_TPS_EPS)
            u_x += w_x[i] * k
            u_y += w_y[i] * k

        u_x = u_x.astype(np.float32)
        u_y = u_y.astype(np.float32)

    if post_smooth_sigma and post_smooth_sigma > 0:
        u_x = gaussian_filter(u_x, sigma=float(post_smooth_sigma))
        u_y = gaussian_filter(u_y, sigma=float(post_smooth_sigma))

    # Hard-enforce boundary anchoring (important if you enable post_smooth_sigma or ROI attenuation)
    u_x[0, :] = 0.0
    u_x[-1, :] = 0.0
    u_x[:, 0] = 0.0
    u_x[:, -1] = 0.0

    u_y[0, :] = 0.0
    u_y[-1, :] = 0.0
    u_y[:, 0] = 0.0
    u_y[:, -1] = 0.0

    return u_x.astype(np.float32), u_y.astype(np.float32)


# ============================================================
# ROI attenuation (optional localization around bboxes)
# ============================================================
def _roi_attenuation(H: int, W: int, bboxes: List[BBox], decay_sigma: float) -> np.ndarray:
    """Localize deformation near bboxes: inside=1, outside decays with distance."""
    if not bboxes or decay_sigma <= 0:
        return np.ones((H, W), dtype=np.float32)

    mask = np.zeros((H, W), dtype=bool)
    for (x1, y1, x2, y2) in bboxes:
        x1i = int(np.floor(x1)); x2i = int(np.ceil(x2))
        y1i = int(np.floor(y1)); y2i = int(np.ceil(y2))
        x1i = max(0, min(W - 1, x1i)); x2i = max(0, min(W - 1, x2i))
        y1i = max(0, min(H - 1, y1i)); y2i = max(0, min(H - 1, y2i))
        if x2i <= x1i or y2i <= y1i:
            continue
        mask[y1i:y2i + 1, x1i:x2i + 1] = True

    if not mask.any():
        return np.ones((H, W), dtype=np.float32)

    dist = distance_transform_edt(~mask).astype(np.float32)
    att = np.exp(-(dist ** 2) / (2.0 * (float(decay_sigma) ** 2))).astype(np.float32)
    return att


# ============================================================
# Fold-free check + inverse warp (nearest neighbor)
# ============================================================
def _min_det_jacobian(u_x: np.ndarray, u_y: np.ndarray) -> float:
    """Return min det(J) for phi=(x-u_x, y-u_y)."""
    if _HAS_NUMBA:
        return float(_min_detJ_numba(u_x.astype(np.float32), u_y.astype(np.float32)))

    # Numpy fallback: uses gradient (allocates a few arrays)
    dux_dy, dux_dx = np.gradient(u_x)
    duy_dy, duy_dx = np.gradient(u_y)
    
    # FIXED: Inverse warp -> I - grad(u)
    detJ = (1.0 - dux_dx) * (1.0 - duy_dy) - (dux_dy * duy_dx)
    return float(detJ.min())


def _warp_inverse_nearest(arr: np.ndarray, u_x: np.ndarray, u_y: np.ndarray, grid_cache) -> np.ndarray:
    """Inverse warp NN: out(y,x)=in(y-u_y, x-u_x)."""
    H, W = arr.shape[:2]
    rows, cols, _coords = grid_cache
    src_x = cols - u_x
    src_y = rows - u_y

    xi = np.rint(src_x).astype(np.int32)
    yi = np.rint(src_y).astype(np.int32)
    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    if arr.ndim == 2:
        return arr[yi, xi]
    return arr[yi, xi, :]


def _to_numpy(img: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert input to numpy float32 + meta for round-trip."""
    if _HAS_PIL and isinstance(img, Image.Image):
        arr0 = np.array(img)
        meta = {"kind": "pil", "mode": img.mode, "dtype": arr0.dtype}
        return arr0.astype(np.float32, copy=False), meta

    if _HAS_TORCH and torch is not None and torch.is_tensor(img):
        t = img.detach().cpu()
        arr0 = t.numpy()
        meta = {"kind": "torch", "dtype": t.dtype, "shape": tuple(t.shape)}
        if arr0.ndim == 3 and arr0.shape[0] in (1, 3, 4):  # CHW
            meta["channel_first"] = True
            arr0 = np.transpose(arr0, (1, 2, 0))
        else:
            meta["channel_first"] = False
        return arr0.astype(np.float32, copy=False), meta

    arr0 = np.asarray(img)
    meta = {"kind": "numpy", "dtype": arr0.dtype, "shape": tuple(arr0.shape)}
    return arr0.astype(np.float32, copy=False), meta


def _from_numpy(arr: np.ndarray, meta: Dict[str, Any]):
    """Convert numpy back to original type."""
    kind = meta.get("kind", "numpy")
    dtype = meta.get("dtype", None)

    if kind == "numpy":
        if dtype is not None and np.dtype(dtype).kind in ("u", "i"):
            info = np.iinfo(dtype)
            return np.clip(arr, info.min, info.max).astype(dtype)
        return arr.astype(dtype) if dtype is not None else arr

    if kind == "pil":
        if not _HAS_PIL:
            raise RuntimeError("PIL not available, cannot return PIL image.")
        if dtype is not None and np.dtype(dtype).kind in ("u", "i"):
            info = np.iinfo(dtype)
            arr_u = np.clip(arr, info.min, info.max).astype(dtype)
        else:
            arr_u = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_u)

    if kind == "torch":
        if not _HAS_TORCH or torch is None:
            raise RuntimeError("torch not available, cannot return torch tensor.")
        out = arr
        if bool(meta.get("channel_first", True)) and out.ndim == 3:
            out = np.transpose(out, (2, 0, 1))
        t = torch.from_numpy(out)
        if "dtype" in meta and meta["dtype"] is not None:
            try:
                t = t.to(dtype=meta["dtype"])
            except Exception:
                pass
        return t

    return arr


@dataclass
class TPEDInfo:
    success: bool
    fold_free: bool
    min_detJ: float
    used_sigma_abs: float
    used_delta_abs: float
    n_control_points: int
    stage: int
    tries: int
    control_points_xy: np.ndarray  # (n,2) non-border
    dx: np.ndarray
    dy: np.ndarray


def TPED(
    img: Any,
    bboxes: Optional[Union[BBox, Sequence[BBox], np.ndarray]] = None,
    sigma: float = 0.1,
    delta_max: float = 0.1,
    max_control_point_number: int = 3,
    min_control_point_number: int = 2,
    *,
    relative: bool = True,
    localize: Optional[bool] = None,
    det_threshold: float = 1e-3,
    max_tries: int = 10,
    halve_delta_on_fail: bool = True,
    is_fold_free: bool = True,
    rbf_smooth: float = 0.0,
    edge_samples: int = 5,
    rng: Optional[Union[int, np.random.Generator]] = None,
    return_info: bool = False,
):
    """
    Main API: TP.TPED(img, bboxes=..., sigma=..., delta_max=...)

    Notes:
    - sigma/delta_max default to relative=True:
        sigma_abs = sigma * base_length
        delta_abs = delta_max * base_length
      base_length = min bbox side; if no bbox -> min(H,W)
    - fold-free enforcement:
        try max_tries times; if all fail, halve delta_max and retry max_tries times; still fail -> return original
    """
    rng_g = _get_rng(rng)
    arr, meta = _to_numpy(img)

    if arr.ndim not in (2, 3):
        raise ValueError("img must be 2D (H,W) or 3D (H,W,C)/(C,H,W).")

    H, W = arr.shape[:2]
    grid_cache = _precompute_grid(H, W)

    bbs = _as_bbox_list(bboxes, H, W)
    if localize is None:
        localize = True if bbs else False

    base_len = _base_length(bbs, H, W)
    sigma_abs = float(sigma) * base_len if relative else float(sigma)
    delta_abs0 = float(delta_max) * base_len if relative else float(delta_max)
    delta_abs0 = max(0.0, delta_abs0)

    # Control points per bbox: 2 ~ min(5, max_control_point_number)
    if max_control_point_number < 2:
        max_control_point_number = 2
    max_cp = int(min(5, max_control_point_number))
    min_cp = int(max(2, min_control_point_number))
    if max_cp < min_cp:
        max_cp = min_cp

    border_pts = _make_border_points(W, H, edge_samples=edge_samples)
    n_border = int(border_pts.shape[0])

    def try_once(delta_abs: float) -> Tuple[np.ndarray, TPEDInfo]:
        ctrl_pts_list = []
        dx_list = []
        dy_list = []

        if bbs:
            for bb in bbs:
                n_ctrl = int(rng_g.integers(min_cp, max_cp + 1))
                pts = _sample_points_in_box(rng_g, bb, n_ctrl)
                dx, dy = _sample_displacements_keep_inside(rng_g, pts, delta_abs, H, W)
                ctrl_pts_list.append(pts)
                dx_list.append(dx)
                dy_list.append(dy)
        else:
            n_ctrl = int(rng_g.integers(min_cp, max_cp + 1))
            pts = _sample_points_in_image(rng_g, H, W, n_ctrl)
            dx, dy = _sample_displacements_keep_inside(rng_g, pts, delta_abs, H, W)
            ctrl_pts_list.append(pts)
            dx_list.append(dx)
            dy_list.append(dy)

        ctrl_nb = np.vstack(ctrl_pts_list).astype(np.float32) if ctrl_pts_list else np.zeros((0, 2), np.float32)
        dx_nb = np.concatenate(dx_list).astype(np.float32) if dx_list else np.zeros((0,), np.float32)
        dy_nb = np.concatenate(dy_list).astype(np.float32) if dy_list else np.zeros((0,), np.float32)

        ctrl_all = np.vstack([border_pts, ctrl_nb]).astype(np.float32)
        dx_all = np.hstack([np.zeros(n_border, dtype=np.float32), dx_nb])
        dy_all = np.hstack([np.zeros(n_border, dtype=np.float32), dy_nb])

        # TPS field. 
        # FIXED: Scaled down post_smooth_sigma by 50.0 to prevent border destruction
        u_x, u_y = _tps_field(
            H, W,
            ctrl_xy=ctrl_all,
            dx=dx_all,
            dy=dy_all,
            grid_cache=grid_cache,
            smooth=rbf_smooth,
            post_smooth_sigma=float(max(0.0, sigma_abs / 50.0)), 
        )

        # Optional: localize deformation around bboxes
        if localize and bbs:
            decay_sigma = max(1.0, 2.0 * sigma_abs) if sigma_abs > 0 else max(1.0, 0.2 * base_len)
            att = _roi_attenuation(H, W, bbs, decay_sigma=float(decay_sigma))
            u_x *= att
            u_y *= att

            # re-enforce boundary anchoring after attenuation
            u_x[0, :] = 0.0; u_x[-1, :] = 0.0; u_x[:, 0] = 0.0; u_x[:, -1] = 0.0
            u_y[0, :] = 0.0; u_y[-1, :] = 0.0; u_y[:, 0] = 0.0; u_y[:, -1] = 0.0

        min_det = _min_det_jacobian(u_x, u_y)
        fold_free = (min_det > det_threshold)

        aug = _warp_inverse_nearest(arr, u_x, u_y, grid_cache=grid_cache)

        info = TPEDInfo(
            success=True,
            fold_free=fold_free,
            min_detJ=float(min_det),
            used_sigma_abs=float(sigma_abs),
            used_delta_abs=float(delta_abs),
            n_control_points=int(ctrl_nb.shape[0]),
            stage=0,
            tries=0,
            control_points_xy=ctrl_nb,
            dx=dx_nb,
            dy=dy_nb,
        )
        return aug, info

    # stage0: delta_abs0; stage1(optional): delta_abs0/2
    stages = [delta_abs0]
    if halve_delta_on_fail and delta_abs0 > 0:
        stages.append(delta_abs0 * 0.5)

    last_info = TPEDInfo(
        success=False,
        fold_free=False,
        min_detJ=float("-inf"),
        used_sigma_abs=float(sigma_abs),
        used_delta_abs=float(delta_abs0),
        n_control_points=0,
        stage=-1,
        tries=0,
        control_points_xy=np.zeros((0, 2), np.float32),
        dx=np.zeros((0,), np.float32),
        dy=np.zeros((0,), np.float32),
    )

    for stage_idx, delta_abs in enumerate(stages):
        for t in range(int(max_tries)):
            aug, info = try_once(float(delta_abs))
            info.stage = stage_idx
            info.tries = t + 1
            last_info = info

            if (not is_fold_free) or info.fold_free:
                out_img = _from_numpy(aug, meta)
                if return_info:
                    info.success = True
                    return out_img, info
                return out_img

    # All failed -> return original image
    out_img = _from_numpy(arr, meta)
    last_info.success = False
    last_info.fold_free = False
    if return_info:
        return out_img, last_info
    return out_img


class TPEDTransform:
    """
    torchvision-style wrapper:
        t = TP.TPEDTransform(...)
        img2 = t(img, bboxes=bboxes)

    Note:
      torchvision.transforms.Compose passes only img.
      If you need bbox-guided TPED, call TPED manually inside Dataset.__getitem__.
    """
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def __call__(self, img, bboxes=None):
        return TPED(img, bboxes=bboxes, **self.kwargs)