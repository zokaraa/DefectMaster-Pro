# TPED: Topology-Preserving Elastic Deformation (Fold-Free TPS-RBF Augmentation)

## What is TPED?

**TPED (Topology-Preserving Elastic Deformation)** is a physics-inspired data augmentation method designed for microscopy-like patterns and other morphology-driven images. It generates realistic, smooth geometric variations while **preserving topology** by enforcing a **fold-free** constraint.

In practice, TPED constructs a smooth deformation field using **Thin-Plate Spline (TPS) Radial Basis Function interpolation** from a small set of randomly perturbed control points. A set of boundary anchor points is included with zero displacement to keep the image boundary stable.

TPED is particularly useful under **data scarcity** and **class imbalance** because it increases effective sample diversity without corrupting defect semantics.

---

## Core Principle (Mathematical Form)

Let the deformation be a mapping:

$phi(x, y) = (x + u_x(x, y), y + u_y(x, y))$

Given a grayscale image intensity rho(x, y), the augmented image is produced via **inverse warping**:

rho_aug(x, y) = rho(phi^{-1}(x, y)) ≈ rho(x - u_x(x, y), y - u_y(x, y))

In `TPED.py`, if (x - u_x, y - u_y) falls off-grid, it is sampled with **nearest-neighbor** lookup for simplicity and robustness.

---

## How the Elastic Field is Built

### 1) Control points (bbox-guided or global)

- If **bboxes** are provided, TPED samples control points **inside each bbox** (2–K points per bbox).
- If **bboxes are None**, TPED samples control points in the whole image.

### 2) Boundary anchor points

A fixed set of border points (corners + sampled edges) is added with **zero displacement** to stabilize image boundaries.

### 3) TPS-RBF interpolation

The displacement field is obtained by TPS-RBF interpolation from control point displacements:

- u_x(x, y) interpolated from (x_i, y_i) → dx_i
- u_y(x, y) interpolated from (x_i, y_i) → dy_i

An optional Gaussian smoothing step is applied to ensure geometric plausibility.

### 4) Optional localization around bboxes

If `localize=True` and bboxes exist, TPED applies a smooth attenuation so deformation is concentrated near bboxes and decays away from them.

---

## Fold-Free (Topology-Preserving) Enforcement

To avoid topology-breaking folding or flipping, TPED checks the Jacobian determinant:

J = [[1 + ∂u_x/∂x, ∂u_x/∂y],
     [∂u_y/∂x, 1 + ∂u_y/∂y]]

A deformation is considered fold-free if:

det(J) > 0  (or > det_threshold in practice)

If `is_fold_free=True` (default), TPED:

1. Tries up to `max_tries` times with the current `delta_max`
2. If still not fold-free, halves `delta_max` and retries `max_tries` times
3. If it still fails, returns the original image

This design ensures **safe-by-construction** data augmentation.

---

## TPED.py Usage

### Minimal usage

import TPED as TP
img_aug = TP.TPED(img)

### With a single bbox

import TPED as TP

bbox = (x1, y1, x2, y2)  # pixel coordinates
img_aug = TP.TPED(img, bboxes=bbox)

### With multiple bboxes

import TPED as TP

bboxes = [(x1, y1, x2, y2), (x1, y1, x2, y2)]
img_aug, info = TP.TPED(img, bboxes=bboxes, return_info=True)

print(info.fold_free, info.min_detJ, info.used_delta_abs, info.n_control_points)

---

## Important Parameters

- sigma (default: 0.1)
  Relative smoothing scale. Actual value:
  sigma_abs = sigma × base_length

- delta_max (default: 0.1)
  Relative maximum displacement. Actual value:
  delta_abs = delta_max × base_length

- base_length
  If bboxes exist: minimum side length among all bboxes.
  Otherwise: min(H, W).

- max_control_point_number (default: 3)
  Each bbox samples 2 to min(5, max_control_point_number) control points.

- is_fold_free (default: True)
  Enforces fold-free deformation via Jacobian determinant checking.

- det_threshold (default: 1e-3)
  Fold-free condition: min(det(J)) > det_threshold.

---

## Practical Tips

1. For microscopy defect data, provide bboxes when available.
2. If no bboxes are available, TPED still works as a global elastic augmentation.
3. If deformation is too strong, reduce delta_max (e.g., 0.05).
4. If deformation is too smooth, reduce sigma (e.g., 0.05).
5. For speed, reuse the same Python process; grid caching is handled internally.

---

## Output Types

TPED attempts to return the same type as the input:

- PIL.Image → PIL.Image
- numpy.ndarray → numpy.ndarray
- torch.Tensor → torch.Tensor (best effort)

---

## License / Citation

If you use TPED in academic work, please cite the corresponding paper or repository.
