# Apply TPED to a single image and save a side-by-side figure.
# Requirements: numpy, scipy, matplotlib, pillow

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import TPED_fast as TP  # TPED.py should be in the same folder


def main():
    # ----------- user settings -----------
    input_path = "input.png"  # <- replace with your image path
    output_path = "tped_before_after.png"

    # Optional bbox examples (pixel coordinates):
    # bboxes = None
    # bboxes = (50, 60, 200, 220)
    # bboxes = [(50, 60, 200, 220), (260, 80, 360, 180)]
    bboxes = None

    # TPED parameters (relative by default)
    sigma = 0.10
    delta_max = 0.10
    # -------------------------------------

    img = Image.open(input_path).convert("L")  # grayscale for demonstration

    img_aug, info = TP.TPED(
        img,
        bboxes=bboxes,
        sigma=sigma,
        delta_max=delta_max,
        is_fold_free=True,
        return_info=True,
    )

    # Convert to numpy for plotting
    a = np.array(img)
    b = np.array(img_aug)

    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(a, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(b, cmap="gray")
    ax2.set_title(f"TPED (FF={info.fold_free}, min detJ={info.min_detJ:.2e})")
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", os.path.abspath(output_path))


if __name__ == "__main__":
    main()