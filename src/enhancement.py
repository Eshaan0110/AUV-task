"""Underwater image enhancement.

A single-frame enhancement routine used by the video pipeline. The algorithm
is intentionally classical (no learning), so it runs in real time on modest
hardware and is easy to reason about on an AUV.

Pipeline for one BGR frame:
    1. Per-channel min-max normalisation to stretch the histogram.
    2. Convert to CIE L*a*b* so brightness is separated from colour.
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to
       the L (luminance) channel only. Touching L keeps hues stable while
       lifting local contrast in the murky midtones typical of underwater
       footage.
    4. Merge the channels back and convert to BGR for display / writing.
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_frame(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (10, 50),
) -> np.ndarray:
    """Enhance a single underwater BGR frame.

    Parameters
    ----------
    frame:
        Input frame in OpenCV BGR order.
    clip_limit:
        CLAHE contrast clipping threshold. Higher values give more
        aggressive local contrast.
    tile_grid_size:
        CLAHE tile grid ``(width, height)``. Smaller tiles adapt more
        locally at the cost of blockiness.

    Returns
    -------
    np.ndarray
        Enhanced frame in BGR order, same shape and dtype as ``frame``.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    c0, c1, c2 = cv2.split(img_rgb)
    c0 = cv2.normalize(c0, None, 0, 255, cv2.NORM_MINMAX)
    c1 = cv2.normalize(c1, None, 0, 255, cv2.NORM_MINMAX)
    c2 = cv2.normalize(c2, None, 0, 255, cv2.NORM_MINMAX)
    img_stretched = cv2.merge((c0, c1, c2))

    lab = cv2.cvtColor(img_stretched, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalised = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_equalised, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)
