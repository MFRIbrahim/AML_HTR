# ------------------------------------------------------------------------------- #
# ---From https://github.com/githubharald/DeslantImg/ reimplemented in Python3--- #
# ------------------------------------------------------------------------------- #
import math

import cv2
import numpy as np


class Result:
    def  __init__(self):
        """
        Data structure for internal usage.
        """
        self.sum_alpha = 0.0
        self.transform = None
        self.size = 0

    def __ge__(self, other):
        return self.sum_alpha >= other.sum_alpha


def deslant_image(img, bgcolor=255, alpha_vals=(-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)):
    """
    Deslant image by calculating its slope and then rotating it overcome the effect of that shift.
    Args:
        img:
        bgcolor:
        alpha_vals:

    Returns:

    """
    transform_to_float = np.max(img) <= 1
    if transform_to_float:
        img = (img * 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]

    _, img_bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = []
    for i in range(len(alpha_vals)):
        result = Result()
        alpha = alpha_vals[i]
        shift_x = np.max([-alpha * img_bw.shape[1], 0])
        result.size = (img_bw.shape[1] + math.ceil(abs(alpha*img_bw.shape[1])), img_bw.shape[0])
        result.transform = np.zeros((2,3), dtype=np.float)
        result.transform[0, 0] = 1
        result.transform[0, 1] = alpha
        result.transform[0, 2] = shift_x
        result.transform[1, 0] = 0
        result.transform[1, 1] = 1
        result.transform[1, 2] = 0

        img_sheared = cv2.warpAffine(img_bw, result.transform, result.size, cv2.INTER_NEAREST)

        for x in range(img_sheared.shape[1]):
            fg_indices = []
            for y in range(img_sheared.shape[0]):
                if img_sheared[y,x]:
                    fg_indices.append(y)
            if len(fg_indices) == 0:
                continue

            h_alpha = len(fg_indices)
            delta_y_alpha = fg_indices[-1] - fg_indices[0] + 1

            if h_alpha == delta_y_alpha:
                result.sum_alpha += h_alpha * h_alpha

        results.append(result)

    best_result = np.max(results)
    img_deslanted = cv2.warpAffine(img,
                                   best_result.transform,
                                   best_result.size,
                                   cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=bgcolor)

    if transform_to_float:
        img_deslanted = img_deslanted.astype(np.float)/255

    return img_deslanted
