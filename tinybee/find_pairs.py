"""Find potentials pairs.

frac = 20 / tgt_len, it = 3
lowess Locally Weighted Scatterplot Smoothing (LOWESS)

see also statsmodels.nonparametric.kernel_regression
"""
# pylint: disable=broad-except, line-too-long

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import savgol_filter
from logzero import logger


def find_pairs(
    arr1: Union[List[float], np.array],
    window_length: int = 10,  # odd int
    polyorder: int = 1,
    thr: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """Find pairs via savgol-filter."""
    if isinstance(arr1, list):
        try:
            arr = np.array(arr1)
        except Exception as exc:
            logger.erorr(exc)
            raise SystemExit(1) from exc
    else:
        arr = arr1.copy()
    _, len1 = arr.shape

    if not isinstance(window_length, int):
        try:
            window_length = int(window_length)
        except Exception as exc:
            logger.erorr(exc)
            raise SystemExit(1) from exc
    if window_length < 0:
        window_length = 1

    if window_length > len1:
        window_length = len1

    if window_length % 2 == 0:
        window_length += 1  # window_length must be odd

    yargmax = arr.argmax(axis=0)
    ymax = arr.max(axis=0)
    mean_, std_ = ymax.mean(), ymax.std()

    # yhat = savgol_filter(y, 13, 1)
    yhat = savgol_filter(yargmax, window_length, polyorder)

    idx_idy_val = [[idx, idy, val] for idx, (idy, val) in enumerate(zip(yargmax, ymax))]

    if thr is None:
        thr = mean_ - .68 * std_
    if thr < 0:
        thr = mean_ - 2 * std_

    _ = [[idx, idy, val] for idx, idy, val in idx_idy_val if val / (1 + abs(idy - yhat[idx])**2) > thr]

    # return np.array(_)
    return _
