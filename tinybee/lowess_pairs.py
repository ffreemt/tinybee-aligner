"""Gen paris based on lowess.

cf savgol find_pairs
cf sklearn.kernelreg
"""
# pylint: disable=broad-except, too-many-locals, duplicate-code

from typing import List, Optional, Tuple, Union

import numpy as np
import statsmodels.api as sm

from logzero import logger


def lowess_pairs(
    arr1: Union[List[float], np.ndarray],
    frac: Optional[float] = None,  # default to 20/arr.shape[1],
    thr: Optional[float] = None,
    **kwargs,
) -> List[Tuple[int, int, float]]:
    """Gen pairs via lowess.

    arr =
    """
    lowess = sm.nonparametric.lowess
    if isinstance(arr1, list):
        try:
            arr = np.array(arr1)
        except Exception as exc:
            logger.debug(exc)
            raise SystemExit(1) from exc
    else:
        arr = arr1

    _, tgt_len = arr.shape  # src_len

    if frac is None:
        frac = 20 / tgt_len
    elif frac > 1:
        frac = 1.
    kwargs.update({"frac": frac})

    # use lowess' original default if set to < 0
    if frac < 0.:
        del kwargs['frac']

    x = np.arange(tgt_len)  # pylint: disable=invalid-name
    yargmax = np.array(arr).argmax(axis=0)
    ymax = np.array(arr).max(axis=0)
    mean_, std_ = ymax.mean(), ymax.std()

    yhat = lowess(yargmax, x, **kwargs)[:, 1]

    # _ = [(int(elm[0]), argmax, elm[1]) for (argmax, elm) in [*zip(yargmax, yhat)]]
    # _ = [(idx, idy, val) for idx, idy, val in idx_idy_val if val / (1 + abs(idy - yhat[idx])**2) > thr]

    # return _

    idx_idy_val = [[idx, idy, val] for idx, (idy, val) in enumerate(zip(yargmax, ymax))]

    if thr is not None and thr < 0:
        thr = mean_ - 2 * std_
    if thr is None:
        thr = mean_ - .68 * std_
    _ = [(idx, idy, val) for idx, idy, val in idx_idy_val if val / (1 + abs(idy - yhat[idx])**2) > thr]
    # _ = [(idx, idy, val) for idx, idy, val in idx_idy_val]

    # return np.array(_)
    return _
