"""Fetch doc/para/send corr matrix from fastai server."""
from typing import List, Optional, Union

import httpx
import logzero
from halo import Halo
from logzero import logger

logzero.loglevel(10)

URL = "http://127.0.0.1:8000/text/"
URL = "http://216.24.255.63:8000/text/"


# fmt: off
def fetch_sent_corr(
        sents1: Union[str, List[str]],
        sents2: Optional[Union[str, List[str]]] = None,
        url: str = URL,
) -> Optional[List[List[str]]]:
    # fmt: on
    """Fetch doc/para/send corr matrix from fastai server."""
    if isinstance(sents1, str):
        sents1 = [sents1]
    if isinstance(sents2, str):
        sents2 = [sents2]


    len1 = len(sents1)
    len2 = 0
    if sents2 is not None:
        len2 = len(sents2)

    quo, rem = divmod(len1 + len2, 32)
    time_ = round((quo + 1 if rem else 0) * 8.5 / 60., 1)  # m
    logger.info("\n\t est toa: %s m", time_)

    with Halo():
        data = {
            "text1": sents1,
            "text2": sents2,
        }
        try:
            resp = httpx.post(
                url,
                json=data,
                timeout=None,
            )
            resp.raise_for_status()
        except Exception as exc:
            resp = None
            logger.error("exc: %s", exc)
            raise

        jdata = {}
        try:
            jdata = resp.json()  # type: ignore
        except Exception as exc:
            logger.error("exc: %s", exc)
            jdata.update({"corr": [[0]]})
            raise

    return jdata.get("corr")
