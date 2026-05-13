# -*- coding: utf-8 -*-
"""Helpers for on-disk per-slice SAM / AI embeddings (slice_<index>.npy)."""

import os
import os.path as osp
import re

_SLICE_NPY_RE = re.compile(r"^slice_(\d+)\.npy$", re.IGNORECASE)


def count_slice_embedding_files(embedding_dir):
    """
    Count ``slice_<i>.npy`` files under ``embedding_dir``.

    Returns 0 if ``embedding_dir`` is falsy or not a directory.
    """
    if not embedding_dir:
        return 0
    path = osp.expanduser(embedding_dir)
    if not osp.isdir(path):
        return 0
    n = 0
    try:
        for name in os.listdir(path):
            if _SLICE_NPY_RE.match(name):
                n += 1
    except OSError:
        return 0
    return n
