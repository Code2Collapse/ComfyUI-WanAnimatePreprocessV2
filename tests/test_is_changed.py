"""IS_CHANGED regression tests for WanAnimatePreprocessV2."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_util():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("wan_is_changed_util", ROOT / "_is_changed_util.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hash_args_and_kwargs_stable_for_scalars():
    util = _load_util()
    a = util.hash_args_and_kwargs(width=832, height=480, label="pose")
    b = util.hash_args_and_kwargs(width=832, height=480, label="pose")
    assert a == b
    assert len(a) == 32


def test_hash_args_and_kwargs_changes_when_scalar_changes():
    util = _load_util()
    a = util.hash_args_and_kwargs(width=832)
    b = util.hash_args_and_kwargs(width=833)
    assert a != b


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed in this interpreter",
)
def test_hash_args_and_kwargs_tensor_fingerprint():
    import torch

    util = _load_util()
    t1 = torch.zeros(1, 8, 8, 3)
    t2 = torch.ones(1, 8, 8, 3)
    assert util.hash_args_and_kwargs(images=t1) != util.hash_args_and_kwargs(images=t2)


def test_is_changed_pattern_never_returns_nan():
    util = _load_util()

    class _StubNode:
        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return util.hash_args_and_kwargs(**kwargs)

    out = _StubNode.IS_CHANGED(pose_data={"n_frames": 3}, threshold=0.3)
    assert isinstance(out, str)
    assert out == out
