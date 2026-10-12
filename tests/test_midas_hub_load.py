"""DepthPoseCannyCombinedV2 built_in_midas: the MiDaS torch.hub load must survive comfyui_controlnet_aux (L7.87).

MiDaS's hub repo ships `midas/` without an __init__.py (a namespace package); controlnet_aux puts a REGULAR `midas`
package on sys.path whose modules use `from ..util import`, and a regular package wins over a namespace portion -
"attempted relative import beyond top-level package" on every install with controlnet_aux (measured on the owner's
stack 2026-10-12, docs/evidence/L7.87). `_midas_hub_load` is exercised here against exactly that layout, without
network: torch.hub is faked, the collision is real.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_helper(fake_torch):
    tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_midas_hub_load")
    ns = {"os": os, "torch": fake_torch}
    exec(compile(ast.Module([fn], []), str(ROOT / "nodes.py"), "exec"), ns)
    return ns["_midas_hub_load"]


@pytest.fixture
def layout(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    repo = hub / "intel-isl_MiDaS_master"
    (repo / "midas").mkdir(parents=True)                       # namespace package: no __init__.py, like upstream
    (repo / "midas" / "dpt_depth.py").write_text("class DPTDepthModel:\n    pass\n", encoding="utf-8")
    aux_src = tmp_path / "aux_src" / "custom_controlnet_aux"
    (aux_src / "midas").mkdir(parents=True)
    (aux_src / "midas" / "__init__.py").write_text("from ..util import HWC3\n", encoding="utf-8")   # regular, breaks
    monkeypatch.syspath_prepend(str(aux_src))                  # what controlnet_aux's __init__ does (sys.path.append)
    for k in [k for k in sys.modules if k == "midas" or k.startswith("midas.")]:
        monkeypatch.delitem(sys.modules, k)

    def fake_load(repo_id, entry, trust_repo=True):
        assert repo_id == "intel-isl/MiDaS"
        sys.path.insert(0, str(repo))                         # torch.hub puts the repo first, as the real one does
        try:
            from midas.dpt_depth import DPTDepthModel         # the hubconf import that failed
            return (entry, DPTDepthModel)
        finally:
            sys.path.remove(str(repo))

    torch = types.SimpleNamespace(hub=types.SimpleNamespace(get_dir=lambda: str(hub), load=fake_load))
    return torch, repo


def test_the_collision_is_real_without_the_shim(layout):
    torch, _repo = layout
    with pytest.raises(ImportError, match="relative import beyond top-level package"):
        torch.hub.load("intel-isl/MiDaS", "MiDaS_small")


def test_hub_midas_wins_during_the_load_and_aux_is_put_back(layout):
    torch, _repo = layout
    sentinel = types.ModuleType("midas")                     # something already imported under that name
    sys.modules["midas"] = sentinel
    try:
        entry, cls = _load_helper(torch)("MiDaS_small")
        assert entry == "MiDaS_small" and cls.__name__ == "DPTDepthModel"
        assert sys.modules["midas"] is sentinel              # restored exactly
        assert "midas.dpt_depth" not in sys.modules
    finally:
        sys.modules.pop("midas", None)
