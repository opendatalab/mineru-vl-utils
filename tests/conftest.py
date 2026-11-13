"""共享真实 vLLM 异常契约样例，测试无需安装 GPU 引擎。"""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(params=["0.19.1", "0.30.0"])
def native_errors(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """执行固定官方异常模块，哈希校验防止测试替身悄悄改变真实继承关系。"""
    root = Path(__file__).parent / "fixtures" / "vllm_errors"
    records = json.loads((root / "manifest.json").read_text())
    for record in sorted(records, key=lambda item: len(item["file"])):
        if not record["file"].startswith(request.param):
            continue
        source = (root / record["file"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == record["sha256"]
        name = "vllm.v1.engine.exceptions" if "v1_engine" in record["file"] else "vllm.exceptions"
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, record["url"], "exec"), module.__dict__)
    return sys.modules["vllm.v1.engine.exceptions"]
