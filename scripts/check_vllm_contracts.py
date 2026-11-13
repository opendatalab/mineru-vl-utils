"""检查已发布 vLLM 的调用契约，并用官方预处理函数验证同步入口保留视觉特征。"""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from PIL import Image
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient

VERSIONS = (
    "0.19.1",
    "0.20.0",
    "0.20.1",
    "0.20.2",
    "0.21.0",
    "0.22.0",
    "0.22.1",
    "0.23.0",
    "0.24.0",
    "0.25.0",
    "0.25.1",
    "0.26.0",
    "0.27.0",
    "0.27.1",
    "0.28.0",
    "0.29.0",
    "0.30.0",
)


def source(cache: Path, version: str, path: str) -> str:
    """只读取固定官方标签的源码，并缓存以支持离线重复验证。"""
    target = cache / version / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            f"https://raw.githubusercontent.com/vllm-project/vllm/v{version}/{path}", timeout=30
        ) as response:
            target.write_bytes(response.read())
    return target.read_text()


def class_node(text: str, name: str) -> ast.ClassDef:
    """从静态源码选取指定类，不导入 vLLM 或触发 GPU 初始化。"""
    return next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)


def method(node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """定位真实类方法，缺失时使契约检查失败。"""
    return next(item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)


def check_version(cache: Path, version: str) -> dict:
    """校验生成参数与 logits processor，实际执行上游无模型预处理分支。"""
    llm = class_node(source(cache, version, "vllm/entrypoints/llm.py"), "LLM")
    asynchronous = class_node(source(cache, version, "vllm/v1/engine/async_llm.py"), "AsyncLLM")
    renderer_source = source(cache, version, "vllm/renderers/base.py")
    renderer = class_node(renderer_source, "BaseRenderer")
    for cls, name, required in [
        (llm, "generate", {"prompts", "sampling_params", "use_tqdm"}),
        (asynchronous, "generate", {"prompt", "sampling_params", "request_id", "priority"}),
        (renderer, "render_cmpl", {"prompts"}),
        (renderer, "render_cmpl_async", {"prompts"}),
    ]:
        args = method(cls, name).args
        assert required <= {arg.arg for arg in args.args + args.kwonlyargs}, (version, name)
    method(llm, "get_tokenizer")
    method(asynchronous, "from_engine_args")
    method(asynchronous, "tokenizer")
    method(asynchronous, "check_health")
    method(asynchronous, "shutdown")
    errors_source = source(cache, version, "vllm/v1/engine/exceptions.py")
    fatal = class_node(errors_source, "EngineDeadError")
    # 保留支持范围内真实继承变化的证据，不能假设致命错误继承 RuntimeError。
    fatal_bases = [ast.unparse(base) for base in fatal.bases]
    assert fatal_bases in (["Exception"], ["VLLMServerError"]), (version, fatal_bases)
    if fatal_bases == ["VLLMServerError"]:
        base_source = source(cache, version, "vllm/exceptions.py")
        assert [ast.unparse(base) for base in class_node(base_source, "VLLMServerError").bases] == ["VLLMError"]
        assert [ast.unparse(base) for base in class_node(base_source, "VLLMError").bases] == ["Exception"]
    processor = class_node(source(cache, version, "vllm/v1/sample/logits_processor/interface.py"), "LogitsProcessor")
    for name in ("__init__", "apply", "update_state", "is_argmax_invariant"):
        method(processor, name)
    config = class_node(source(cache, version, "vllm/config/compilation.py"), "CompilationConfig")
    fields = {item.target.id for item in config.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    assert "mode" in fields and "level" not in fields
    # 只执行预处理函数本体；所有 GPU、tokenizer 和模型计算均不在执行范围。
    node = method(renderer, "_process_tokens")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    scope = {"tokens_input": lambda ids: {"type": "token", "prompt_token_ids": ids}}
    exec(compile(module, f"vllm-{version}-renderer", "exec"), scope)
    process_tokens = scope["_process_tokens"]

    class Engine:
        """用真实上游预处理替代 GPU 推理，记录实际保留下来的图像像素。"""

        def __init__(self) -> None:
            """保留与同步 vLLM 相同的 renderer 属性，捕获意外的提前渲染。"""
            self.renderer = SimpleNamespace(render_cmpl=self.render)
            self.calls = 0
            self.features = []

        def render(self, prompts: list[dict]) -> list[dict]:
            """只模拟模型特征物化，分支判断直接来自官方 renderer 源码。"""
            self.calls += 1
            instance = SimpleNamespace(
                _process_multimodal=lambda ids, data, **kwargs: {
                    "type": "multimodal",
                    "prompt_token_ids": ids,
                    "mm_kwargs": data["image"],
                }
            )
            return [process_tokens(instance, {"prompt_token_ids": [1, 2], **prompt}) for prompt in prompts]

        def generate(self, prompts: list[dict], **kwargs: object) -> list[object]:
            """同步入口内部渲染一次，再读取视觉特征，检测重入渲染引起的数据丢失。"""
            rendered = self.render(prompts)
            self.features.extend([image.getpixel((0, 0)) for prompt in rendered for image in prompt.get("mm_kwargs", [])])
            return [SimpleNamespace(text="recognized") for _ in prompts]

    client = object.__new__(VllmEngineVlmClient)
    client.use_tqdm = False
    client.vllm_llm = Engine()
    client.get_output_content = lambda output: output.text
    images = [[Image.new("RGB", (2, 2), "red")], [Image.new("RGB", (2, 2), "blue")]]
    assert client._predict_one_batch(images, ["first", "second"], [None, None]) == ["recognized", "recognized"]
    assert client.vllm_llm.calls == 1
    assert client.vllm_llm.features == [(255, 0, 0), (0, 0, 255)]
    return {
        "version": version,
        "success": True,
        "renderer_sha256": hashlib.sha256(renderer_source.encode()).hexdigest(),
        "features": client.vllm_llm.features,
        "engine_dead_bases": fatal_bases,
        "engine_errors_sha256": hashlib.sha256(errors_source.encode()).hexdigest(),
        "public_async_health": True,
    }


def main() -> None:
    """输出逐版本检查证据，不把静态和 CPU 预处理验证描述成 GPU 推理验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [check_version(args.cache, version) for version in VERSIONS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu_inference_tested": False, "results": records}, indent=2) + "\n")
    print(f"{len(records)} vLLM versions passed interface and visual-input contracts")


if __name__ == "__main__":
    main()
