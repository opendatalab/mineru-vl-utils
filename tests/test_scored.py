"""Mock 单元测试：测试 predict_scored / score 的 logprobs 提取和指标计算逻辑。

策略：
- 真实 Qwen2-VL-2B-Instruct tokenizer（纯 CPU，无需模型权重）
- mock vllm.LLM.generate() 返回值，构造符合 vLLM RequestOutput 结构的对象
- 通过 object.__new__() 绕过 __init__ 中的 vllm import
"""

import asyncio
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from mineru_vl_utils.vlm_client.base_client import (
    SamplingParams,
    ScoredOutput,
    compute_confidence_metrics,
)
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient
from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient


# ---------------------------------------------------------------------------
# Mock vLLM types
# ---------------------------------------------------------------------------


@dataclass
class MockLogprob:
    logprob: float


@dataclass
class MockCompletionOutput:
    text: str
    token_ids: list[int]
    logprobs: list[dict[int, MockLogprob]]
    finish_reason: str = "stop"


@dataclass
class MockRequestOutput:
    outputs: list[MockCompletionOutput]
    finished: bool = True
    # for score (prompt_logprobs)
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_logprobs: list[dict[int, MockLogprob] | None] = field(default_factory=list)


class MockVllmSamplingParams:
    """可以像 vllm.SamplingParams 一样接受任意 kwargs 并允许后续赋值。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockRenderer:
    """记录 render_cmpl 调用，并返回模拟新版 vLLM EngineInput。"""

    def __init__(self):
        self.calls = []

    def render_cmpl(self, prompts):
        self.calls.append(prompts)
        return [{"type": "tokens", "prompt_token_ids": [idx + 1]} for idx, _ in enumerate(prompts)]


class MockAsyncRenderer(MockRenderer):
    """记录 render_cmpl_async 调用，并返回模拟新版 vLLM EngineInput。"""

    async def render_cmpl_async(self, prompts):
        self.calls.append(prompts)
        return [{"type": "tokens", "prompt_token_ids": [idx + 10]} for idx, _ in enumerate(prompts)]


class MockAsyncVllm:
    """模拟 vLLM AsyncLLM.generate 的异步迭代接口。"""

    def __init__(self, outputs: list[MockRequestOutput], renderer=None):
        self.outputs = outputs
        self.renderer = renderer
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)

        async def _gen():
            for output in self.outputs:
                yield output

        return _gen()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture
def client(tokenizer):
    """构造 VllmEngineVlmClient 实例，绕过 __init__，使用真实 tokenizer。"""
    c = object.__new__(VllmEngineVlmClient)
    c.prompt = "What is the text in the illustrate?"
    c.system_prompt = "You are a helpful assistant."
    c.sampling_params = None
    c.text_before_image = False
    c.allow_truncated_content = False
    c.tokenizer = tokenizer
    c.model_max_length = 4096
    c.VllmSamplingParams = MockVllmSamplingParams
    c.batch_size = 0
    c.use_tqdm = False
    c.debug = False
    c.vllm_llm = MagicMock()
    c.vllm_llm.renderer = None
    return c


@pytest.fixture
def async_client(tokenizer):
    """构造 VllmAsyncEngineVlmClient 实例，绕过 __init__，使用真实 tokenizer。"""
    c = object.__new__(VllmAsyncEngineVlmClient)
    c.prompt = "What is the text in the illustrate?"
    c.system_prompt = "You are a helpful assistant."
    c.sampling_params = None
    c.text_before_image = False
    c.allow_truncated_content = False
    c.tokenizer = tokenizer
    c.model_max_length = 4096
    c.VllmSamplingParams = MockVllmSamplingParams
    c.VllmRequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
    c.max_concurrency = 100
    c.debug = False
    c.vllm_async_llm = MockAsyncVllm([])
    return c


# ---------------------------------------------------------------------------
# compute_confidence_metrics 纯函数测试
# ---------------------------------------------------------------------------


class TestComputeConfidenceMetrics:
    def test_empty(self):
        ppl, min_lp, std = compute_confidence_metrics([])
        assert ppl == float("inf")
        assert min_lp == float("-inf")
        assert std == 0.0

    def test_single_token(self):
        ppl, min_lp, std = compute_confidence_metrics([-1.0])
        assert ppl == pytest.approx(math.exp(1.0))
        assert min_lp == -1.0
        assert std == 0.0  # single token has no spread

    def test_all_low_confidence(self):
        ppl, min_lp, std = compute_confidence_metrics([-3.0, -4.0, -5.0])
        assert min_lp == -5.0
        assert std > 0.0

    def test_mixed(self):
        logprobs = [-0.5, -1.0, -3.0, -0.2]
        ppl, min_lp, std = compute_confidence_metrics(logprobs)
        expected_ppl = math.exp(-sum(logprobs) / len(logprobs))
        mean = sum(logprobs) / len(logprobs)
        expected_std = math.sqrt(sum((lp - mean) ** 2 for lp in logprobs) / len(logprobs))
        assert ppl == pytest.approx(expected_ppl)
        assert min_lp == -3.0
        assert std == pytest.approx(expected_std)


# ---------------------------------------------------------------------------
# predict_scored 测试
# ---------------------------------------------------------------------------


class TestPredictScored:
    def test_renderer_outputs_are_passed_to_generate(self, client):
        """验证新版 vLLM renderer 输出会直接传给 generate。"""
        renderer = MockRenderer()
        client.vllm_llm.renderer = renderer
        mock_output = MockRequestOutput(
            outputs=[MockCompletionOutput(text="ok", token_ids=[], logprobs=[])]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.batch_predict(images=[None], prompts=["Describe this."])

        assert result == ["ok"]
        call_kwargs = client.vllm_llm.generate.call_args.kwargs
        assert call_kwargs["prompts"] == [{"type": "tokens", "prompt_token_ids": [1]}]
        assert renderer.calls[0][0]["prompt"]
        assert "type" not in renderer.calls[0][0]

    def test_raw_prompt_is_used_when_renderer_is_unavailable(self, client):
        """验证旧版 vLLM 没有 renderer 时继续使用 raw prompt。"""
        client.vllm_llm.renderer = None
        mock_output = MockRequestOutput(
            outputs=[MockCompletionOutput(text="ok", token_ids=[], logprobs=[])]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.batch_predict(images=[None], prompts=["Describe this."])

        assert result == ["ok"]
        call_kwargs = client.vllm_llm.generate.call_args.kwargs
        assert "prompt" in call_kwargs["prompts"][0]
        assert "type" not in call_kwargs["prompts"][0]

    def test_basic(self, client):
        """验证 predict_scored 正确提取 logprobs 并计算指标。"""
        # 用 tokenizer 编码一段文本，模拟模型输出
        generated_text = "Hello, world!"
        token_ids = client.tokenizer.encode(generated_text, add_special_tokens=False)
        # 为每个 token 分配 logprob
        fake_logprobs = [-0.1 * (i + 1) for i in range(len(token_ids))]

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text=generated_text,
                    token_ids=token_ids,
                    logprobs=[
                        {tid: MockLogprob(lp)} for tid, lp in zip(token_ids, fake_logprobs)
                    ],
                )
            ]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.predict_scored(image=None, prompt="Describe this.")
        assert isinstance(result, ScoredOutput)
        assert result.text == generated_text
        assert result.token_ids == token_ids
        assert result.logprobs == pytest.approx(fake_logprobs)

        expected_ppl = math.exp(-sum(fake_logprobs) / len(fake_logprobs))
        assert result.perplexity == pytest.approx(expected_ppl)
        assert result.min_logprob == min(fake_logprobs)

    def test_logprobs_enabled_in_sampling_params(self, client):
        """验证 batch_predict_scored 在调 generate 前设置了 logprobs=0。"""
        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="x",
                    token_ids=[100],
                    logprobs=[{100: MockLogprob(-0.5)}],
                )
            ]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.predict_scored(image=None, prompt="test")

        # 检查传给 generate 的 sampling_params 有 logprobs=0
        call_kwargs = client.vllm_llm.generate.call_args
        sp_list = call_kwargs.kwargs.get("sampling_params") or call_kwargs[1].get("sampling_params")
        for sp in sp_list:
            assert hasattr(sp, "logprobs") and sp.logprobs == 0

    def test_predict_scored_uses_renderer_outputs(self, client):
        """验证 predict_scored 也走 renderer 输出而不是 raw prompt。"""
        renderer = MockRenderer()
        client.vllm_llm.renderer = renderer
        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="x",
                    token_ids=[100],
                    logprobs=[{100: MockLogprob(-0.5)}],
                )
            ]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.predict_scored(image=None, prompt="test")

        call_kwargs = client.vllm_llm.generate.call_args.kwargs
        assert call_kwargs["prompts"] == [{"type": "tokens", "prompt_token_ids": [1]}]
        assert "prompt" in renderer.calls[0][0]

    def test_batch(self, client):
        """batch_predict_scored 多个样本。"""
        outputs = []
        for i in range(3):
            text = f"text_{i}"
            tids = client.tokenizer.encode(text, add_special_tokens=False)
            lps = [-0.2 * (j + 1) for j in range(len(tids))]
            outputs.append(
                MockRequestOutput(
                    outputs=[
                        MockCompletionOutput(
                            text=text,
                            token_ids=tids,
                            logprobs=[{t: MockLogprob(lp)} for t, lp in zip(tids, lps)],
                        )
                    ]
                )
            )
        client.vllm_llm.generate.return_value = outputs

        results = client.batch_predict_scored(
            images=[None, None, None],
            prompts=["p1", "p2", "p3"],
        )
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.text == f"text_{i}"
            assert r.perplexity > 0


# ---------------------------------------------------------------------------
# score (teacher forcing) 测试
# ---------------------------------------------------------------------------


class TestScore:
    def test_basic(self, client, tokenizer):
        """验证 score 使用 prompt_logprobs 正确提取 scored_text 部分的 logprobs。"""
        scored_text = "This is a test answer."

        # 用真实 tokenizer 计算 token 数量差
        messages = client.build_messages("Describe this.", 0)
        messages_with_assistant = messages + [{"role": "assistant", "content": scored_text}]

        full_prompt = tokenizer.apply_chat_template(
            messages_with_assistant, tokenize=False, add_generation_prompt=False
        )
        base_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_token_ids = tokenizer.encode(full_prompt)
        base_token_ids = tokenizer.encode(base_prompt)
        scored_token_count = len(full_token_ids) - len(base_token_ids)
        assert scored_token_count > 0, "scored_text should produce at least 1 token"

        # 为整个 prompt 构造 prompt_logprobs
        # prompt_logprobs[0] = None (BOS), 其余为 {token_id: Logprob}
        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]  # BOS
        for i in range(1, len(full_token_ids)):
            tid = full_token_ids[i]
            # scored_text 部分给低一点的 logprob，前面部分给高的
            if i >= len(base_token_ids):
                lp = -0.5  # scored part
            else:
                lp = -0.01  # prompt part (high confidence)
            prompt_logprobs.append({tid: MockLogprob(lp)})

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="",  # score 模式下 generate 的输出文本不重要
                    token_ids=[full_token_ids[-1]],  # max_tokens=1 产生的无关 token
                    logprobs=[{full_token_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_token_ids,
            prompt_logprobs=prompt_logprobs,
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.score(image=None, scored_text=scored_text, prompt="Describe this.")

        assert isinstance(result, ScoredOutput)
        assert result.text == scored_text
        assert len(result.token_ids) == scored_token_count
        assert len(result.logprobs) == scored_token_count
        # 所有 scored part 的 logprob 都是 -0.5
        assert all(lp == pytest.approx(-0.5) for lp in result.logprobs)
        assert result.perplexity == pytest.approx(math.exp(0.5))

    def test_prompt_logprobs_enabled(self, client):
        """验证 score 在调 generate 前设置了 prompt_logprobs=0 和 max_tokens=1。"""
        scored_text = "answer"
        messages = client.build_messages("q", 0)
        messages_with = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt = client.tokenizer.apply_chat_template(
            messages_with, tokenize=False, add_generation_prompt=False
        )
        full_ids = client.tokenizer.encode(full_prompt)

        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
        for i in range(1, len(full_ids)):
            prompt_logprobs.append({full_ids[i]: MockLogprob(-0.3)})

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="", token_ids=[full_ids[-1]],
                    logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_ids,
            prompt_logprobs=prompt_logprobs,
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.score(image=None, scored_text=scored_text, prompt="q")

        call_kwargs = client.vllm_llm.generate.call_args
        sp_list = call_kwargs.kwargs.get("sampling_params") or call_kwargs[1].get("sampling_params")
        for sp in sp_list:
            assert sp.prompt_logprobs == 0
            assert sp.max_tokens == 1

    def test_score_uses_renderer_outputs(self, client):
        """验证 score 的 prompt_logprobs 路径也走 renderer 输出。"""
        renderer = MockRenderer()
        client.vllm_llm.renderer = renderer
        scored_text = "answer"
        messages = client.build_messages("q", 0)
        messages_with = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt = client.tokenizer.apply_chat_template(
            messages_with, tokenize=False, add_generation_prompt=False
        )
        full_ids = client.tokenizer.encode(full_prompt)
        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
        for i in range(1, len(full_ids)):
            prompt_logprobs.append({full_ids[i]: MockLogprob(-0.3)})
        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="",
                    token_ids=[full_ids[-1]],
                    logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_ids,
            prompt_logprobs=prompt_logprobs,
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.score(image=None, scored_text=scored_text, prompt="q")

        call_kwargs = client.vllm_llm.generate.call_args.kwargs
        assert call_kwargs["prompts"] == [{"type": "tokens", "prompt_token_ids": [1]}]
        assert "prompt" in renderer.calls[0][0]

    def test_correct_label_lower_ppl_than_random(self, client, tokenizer):
        """模拟：正确标注的 PPL 应低于随机文本的 PPL。"""
        correct_text = "The quick brown fox"
        random_text = "asdf qwer zxcv bnm"

        def mock_score_one(scored_text, logprob_value):
            messages = client.build_messages("q", 0)
            messages_with = messages + [{"role": "assistant", "content": scored_text}]
            full_prompt = tokenizer.apply_chat_template(
                messages_with, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_prompt)

            prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
            for i in range(1, len(full_ids)):
                prompt_logprobs.append({full_ids[i]: MockLogprob(logprob_value)})

            return MockRequestOutput(
                outputs=[
                    MockCompletionOutput(
                        text="", token_ids=[full_ids[-1]],
                        logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                    )
                ],
                prompt_token_ids=full_ids,
                prompt_logprobs=prompt_logprobs,
            )

        # 正确标注 → 高置信度 (logprob 接近 0)
        client.vllm_llm.generate.return_value = [mock_score_one(correct_text, -0.3)]
        result_correct = client.score(image=None, scored_text=correct_text, prompt="q")

        # 随机文本 → 低置信度 (logprob 很负)
        client.vllm_llm.generate.return_value = [mock_score_one(random_text, -4.0)]
        result_random = client.score(image=None, scored_text=random_text, prompt="q")

        assert result_correct.perplexity < result_random.perplexity
        assert result_correct.logprob_std >= 0
        assert result_random.logprob_std >= 0


class TestAsyncRendererCompatibility:
    def test_aio_predict_uses_async_renderer_output(self, async_client):
        """验证 aio_predict 使用 render_cmpl_async 的 EngineInput。"""
        renderer = MockAsyncRenderer()
        output = MockRequestOutput(
            outputs=[MockCompletionOutput(text="ok", token_ids=[], logprobs=[])]
        )
        async_client.vllm_async_llm = MockAsyncVllm([output], renderer=renderer)

        result = asyncio.run(async_client.aio_predict(image=None, prompt="Describe this."))

        assert result == "ok"
        call_kwargs = async_client.vllm_async_llm.generate_calls[0]
        assert call_kwargs["prompt"] == {"type": "tokens", "prompt_token_ids": [10]}
        assert "prompt" in renderer.calls[0][0]

    def test_aio_predict_scored_uses_async_renderer_output(self, async_client):
        """验证 aio_predict_scored 使用 render_cmpl_async 的 EngineInput。"""
        renderer = MockAsyncRenderer()
        output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="x",
                    token_ids=[100],
                    logprobs=[{100: MockLogprob(-0.5)}],
                )
            ]
        )
        async_client.vllm_async_llm = MockAsyncVllm([output], renderer=renderer)

        result = asyncio.run(async_client.aio_predict_scored(image=None, prompt="test"))

        assert result.text == "x"
        call_kwargs = async_client.vllm_async_llm.generate_calls[0]
        assert call_kwargs["prompt"] == {"type": "tokens", "prompt_token_ids": [10]}

    def test_aio_score_uses_async_renderer_output(self, async_client):
        """验证 aio_score 的 prompt_logprobs 路径使用 render_cmpl_async。"""
        renderer = MockAsyncRenderer()
        scored_text = "answer"
        messages = async_client.build_messages("q", 0)
        messages_with = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt = async_client.tokenizer.apply_chat_template(
            messages_with, tokenize=False, add_generation_prompt=False
        )
        full_ids = async_client.tokenizer.encode(full_prompt)
        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
        for i in range(1, len(full_ids)):
            prompt_logprobs.append({full_ids[i]: MockLogprob(-0.3)})
        output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="",
                    token_ids=[full_ids[-1]],
                    logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_ids,
            prompt_logprobs=prompt_logprobs,
        )
        async_client.vllm_async_llm = MockAsyncVllm([output], renderer=renderer)

        result = asyncio.run(async_client.aio_score(image=None, scored_text=scored_text, prompt="q"))

        assert result.text == scored_text
        call_kwargs = async_client.vllm_async_llm.generate_calls[0]
        assert call_kwargs["prompt"] == {"type": "tokens", "prompt_token_ids": [10]}
