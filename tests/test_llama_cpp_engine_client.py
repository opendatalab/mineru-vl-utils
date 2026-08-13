"""Mock 单元测试：LlamaCppEngineVlmClient 的消息构造、参数改名映射、异常转换逻辑。

策略：object.__new__(Engine) 绕过 Engine.__init__ (会真的加载 GGUF 模型)，
用 unittest.mock 替换 generate()/agenerate()，不需要真实模型/GPU。
"""

import asyncio

import pytest
from mineru_llama_cpp import EngineError, GenerateResult, InvalidRequestError
from mineru_llama_cpp import Engine as LlamaCppEngine
from PIL import Image

from mineru_vl_utils.vlm_client.base_client import RequestError, SamplingParams, ServerError
from mineru_vl_utils.vlm_client.llama_cpp_engine_client import LlamaCppEngineVlmClient


def _make_client(mock_engine: LlamaCppEngine) -> LlamaCppEngineVlmClient:
    c = LlamaCppEngineVlmClient(llama_cpp_engine=mock_engine)
    return c


@pytest.fixture
def mock_engine() -> LlamaCppEngine:
    """绕过 Engine.__init__（会真的加载 GGUF 模型），构造一个足以通过 isinstance 检查的空壳。"""
    return object.__new__(LlamaCppEngine)


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="white")


def _stub_generate_result(content: str = "hello", finish_reason: str = "stop") -> GenerateResult:
    return GenerateResult(
        content=content,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        tokens_evaluated=10,
        tokens_predicted=5,
        timings=None,
    )


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_default_image_before_text(mock_engine):
    c = _make_client(mock_engine)
    messages = c.build_messages(["data:image/png;base64,AAAA"], "describe this")
    assert messages[0] == {"role": "system", "content": c.system_prompt}
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert user_content[1] == {"type": "text", "text": "describe this"}


def test_build_messages_text_before_image(mock_engine):
    c = _make_client(mock_engine)
    c.text_before_image = True
    messages = c.build_messages(["data:image/png;base64,AAAA"], "describe this")
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "text", "text": "describe this"}
    assert user_content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_build_messages_placeholder_splits_multi_image(mock_engine):
    c = _make_client(mock_engine)
    image_urls = ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]
    messages = c.build_messages(image_urls, "before<image>middle<image>after")
    user_content = messages[1]["content"]
    # split("<image>", maxsplit=2) on a prompt with exactly 2 occurrences
    # splits at both, yielding 3 parts ("before"/"middle"/"after") -- the
    # trailing "after" has no image to pair with and is kept as trailing
    # text, same behavior as http_client.py's build_request_body.
    assert user_content == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "middle"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        {"type": "text", "text": "after"},
    ]


# ---------------------------------------------------------------------------
# build_llama_cpp_sampling_params
# ---------------------------------------------------------------------------


def test_sampling_params_field_renames(mock_engine):
    c = _make_client(mock_engine)
    sp = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=128,
        no_repeat_ngram_size=3,
    )
    llama_cpp_sp = c.build_llama_cpp_sampling_params(sp)
    assert llama_cpp_sp.temperature == 0.5
    assert llama_cpp_sp.top_p == 0.9
    assert llama_cpp_sp.top_k == 40
    assert llama_cpp_sp.repeat_penalty == 1.1  # renamed from repetition_penalty
    assert llama_cpp_sp.n_predict == 128  # renamed from max_new_tokens
    # no_repeat_ngram_size has no equivalent field and must not raise/appear
    assert not hasattr(llama_cpp_sp, "no_repeat_ngram_size") or llama_cpp_sp.no_repeat_ngram_size is None


def test_sampling_params_unset_fields_stay_unset(mock_engine):
    c = _make_client(mock_engine)
    llama_cpp_sp = c.build_llama_cpp_sampling_params(None)
    assert llama_cpp_sp.temperature is None
    assert llama_cpp_sp.n_predict is None


# ---------------------------------------------------------------------------
# get_output_content / finish_reason handling
# ---------------------------------------------------------------------------


def test_get_output_content_stop(mock_engine):
    c = _make_client(mock_engine)
    result = _stub_generate_result(content="done", finish_reason="stop")
    assert c.get_output_content(result) == "done"


def test_get_output_content_length_raises_by_default(mock_engine):
    c = _make_client(mock_engine)
    result = _stub_generate_result(finish_reason="length")
    with pytest.raises(RequestError):
        c.get_output_content(result)


def test_get_output_content_length_allowed(mock_engine):
    c = _make_client(mock_engine)
    c.allow_truncated_content = True
    result = _stub_generate_result(content="truncated", finish_reason="length")
    assert c.get_output_content(result) == "truncated"


# ---------------------------------------------------------------------------
# predict() / aio_predict() happy path + exception mapping
# ---------------------------------------------------------------------------


def test_predict_happy_path(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)
    monkeypatch.setattr(mock_engine, "generate", lambda messages, sp: _stub_generate_result("hi"))
    assert c.predict(image=image) == "hi"


def test_predict_maps_invalid_request_error(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    def _raise(messages, sp):
        raise InvalidRequestError("bad request")

    monkeypatch.setattr(mock_engine, "generate", _raise)
    with pytest.raises(RequestError):
        c.predict(image=image)


def test_predict_maps_engine_error(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    def _raise(messages, sp):
        raise EngineError("internal failure")

    monkeypatch.setattr(mock_engine, "generate", _raise)
    with pytest.raises(ServerError):
        c.predict(image=image)


def test_aio_predict_happy_path(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    async def _agenerate(messages, sp):
        return _stub_generate_result("hi async")

    monkeypatch.setattr(mock_engine, "agenerate", _agenerate)
    assert asyncio.run(c.aio_predict(image=image)) == "hi async"


def test_aio_predict_maps_context_exceeded_error(mock_engine, image, monkeypatch):
    """ContextExceededError is a subclass of InvalidRequestError -- must map the same way."""
    from mineru_llama_cpp import ContextExceededError

    c = _make_client(mock_engine)

    async def _raise(messages, sp):
        raise ContextExceededError("too long")

    monkeypatch.setattr(mock_engine, "agenerate", _raise)
    with pytest.raises(RequestError):
        asyncio.run(c.aio_predict(image=image))


# ---------------------------------------------------------------------------
# batch_predict() / aio_batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_preserves_order(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)
    call_count = 0

    def _generate(messages, sp):
        nonlocal call_count
        call_count += 1
        # echo back which text prompt was embedded in this call
        text_part = next(p for p in messages[-1]["content"] if p["type"] == "text")
        return _stub_generate_result(content=text_part["text"])

    monkeypatch.setattr(mock_engine, "generate", _generate)
    results = c.batch_predict(images=[image, image, image], prompts=["a", "b", "c"])
    assert results == ["a", "b", "c"]
    assert call_count == 3


def test_aio_batch_predict_preserves_order(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    async def _agenerate(messages, sp):
        text_part = next(p for p in messages[-1]["content"] if p["type"] == "text")
        return _stub_generate_result(content=text_part["text"])

    monkeypatch.setattr(mock_engine, "agenerate", _agenerate)
    results = asyncio.run(c.aio_batch_predict(images=[image, image, image], prompts=["a", "b", "c"]))
    assert results == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_none_engine():
    with pytest.raises(ValueError):
        LlamaCppEngineVlmClient(llama_cpp_engine=None)


def test_constructor_rejects_wrong_type():
    with pytest.raises(ValueError):
        LlamaCppEngineVlmClient(llama_cpp_engine=object())
