"""验证 HTTP 连接复用、显式关闭及批量取消清理。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.vlm_client.http_client import HttpVlmClient
from mineru_vl_utils.vlm_client.utils import gather_tasks


def test_http_reuses_loop_pool_and_closes_all_owned_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一循环的多批请求共用连接池，显式关闭后拒绝新请求。"""
    created = []

    async def factory(self: HttpVlmClient) -> httpx.AsyncClient:
        """注入真实 httpx 客户端和无网络传输，记录关闭状态。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
                )
            )
        )
        created.append(client)
        return client

    monkeypatch.setattr(HttpVlmClient, "_new_aio_client", factory)
    client = HttpVlmClient(server_url="http://test", model_name="test", skip_model_name_checking=True)

    async def run() -> None:
        """验证两批及并发请求只构造一次池，并在所属循环关闭。"""
        assert await client.aio_batch_predict([None, None]) == ["ok", "ok"]
        assert await client.aio_predict(None) == "ok"
        assert len(created) == 1
        await client.aclose()
        await client.aclose()
        assert all(pool.is_closed for pool in created)
        assert client._client.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await client.aio_predict(None)

    asyncio.run(run())


def test_sync_batch_closes_temporary_loop_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """既有同步批量接口不得在临时事件循环退出后留下连接。"""
    created = []

    async def factory(self: HttpVlmClient) -> httpx.AsyncClient:
        """为每个临时循环建立一个可检查的连接池。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
                )
            )
        )
        created.append(client)
        return client

    monkeypatch.setattr(HttpVlmClient, "_new_aio_client", factory)
    client = HttpVlmClient(server_url="http://test", model_name="test", skip_model_name_checking=True)
    try:
        assert client.batch_predict([None]) == ["ok"]
        assert client.batch_predict([None]) == ["ok"]
        assert len(created) == 2
        assert all(pool.is_closed for pool in created)
        assert not client._aio_client_cache
    finally:
        asyncio.run(client.aclose())


def test_mineru_client_close_does_not_own_injected_engine_or_executor() -> None:
    """高层客户端关闭不能越权关闭宿主持有的引擎或线程池。"""
    client = object.__new__(MinerUClient)
    engine, executor = Mock(), Mock()
    client.backend = "vllm-async-engine"
    client.client = SimpleNamespace(vllm_async_llm=engine, aclose=AsyncMock())
    client.executor = executor
    asyncio.run(client.aclose())
    client.client.aclose.assert_awaited_once()
    engine.shutdown.assert_not_called()
    executor.shutdown.assert_not_called()


def test_failed_batch_waits_for_sibling_cleanup_under_repeated_cancel() -> None:
    """单项失败触发整批清理，外部重复取消不能遗留后台子任务。"""

    async def run() -> None:
        """受控触发失败并等待另一个请求的异步清理。"""
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        stopped = False

        async def slow() -> None:
            """模拟已经提交的请求及异步取消清理。"""
            nonlocal stopped
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                stopped = True

        async def fail() -> None:
            """另一个子请求启动后才报告错误。"""
            await entered.wait()
            raise ValueError("request failed")

        batch = asyncio.create_task(gather_tasks([slow(), fail()]))
        await cleaning.wait()
        batch.cancel()
        await asyncio.sleep(0)
        batch.cancel()
        await asyncio.sleep(0)
        assert not batch.done()
        release.set()
        with pytest.raises(ValueError, match="request failed"):
            await batch
        assert stopped

    asyncio.run(run())
