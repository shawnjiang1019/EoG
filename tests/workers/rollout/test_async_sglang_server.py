

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import DictConfig

@patch.dict(
    "sys.modules",
    {
        "verl.workers.rollout.sglang_rollout.sglang_rollout": MagicMock(SGLangRollout=MagicMock()),
    },
)
class TestAsyncSGLangServer:
    @pytest.fixture
    def server_config(self):
        return DictConfig({"rollout": {"tensor_model_parallel_size": 2}})

    @pytest.mark.asyncio
    @patch("verl.workers.rollout.sglang_rollout.async_sglang_server.ray.util.list_named_actors")
    @patch("verl.workers.rollout.async_server.AsyncServerBase._start_fastapi_server", new_callable=AsyncMock)
    @pytest.mark.filterwarnings("ignore:Ray state API is no longer experimental:DeprecationWarning")
    async def test_init_engine(self, mock_start_fastapi_server, mock_list_actors, server_config):
        mock_list_actors.return_value = [
            {"name": "test_prefixWorkerDict_1:0", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:1", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:0", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:1", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:2", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:3", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:2", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:3", "namespace": "test"},
        ]
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSGLangServer

        ActualClassToInstantiate = AsyncSGLangServer
        if hasattr(AsyncSGLangServer, "__ray_metadata__") and hasattr(
            AsyncSGLangServer.__ray_metadata__, "modified_class"
        ):
            ActualClassToInstantiate = AsyncSGLangServer.__ray_metadata__.modified_class

        def mock_get_actor_side_effect(name, namespace=None):

            actor_mock = MagicMock()

            actor_mock.name = name

            def getitem_mock(key):
                if key == "name":
                    return name

                return MagicMock(name=f"mock.__getitem__('{key}')")

            actor_mock.__getitem__.side_effect = getitem_mock

            return actor_mock

        with patch(
            "verl.workers.rollout.sglang_rollout.async_sglang_server.ray.get_actor",
            side_effect=mock_get_actor_side_effect,
        ):

            instance = ActualClassToInstantiate(server_config, 4, 0, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_0:0"
            assert instance.workers[0].name == "test_prefixWorkerDict_0:0"
            assert instance.workers[1].name == "test_prefixWorkerDict_0:1"

            instance = ActualClassToInstantiate(server_config, 4, 1, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_0:2"
            assert instance.workers[0].name == "test_prefixWorkerDict_0:2"
            assert instance.workers[1].name == "test_prefixWorkerDict_0:3"

            instance = ActualClassToInstantiate(server_config, 4, 3, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_1:2"
            assert instance.workers[0].name == "test_prefixWorkerDict_1:2"
            assert instance.workers[1].name == "test_prefixWorkerDict_1:3"
