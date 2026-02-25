

import unittest
from unittest.mock import MagicMock, patch

from verl.utils import omega_conf_to_dataclass
from verl.utils.profiler import ProfilerConfig
from verl.utils.profiler.nvtx_profile import NsightSystemsProfiler

class TestProfilerConfig(unittest.TestCase):
    def test_config_init(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        arr = cfg.actor_rollout_ref
        for config in [
            cfg.critic.profiler,
            arr.profiler,
            cfg.reward_model.profiler,
        ]:
            profiler_config = omega_conf_to_dataclass(config)
            self.assertEqual(profiler_config.discrete, config.discrete)
            self.assertEqual(profiler_config.all_ranks, config.all_ranks)
            self.assertEqual(profiler_config.ranks, config.ranks)
            assert isinstance(profiler_config, ProfilerConfig)
            with self.assertRaises(AttributeError):
                _ = profiler_config.non_existing_key
            assert config.get("non_existing_key") == profiler_config.get("non_existing_key")
            assert config.get("non_existing_key", 1) == profiler_config.get("non_existing_key", 1)
            assert config["discrete"] == profiler_config["discrete"]
            from dataclasses import FrozenInstanceError

            with self.assertRaises(FrozenInstanceError):
                profiler_config.discrete = False

    def test_frozen_config(self):
        from dataclasses import FrozenInstanceError

        from verl.utils.profiler.config import ProfilerConfig

        config = ProfilerConfig(discrete=True, all_ranks=False, ranks=[0])

        with self.assertRaises(FrozenInstanceError):
            config.discrete = False

        with self.assertRaises(FrozenInstanceError):
            config.all_ranks = True

        with self.assertRaises(FrozenInstanceError):
            config.ranks = [1, 2, 3]

        with self.assertRaises(TypeError):
            config["discrete"] = False

        with self.assertRaises(TypeError):
            config["all_ranks"] = True

        with self.assertRaises(TypeError):
            config["ranks"] = [1, 2, 3]

        config["extra"]["key"] = "value"

class TestNsightSystemsProfiler(unittest.TestCase):

    def setUp(self):
        self.config = ProfilerConfig(all_ranks=True)
        self.rank = 0
        self.profiler = NsightSystemsProfiler(self.rank, self.config)

    def test_initialization(self):
        self.assertEqual(self.profiler.this_rank, True)
        self.assertEqual(self.profiler.this_step, False)
        self.assertEqual(self.profiler.discrete, False)

    def test_start_stop_profiling(self):
        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:

            self.profiler.start()
            self.assertTrue(self.profiler.this_step)
            mock_start.assert_called_once()

            self.profiler.stop()
            self.assertFalse(self.profiler.this_step)
            mock_stop.assert_called_once()

    def test_discrete_profiling(self):
        discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
        profiler = NsightSystemsProfiler(self.rank, discrete_config)

        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:
            profiler.start()
            self.assertTrue(profiler.this_step)
            mock_start.assert_not_called()

            profiler.stop()
            self.assertFalse(profiler.this_step)
            mock_stop.assert_not_called()

    def test_annotate_decorator(self):
        mock_self = MagicMock()
        mock_self.profiler = self.profiler
        mock_self.profiler.this_step = True

        @NsightSystemsProfiler.annotate(message="test")
        def test_func(self, *args, **kwargs):
            return "result"

        with (
            patch("torch.cuda.profiler.start") as mock_start,
            patch("torch.cuda.profiler.stop") as mock_stop,
            patch("verl.utils.profiler.nvtx_profile.mark_start_range") as mock_start_range,
            patch("verl.utils.profiler.nvtx_profile.mark_end_range") as mock_end_range,
        ):
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_start.assert_not_called()
            mock_stop.assert_not_called()

    def test_annotate_discrete_mode(self):
        discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
        profiler = NsightSystemsProfiler(self.rank, discrete_config)
        mock_self = MagicMock()
        mock_self.profiler = profiler
        mock_self.profiler.this_step = True

        @NsightSystemsProfiler.annotate(message="test")
        def test_func(self, *args, **kwargs):
            return "result"

        with (
            patch("torch.cuda.profiler.start") as mock_start,
            patch("torch.cuda.profiler.stop") as mock_stop,
            patch("verl.utils.profiler.nvtx_profile.mark_start_range") as mock_start_range,
            patch("verl.utils.profiler.nvtx_profile.mark_end_range") as mock_end_range,
        ):
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_start.assert_called_once()
            mock_stop.assert_called_once()

if __name__ == "__main__":
    unittest.main()
