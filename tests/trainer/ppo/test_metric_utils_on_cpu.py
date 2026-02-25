

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from verl.trainer.ppo.metric_utils import (
    bootstrap_metric,
    calc_maj_val,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.metric import (
    reduce_metrics,
)

class TestReduceMetrics(unittest.TestCase):

    def test_reduce_metrics_basic(self):
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "accuracy": [0.0, 0.5, 1.0],
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["loss"], 2.0)
        self.assertEqual(result["accuracy"], 0.5)

    def test_reduce_metrics_empty(self):
        metrics = {
            "empty": [],
        }
        result = reduce_metrics(metrics)

        self.assertTrue(np.isnan(result["empty"]))

    def test_reduce_metrics_single_value(self):
        metrics = {
            "single": [5.0],
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["single"], 5.0)

class TestComputeDataMetrics(unittest.TestCase):

    def setUp(self):

        self.batch = MagicMock()
        self.batch.batch = {
            "token_level_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "token_level_rewards": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "returns": torch.tensor([[1.1, 1.2], [1.3, 1.4]]),
            "responses": torch.zeros((2, 2)),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
            "values": torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
        }

    def test_compute_data_metrics_with_critic(self):
        metrics = compute_data_metrics(self.batch, use_critic=True)

        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("critic/advantages/mean", metrics)
        self.assertIn("critic/returns/mean", metrics)
        self.assertIn("critic/values/mean", metrics)
        self.assertIn("critic/vf_explained_var", metrics)
        self.assertIn("response_length/mean", metrics)
        self.assertIn("prompt_length/mean", metrics)

        self.assertAlmostEqual(metrics["critic/score/mean"], 5.0)
        self.assertAlmostEqual(metrics["critic/rewards/mean"], 2.5)

    def test_compute_data_metrics_without_critic(self):
        metrics = compute_data_metrics(self.batch, use_critic=False)

        self.assertNotIn("critic/values/mean", metrics)
        self.assertNotIn("critic/vf_explained_var", metrics)

        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("response_length/mean", metrics)

class TestComputeTimingMetrics(unittest.TestCase):

    def setUp(self):

        self.batch = MagicMock()
        self.batch.batch = {
            "responses": torch.zeros((2, 3)),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
        }

        self.response_info = {
            "prompt_length": torch.tensor([3.0, 3.0]),
            "response_length": torch.tensor([3.0, 3.0]),
            "response_mask": torch.ones((2, 3)),
        }

    @patch("verl.trainer.ppo.metric_utils._compute_response_info")
    def test_compute_timing_metrics(self, mock_compute_response_info):
        mock_compute_response_info.return_value = self.response_info

        timing_raw = {
            "gen": 0.5,
            "ref": 0.3,
            "values": 0.2,
        }

        metrics = compute_timing_metrics(self.batch, timing_raw)

        self.assertEqual(metrics["timing_s/gen"], 0.5)
        self.assertEqual(metrics["timing_s/ref"], 0.3)
        self.assertEqual(metrics["timing_s/values"], 0.2)

        self.assertAlmostEqual(metrics["timing_per_token_ms/gen"], 0.5 * 1000 / 6, places=5)

        self.assertAlmostEqual(metrics["timing_per_token_ms/ref"], 0.3 * 1000 / 12, places=5)
        self.assertAlmostEqual(metrics["timing_per_token_ms/values"], 0.2 * 1000 / 12, places=5)

class TestComputeThroughputMetrics(unittest.TestCase):

    def setUp(self):

        self.batch = MagicMock()
        self.batch.meta_info = {
            "global_token_num": [100, 200, 300],
        }

    def test_compute_throughout_metrics(self):
        timing_raw = {
            "step": 2.0,
        }

        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=1)

        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / 2.0)

        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=2)

        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / (2.0 * 2))

class TestBootstrapMetric(unittest.TestCase):

    def test_bootstrap_metric_basic(self):
        data = [1, 2, 3, 4, 5]
        reduce_fns = [np.mean, np.max]

        result = bootstrap_metric(data, subset_size=3, reduce_fns=reduce_fns, n_bootstrap=100, seed=42)

        self.assertEqual(len(result), 2)

        mean_result, max_result = result
        self.assertEqual(len(mean_result), 2)
        self.assertEqual(len(max_result), 2)

        self.assertAlmostEqual(mean_result[0], 3.0, delta=0.3)

        self.assertGreater(max_result[0], 3.5)
        self.assertLess(max_result[0], 5.0)

    def test_bootstrap_metric_empty(self):
        with self.assertRaises(ValueError):
            bootstrap_metric([], subset_size=1, reduce_fns=[np.mean])

class TestCalcMajVal(unittest.TestCase):

    def test_calc_maj_val_basic(self):
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "A", "val": 0.7},
        ]

        result = calc_maj_val(data, vote_key="pred", val_key="val")

        self.assertEqual(result, 0.9)

    def test_calc_maj_val_tie(self):
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "B", "val": 0.7},
            {"pred": "A", "val": 0.6},
        ]

        result = calc_maj_val(data, vote_key="pred", val_key="val")

        self.assertTrue(result in [0.9, 0.8])

class TestProcessValidationMetrics(unittest.TestCase):

    def test_process_validation_metrics_basic(self):
        data_sources = ["source1", "source1", "source2"]
        sample_inputs = ["prompt1", "prompt1", "prompt2"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
        }

        result = process_validation_metrics(data_sources, sample_inputs, infos_dict, seed=42)

        self.assertIn("source1", result)
        self.assertIn("source2", result)

        self.assertIn("score", result["source1"])

        self.assertIn("mean@2", result["source1"]["score"])

        self.assertAlmostEqual(result["source1"]["score"]["mean@2"], 0.85)

    def test_process_validation_metrics_with_pred(self):
        data_sources = ["source1", "source1", "source1"]
        sample_inputs = ["prompt1", "prompt1", "prompt1"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
            "pred": ["A", "B", "A"],
        }

        result = process_validation_metrics(data_sources, sample_inputs, infos_dict, seed=42)

        self.assertIn("maj@2/mean", result["source1"]["score"])

if __name__ == "__main__":
    unittest.main()
