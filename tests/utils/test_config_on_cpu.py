

import unittest
from dataclasses import dataclass

from omegaconf import OmegaConf

from verl.utils import omega_conf_to_dataclass

@dataclass
class TestDataclass:
    hidden_size: int
    activation: str

@dataclass
class TestTrainConfig:
    batch_size: int
    model: TestDataclass

_cfg_str = """train_config:
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu"""

class TestConfigOnCPU(unittest.TestCase):

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)

class TestPrintCfgCommand(unittest.TestCase):

    def test_command_with_override(self):
        import subprocess

        result = subprocess.run(
            ["python3", "scripts/print_cfg.py", "critic.profiler.discrete=True", "+critic.profiler.extra.any_key=val"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        self.assertIn("critic", result.stdout)
        self.assertIn("profiler", result.stdout)
        self.assertIn("discrete=True", result.stdout)
        self.assertIn("extra={'any_key': 'val'}", result.stdout)

if __name__ == "__main__":
    unittest.main()
