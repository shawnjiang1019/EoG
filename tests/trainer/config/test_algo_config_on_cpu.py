

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, KLControlConfig, PFPPOConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.utils.config import omega_conf_to_dataclass

class TestAlgoConfig(unittest.TestCase):

    def setUp(self):

        self.config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "adaptive",
                "kl_coef": 0.002,
                "horizon": 5000,
                "target_kl": 0.05,
            },
            "use_pf_ppo": True,
            "pf_ppo": {"_target_": "verl.trainer.config.PFPPOConfig", "reweight_method": "max_min", "weight_pow": 3.0},
        }
        self.omega_config = OmegaConf.create(self.config_dict)

    def test_dataclass_creation_from_dict(self):
        config = omega_conf_to_dataclass(self.config_dict)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertTrue(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertTrue(config.use_pf_ppo)

    def test_dataclass_creation_from_omega_config(self):
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)

    def test_nested_configs(self):
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config.kl_ctrl, KLControlConfig)
        self.assertEqual(config.kl_ctrl.type, "adaptive")
        self.assertEqual(config.kl_ctrl.kl_coef, 0.002)
        self.assertEqual(config.kl_ctrl.horizon, 5000)
        self.assertEqual(config.kl_ctrl.target_kl, 0.05)

        self.assertIsInstance(config.pf_ppo, PFPPOConfig)
        self.assertEqual(config.pf_ppo.reweight_method, "max_min")
        self.assertEqual(config.pf_ppo.weight_pow, 3.0)

    def test_default_values(self):
        minimal_config = {"gamma": 0.8}
        config = omega_conf_to_dataclass(minimal_config, AlgoConfig)

        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.lam, 1.0)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertFalse(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertFalse(config.use_pf_ppo)

    def test_get_method_backward_compatibility(self):
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertEqual(config.get("gamma"), 0.99)
        self.assertEqual(config.get("gamma", 1.0), 0.99)

        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_post_init_nested_configs(self):

        minimal_config = AlgoConfig(gamma=0.9)

        self.assertIsNotNone(minimal_config.kl_ctrl)
        self.assertIsInstance(minimal_config.kl_ctrl, KLControlConfig)
        self.assertIsNone(minimal_config.pf_ppo)

    def test_config_init_from_yaml(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        algo_config = omega_conf_to_dataclass(cfg.algorithm)
        from verl.trainer.config import AlgoConfig, PFPPOConfig

        assert isinstance(algo_config, AlgoConfig)
        assert isinstance(algo_config.pf_ppo, PFPPOConfig)

class TestAlgoCompute(unittest.TestCase):

    def setUp(self):
        self.algo_config = AlgoConfig(
            gamma=0.99,
            lam=0.95,
            adv_estimator="gae",
            norm_adv_by_std_in_grpo=True,
            use_kl_in_reward=True,
            kl_penalty="kl",
            kl_ctrl=KLControlConfig(type="adaptive", kl_coef=0.002, horizon=5000, target_kl=0.05),
            use_pf_ppo=True,
            pf_ppo=PFPPOConfig(reweight_method="max_min", weight_pow=3.0),
        )

    def test_advantage_estimator_with_cfg(self):
        config = self.algo_config

        adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.assertIsNotNone(adv_fn)

        batch_size, seq_len = 2, 5
        token_level_rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

    def test_grpo_advantage_estimator_with_cfg(self):
        grpo_config = AlgoConfig(adv_estimator="grpo", norm_adv_by_std_in_grpo=True)

        batch_size, seq_len = 4, 3
        token_level_rewards = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.5, 0.2, 0.0], [1.5, 0.8, 0.0]])
        response_mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 1, 1])

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=grpo_config.norm_adv_by_std_in_grpo,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

if __name__ == "__main__":
    unittest.main()
