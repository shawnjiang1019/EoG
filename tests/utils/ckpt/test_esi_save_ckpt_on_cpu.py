

import os
import time
from datetime import datetime, timedelta
from unittest import TestCase

from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi

class TestShouldSaveCkptEsi(TestCase):
    def test_no_expiration_timestamp(self):
        os.environ.pop("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        os.environ.pop("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        self.assertFalse(should_save_ckpt_esi(100))

    def test_mlp_expiration_valid(self):
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 90)
        self.assertTrue(should_save_ckpt_esi(30))

    def test_mlp_expiration_passed(self):
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time - 10)
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_invalid_timestamp(self):
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = "invalid"
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_expiration_not_reached(self):
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 200)
        self.assertFalse(should_save_ckpt_esi(30))

    def test_aws_expiration_not_reached(self):
        now = datetime.now()
        expiration = now + timedelta(minutes=100)
        os.environ["SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(int(expiration.timestamp()))
        self.assertFalse(should_save_ckpt_esi(30 * 60))

    def test_redundant_time(self):
        current_time = time.time()

        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 120)
        self.assertTrue(should_save_ckpt_esi(30, redundant_time=30))

    def test_zero_max_steps_duration(self):
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 60)
        self.assertFalse(should_save_ckpt_esi(0))
