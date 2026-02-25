

import os
import tempfile

import pytest
from omegaconf import OmegaConf

from verl.interactions.base import BaseInteraction
from verl.interactions.gsm8k_interaction import Gsm8kInteraction
from verl.interactions.utils.interaction_registry import (
    get_interaction_class,
    initialize_interactions_from_config,
)

class TestInteractionRegistry:
    def test_get_interaction_class(self):

        base_cls = get_interaction_class("verl.interactions.base.BaseInteraction")
        assert base_cls == BaseInteraction

        gsm8k_cls = get_interaction_class("verl.interactions.gsm8k_interaction.Gsm8kInteraction")
        assert gsm8k_cls == Gsm8kInteraction

    def test_initialize_single_interaction_from_config(self):

        config_content = {
            "interaction": [
                {
                    "name": "test_gsm8k",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            assert len(interaction_map) == 1
            assert "test_gsm8k" in interaction_map
            assert isinstance(interaction_map["test_gsm8k"], Gsm8kInteraction)
            assert interaction_map["test_gsm8k"].name == "test_gsm8k"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_multiple_interactions_from_config(self):
        config_content = {
            "interaction": [
                {
                    "name": "gsm8k_solver",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                },
                {
                    "name": "base_agent",
                    "class_name": "verl.interactions.base.BaseInteraction",
                    "config": {"custom_param": "test_value"},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            assert len(interaction_map) == 2
            assert "gsm8k_solver" in interaction_map
            assert "base_agent" in interaction_map

            assert isinstance(interaction_map["gsm8k_solver"], Gsm8kInteraction)
            assert isinstance(interaction_map["base_agent"], BaseInteraction)

            assert interaction_map["gsm8k_solver"].name == "gsm8k_solver"
            assert interaction_map["base_agent"].name == "base_agent"

            assert interaction_map["base_agent"].config.get("custom_param") == "test_value"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_interaction_without_explicit_name(self):
        config_content = {
            "interaction": [{"class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction", "config": {}}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            assert len(interaction_map) == 1
            assert "gsm8k" in interaction_map
            assert isinstance(interaction_map["gsm8k"], Gsm8kInteraction)
            assert interaction_map["gsm8k"].name == "gsm8k"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_empty_config(self):
        config_content = {"interaction": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)
            assert len(interaction_map) == 0
        finally:
            os.unlink(temp_config_path)

    def test_invalid_class_name(self):
        config_content = {
            "interaction": [{"name": "invalid", "class_name": "invalid.module.InvalidClass", "config": {}}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            with pytest.raises(ModuleNotFoundError):
                initialize_interactions_from_config(temp_config_path)
        finally:
            os.unlink(temp_config_path)

    def test_duplicate_interaction_names(self):
        config_content = {
            "interaction": [
                {"name": "duplicate", "class_name": "verl.interactions.base.BaseInteraction", "config": {}},
                {
                    "name": "duplicate",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            with pytest.raises(ValueError, match="Duplicate interaction name 'duplicate' found"):
                initialize_interactions_from_config(temp_config_path)
        finally:
            os.unlink(temp_config_path)

    def test_auto_name_generation_edge_cases(self):
        config_content = {
            "interaction": [
                {"class_name": "verl.interactions.base.BaseInteraction", "config": {}},
                {"class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction", "config": {}},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            assert len(interaction_map) == 2
            assert "base" in interaction_map
            assert "gsm8k" in interaction_map
        finally:
            os.unlink(temp_config_path)
