

from unittest.mock import patch

import pytest

from verl.interactions.gsm8k_interaction import Gsm8kInteraction

class TestGsm8kInteraction:

    def setup_method(self):
        self.config = {"name": "gsm8k"}
        self.interaction = Gsm8kInteraction(self.config)

    def test_init(self):
        assert self.interaction._instance_dict == {}
        assert self.interaction.config == self.config
        assert self.interaction.name == "gsm8k"

    @pytest.mark.asyncio
    async def test_start_interaction_with_instance_id(self):
        instance_id = "test_instance"
        ground_truth = "42"

        result_id = await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert result_id == instance_id
        assert instance_id in self.interaction._instance_dict
        assert self.interaction._instance_dict[instance_id]["response"] == ""
        assert self.interaction._instance_dict[instance_id]["ground_truth"] == ground_truth
        assert self.interaction._instance_dict[instance_id]["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_start_interaction_without_instance_id(self):
        ground_truth = "42"

        result_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        assert result_id is not None
        assert len(result_id) == 36
        assert result_id in self.interaction._instance_dict
        assert self.interaction._instance_dict[result_id]["ground_truth"] == ground_truth

    @pytest.mark.asyncio
    async def test_start_interaction_without_ground_truth(self):
        instance_id = "test_instance"

        result_id = await self.interaction.start_interaction(instance_id=instance_id)

        assert result_id == instance_id
        assert self.interaction._instance_dict[instance_id]["ground_truth"] is None

    @pytest.mark.asyncio
    async def test_generate_response_correct_answer_with_prefix(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "user", "content": "#### 42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert reward == 1.0
        assert metadata == {}
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 42"

    @pytest.mark.asyncio
    async def test_generate_response_correct_answer_without_prefix(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "user", "content": "42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert reward == 1.0
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 42"

    @pytest.mark.asyncio
    async def test_generate_response_incorrect_answer(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "user", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert response == "Your response is incorrect! You need to reflect on your answer and try again."
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 24"

    @pytest.mark.asyncio
    async def test_generate_response_multiple_messages(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me think about this..."},
            {"role": "user", "content": "#### 42"},
        ]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 42"

    @pytest.mark.asyncio
    async def test_generate_response_no_user_message(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "assistant", "content": "Hello!"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert self.interaction._instance_dict[instance_id]["response"] == "#### "

    @pytest.mark.asyncio
    async def test_calculate_score_direct_call(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        self.interaction._instance_dict[instance_id]["response"] = "#### 42"

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0) as mock_compute:
            score = await self.interaction.calculate_score(instance_id)

            assert score == 1.0
            mock_compute.assert_called_once_with("#### 42", "42", method="flexible", format_score=0.0, score=1.0)

    @pytest.mark.asyncio
    async def test_calculate_score_with_kwargs(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        self.interaction._instance_dict[instance_id]["response"] = "#### 24"

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0) as mock_compute:
            score = await self.interaction.calculate_score(instance_id, extra_param="test")

            assert score == 0.0
            mock_compute.assert_called_once_with("#### 24", "42", method="flexible", format_score=0.0, score=1.0)

    @pytest.mark.asyncio
    async def test_finalize_interaction(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id)

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_interaction_with_kwargs(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id, extra_param="test")

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_interaction(self):
        instance_id = "nonexistent_instance"

        with pytest.raises(KeyError):
            await self.interaction.finalize_interaction(instance_id)

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_correct(self):
        ground_truth = "42"

        instance_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        messages = [{"role": "user", "content": "42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert reward == 1.0

        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_incorrect(self):
        ground_truth = "42"

        instance_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        messages = [{"role": "user", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "42"})

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert reward == 1.0

        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interactions(self):
        ground_truth_1 = "42"
        ground_truth_2 = "24"

        instance_id_1 = await self.interaction.start_interaction(ground_truth=ground_truth_1)
        instance_id_2 = await self.interaction.start_interaction(ground_truth=ground_truth_2)

        assert len(self.interaction._instance_dict) == 2
        assert instance_id_1 in self.interaction._instance_dict
        assert instance_id_2 in self.interaction._instance_dict

        messages_1 = [{"role": "user", "content": "42"}]
        messages_2 = [{"role": "user", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", side_effect=[1.0, 1.0]):
            should_terminate_1, _, reward_1, _ = await self.interaction.generate_response(instance_id_1, messages_1)
            should_terminate_2, _, reward_2, _ = await self.interaction.generate_response(instance_id_2, messages_2)

        assert should_terminate_1 is True
        assert should_terminate_2 is True
        assert reward_1 == 1.0
        assert reward_2 == 1.0

        await self.interaction.finalize_interaction(instance_id_1)
        await self.interaction.finalize_interaction(instance_id_2)

        assert len(self.interaction._instance_dict) == 0

    @pytest.mark.asyncio
    async def test_edge_case_empty_messages(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = []

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == "#### "

    @pytest.mark.asyncio
    async def test_edge_case_message_without_content(self):
        instance_id = "test_instance"
        ground_truth = "42"

        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [
            {"role": "user"}
        ]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == "#### None"

    def test_inheritance_from_base_interaction(self):
        from verl.interactions.base import BaseInteraction

        assert isinstance(self.interaction, BaseInteraction)

        assert hasattr(self.interaction, "start_interaction")
        assert hasattr(self.interaction, "generate_response")
        assert hasattr(self.interaction, "calculate_score")
        assert hasattr(self.interaction, "finalize_interaction")

        assert callable(self.interaction.start_interaction)
        assert callable(self.interaction.generate_response)
        assert callable(self.interaction.calculate_score)
        assert callable(self.interaction.finalize_interaction)

    def test_name_attribute_initialization(self):

        config_with_name = {"name": "custom_gsm8k"}
        interaction_with_name = Gsm8kInteraction(config_with_name)
        assert interaction_with_name.name == "custom_gsm8k"

        config_without_name = {}
        interaction_without_name = Gsm8kInteraction(config_without_name)
        assert interaction_without_name.name == "interaction_agent"

        assert hasattr(self.interaction, "name")
        assert self.interaction.name == "gsm8k"
