"""Module containing regularized PPO algorithm."""
import logging
from typing import Generator, Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

logging.getLogger(__name__)


class MaskedRolloutBuffer(RolloutBuffer):
    """Custom SB3 RolloutBuffer class that filters out invalid samples."""

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """GAE (General Advantage Estimation) to compute advantages and returns."""
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # EDIT_1: Map NaNs to 1
                dones = np.nan_to_num(dones, nan=1.0)

                next_non_terminal = 1.0 - dones
                next_values = last_values

            else:
                # EDIT_1: Map NaNs to 1
                episode_starts = np.nan_to_num(self.episode_starts[step + 1], nan=1.0)

                next_non_terminal = 1.0 - episode_starts
                next_values = self.values[step + 1]

            delta = (
                np.nan_to_num(self.rewards[step], nan=0)  # EDIT_2: Set invalid rewards to zero
                + np.nan_to_num(
                    self.gamma * next_values * next_non_terminal, nan=0
                )  # EDIT_3: Set invalid rewards to zero
                - np.nan_to_num(self.values[step], nan=0)  # EDIT_4: Set invalid values to zero
            )

            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        assert not np.isnan(self.advantages).any(), "Advantages arr contains NaN values; check GAE computation"

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            # Create mask
            self.valid_samples_mask = ~np.isnan(self.swap_and_flatten(self.__dict__["rewards"]))

            # Flatten data
            # EDIT_5: And mask out invalid samples
            for tensor in _tensor_names:
                if tensor == "observations":
                    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])[
                        self.valid_samples_mask.flatten(), :
                    ]
                else:
                    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])[self.valid_samples_mask]

                assert not np.isnan(
                    self.__dict__[tensor]
                ).any(), f"{tensor} tensor contains NaN values; something went wrong"

            self.generator_ready = True

        # EDIT_6: Compute total number of samples and create indices
        total_num_samples = self.valid_samples_mask.sum()
        indices = np.random.permutation(total_num_samples)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = total_num_samples

        start_idx = 0
        while start_idx < total_num_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
