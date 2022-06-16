# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Model for an imitation learning agent."""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

from examples.imitation_learning.filters import MeanStdFilter


class ImitationAgent(nn.Module):
    """Pytorch Module for imitation. Output is a Multivariable Gaussian."""

    def __init__(self, cfg):
        """Initialize."""
        super(ImitationAgent, self).__init__()

        self.n_states = cfg['n_inputs']
        self.hidden_layers = cfg.get('hidden_layers', [256, 256])

        self.discrete = cfg['discrete']

        if self.discrete:
            self.actions_discretizations = cfg['actions_discretizations']
            self.actions_bounds = cfg['actions_bounds']
            self.actions_grids = [
                torch.linspace(a_min, a_max, a_count,
                               requires_grad=False).to(cfg['device'])
                for (a_min, a_max), a_count in zip(
                    self.actions_bounds, self.actions_discretizations)
            ]
        else:
            # neural network outputs between -1 and 1 (tanh filter)
            # then output is sampled from a Gaussian distribution
            # N(nn output * mean_scalings, std_devs)
            self.mean_scalings = torch.tensor(cfg['mean_scalings'])
            self.std_devs = torch.tensor(cfg['std_devs'])
            self.covariance_matrix = torch.diag_embed(self.std_devs)

        self._build_model()

    def _build_model(self):
        """Build agent MLP that outputs an action mean and variance from a state input."""
        if self.hidden_layers is None or len(self.hidden_layers) == 0:
            self.nn = nn.Identity()
            pre_head_size = self.n_states
        else:
            self.nn = nn.Sequential(
                MeanStdFilter(self.n_states),
                nn.Linear(self.n_states, self.hidden_layers[0]),
                nn.Tanh(),
                *[
                    nn.Sequential(
                        nn.Linear(self.hidden_layers[i],
                                  self.hidden_layers[i + 1]),
                        nn.Tanh(),
                    ) for i in range(len(self.hidden_layers) - 1)
                ],
            )
            pre_head_size = self.hidden_layers[-1]

        if self.discrete:
            self.heads = nn.ModuleList([
                nn.Linear(pre_head_size, discretization)
                for discretization in self.actions_discretizations
            ])
        else:
            self.head = nn.Sequential(
                nn.Linear(pre_head_size, len(self.mean_scalings)), nn.Tanh())

    def dist(self, state):
        """Construct a distribution from tensor input."""
        x_out = self.nn(state)
        if self.discrete:
            return [Categorical(logits=head(x_out)) for head in self.heads]
        else:
            return MultivariateNormal(
                self.head(x_out) * self.mean_scalings, self.covariance_matrix)

    def forward(self, state, deterministic=False, return_indexes=False):
        """Generate an output from tensor input."""
        dists = self.dist(state)
        if self.discrete:
            actions_idx = [
                d.logits.argmax(axis=-1) if deterministic else d.sample()
                for d in dists
            ]
            actions = [
                action_grid[action_idx] for action_grid, action_idx in zip(
                    self.actions_grids, actions_idx)
            ]
            return (actions, actions_idx) if return_indexes else actions
        else:
            return [dist.argmax(axis=-1) for dist in dists
                    ] if deterministic else [dist.sample() for dist in dists]

    def log_prob(self, state, ground_truth_action, return_indexes=False):
        """Compute the log prob of the expert action for a given input tensor."""
        dist = self.dist(state)
        if self.discrete:
            # find indexes in actions grids whose values are the closest to the ground truth actions
            actions_idx = self.action_to_grid_idx(ground_truth_action)
            # sum log probs of actions indexes wrt. Categorial variables for each action dimension
            log_prob = sum(
                [d.log_prob(actions_idx[:, i]) for i, d in enumerate(dist)])
            return (log_prob, actions_idx) if return_indexes else log_prob
        else:
            return dist.log_prob(ground_truth_action)

    def action_to_grid_idx(self, action):
        """Convert a batch of actions to a batch of action indexes (for discrete actions only)."""
        # action is of shape (batch_size, n_actions)
        # we want to transform it into an array of same shape, but with indexes instead of actions
        # credits https://stackoverflow.com/a/46184652/16207351
        output = torch.zeros_like(action)
        for i, action_grid in enumerate(self.actions_grids):
            actions = action[:, i]

            # get indexes where actions would be inserted in action_grid to keep it sorted
            idxs = torch.searchsorted(action_grid, actions)

            # if it would be inserted at the end, we're looking at the last action
            idxs[idxs == len(action_grid)] -= 1

            # find indexes where previous index is closer (simple grid has constant sampling intervals)
            idxs[action_grid[idxs] - actions > torch.diff(action_grid).mean() *
                 0.5] -= 1

            # write indexes in output
            output[:, i] = idxs
        return output


if __name__ == '__main__':
    model_cfg = {
        'n_inputs': 100,
        'hidden_layers': [256, 256],
        'discrete': False,
        'mean_scalings': [1, 10, 10000],
        'std_devs': [1.0, 1.0, 1.0],
    }
    if True:
        model_cfg.update({
            'discrete': True,
            'actions_discretizations': [5, 10],
            'actions_bounds': [[-3, 3], [0, 10]],
        })

    model = ImitationAgent(model_cfg)

    sample_states = torch.rand(3, model_cfg['n_inputs'])
    actions = model(sample_states)
    print(actions)
    print(model.log_prob(sample_states, actions))
