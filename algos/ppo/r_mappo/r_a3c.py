# Code adapted from https://github.com/facebookresearch/nocturne/blob/main/algos/ppo/r_mappo/r_mappo.py
import numpy as np
import torch
import torch.nn as nn
from algos.ppo.utils.util import get_gard_norm, huber_loss, mse_loss
from algos.ppo.utils.valuenorm import ValueNorm
from algos.ppo.ppo_utils.util import check


class R_A3C():
    """
    Trainer class for A3C to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_A3C_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.a3c_num_episodes = args.a3c_num_episodes
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously"
        )

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, returns, active_masks):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param returns: (torch.Tensor) reward to go returns.
        :param active_masks: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(returns)
            error_original = self.value_normalizer.normalize(
                returns) - values
        else:
            # advantage function
            error_original = returns - values

        if self._use_huber_loss:
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_original = mse_loss(error_original)

        value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss *
                          active_masks).sum() / active_masks.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
    
    def a3c_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(
            **self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        # value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs_batch, rnn_states_batch,
            rnn_states_critic_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surrogate_loss = imp_weights * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(surrogate_loss, dim=-1, keepdim=True) *
                active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                surrogate_loss, dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights