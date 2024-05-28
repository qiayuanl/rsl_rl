# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage.rollout_storage import RolloutStorage


class PPOAug:
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.permutation_matrix = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
        self.reflection_matrix = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
        self.Q = torch.zeros((len(self.permutation_matrix), len(self.permutation_matrix)))
        self.Q[torch.arange(len(self.permutation_matrix)), self.permutation_matrix] = self.reflection_matrix
        self.Rd = torch.eye(3)
        self.Rd[1, 1] = -1
        self.Rd_pseudo = torch.eye(3)
        self.Rd_pseudo[[0, 2], [0, 2]] = -1

        self.reflection_matrix_2d = torch.tensor([[0, 1], [1, 0]])
        # priv_obs_reflect_reps = [self.reflection_matrix_2d, self.Rd_pseudo, self.Q, self.Q, self.Q,
        #                          self.Q, self.Rd, self.Rd_pseudo, self.Rd_pseudo,
        #                          self.Rd, self.Rd_pseudo, torch.eye(1), torch.eye(1),
        #                          self.reflection_matrix_2d, self.reflection_matrix_2d]
        # self.priv_obs_reflect_op = self.get_reflect_op(priv_obs_reflect_reps)

        # obs_reflect_reps = [self.reflection_matrix_2d, self.Rd_pseudo, self.Q, self.Q, self.Q, self.Rd_pseudo,
        #                     self.Rd_pseudo]

        obs_reflect_reps = [self.Rd_pseudo, self.Rd_pseudo, self.Rd_pseudo, self.Rd_pseudo, self.Q, self.Q, self.Q]
        self.obs_reflect_op = self.get_reflect_op(obs_reflect_reps)
        self.priv_obs_reflect_op = self.obs_reflect_op

        self.actions_reflect_op = self.Q.clone().to(self.device)

    def get_reflect_op(self, reps):
        reps_shape = []
        for i in range(len(reps)):
            assert reps[i].shape[1] == reps[i].shape[0]
            reps_shape.append(reps[i].shape[0])

        reflect_op = torch.zeros((sum(reps_shape), sum(reps_shape))).to(self.device)

        for i in range(len(reps)):
            idx0 = sum(reps_shape[:i])
            idx1 = sum(reps_shape[:i + 1])
            reflect_op[idx0:idx1, idx0:idx1] = reps[i]

        return reflect_op.to(self.device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs * 2, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.reflect_actions_log_prob = self.actor_critic.get_actions_log_prob(torch.matmul(self.transition.actions, self.actions_reflect_op)).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.augment_transitions()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def augment_transitions(self):
        t = self.transition

        t.actions = torch.cat([t.actions] + [torch.matmul(t.actions, self.actions_reflect_op)], dim=0)
        t.actions_log_prob = torch.cat([t.actions_log_prob] * 2, dim=0)
        t.action_mean = torch.cat([t.action_mean] + [torch.matmul(t.action_mean, self.actions_reflect_op)], dim=0)
        t.action_sigma = torch.abs(
            torch.cat([t.action_sigma] + [torch.abs(torch.matmul(t.action_sigma, self.actions_reflect_op))], dim=0))
        t.values = torch.cat([t.values] * 2, dim=0)
        t.rewards = torch.cat([t.rewards] * 2, dim=0)
        t.dones = torch.cat([t.dones] * 2, dim=0)
        t.observations = torch.cat([t.observations] + [
            torch.matmul(t.observations.view(t.observations.shape[0], -1, self.obs_reflect_op.shape[0]),
                         self.obs_reflect_op).view(t.observations.shape[0], -1)], dim=0)
        t.critic_observations = torch.cat([t.critic_observations] + [torch.matmul(
            t.critic_observations.view(t.critic_observations.shape[0], -1, self.priv_obs_reflect_op.shape[0])
            , self.priv_obs_reflect_op).view(t.critic_observations.shape[0], -1)], dim=0)

    def augment_values(self, values):
        values = torch.cat([values] * 2, dim=0)
        return values

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        last_values = self.augment_values(last_values)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            # B = obs_batch.shape[0]
            # obs_batch = torch.cat([obs_batch,
            #                        torch.matmul(obs_batch.view(B, -1, self.obs_reflect_op.shape[0]), self.obs_reflect_op).view(B,-1)
            #                        ], dim=0)
            # critic_obs_batch = torch.cat([critic_obs_batch,
            #                         torch.matmul(critic_obs_batch.view(B, -1, self.priv_obs_reflect_op.shape[0]), self.priv_obs_reflect_op).view(B,-1)
            #                         ], dim=0)
            # actions_batch = torch.cat([actions_batch, torch.matmul(actions_batch, self.actions_reflect_op)], dim=0)
            # target_values_batch = torch.cat([target_values_batch, target_values_batch], dim=0)
            # advantages_batch = torch.cat([advantages_batch, advantages_batch], dim=0)
            # returns_batch = torch.cat([returns_batch, returns_batch], dim=0)
            # old_actions_log_prob_batch = torch.cat(old_actions_log_prob_batch, dim=0)
            # old_mu_batch = torch.cat([old_mu_batch, torch.matmul(old_mu_batch, self.actions_reflect_op)], dim=0)
            # old_sigma_batch = torch.cat([old_sigma_batch, torch.abs(torch.matmul(old_sigma_batch, self.actions_reflect_op))], dim=0)

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
