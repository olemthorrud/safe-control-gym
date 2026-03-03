"""Replay buffer class used by static predictor experiments."""

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer class, enabling random batch sampling.

    Experiences are stored as a list of rollouts, where each rollout is
    a list of transitions.
    """

    def __init__(self, max_size, seed):
        self.max_size = max_size
        self.seed = seed
        self.buffer = deque(maxlen=max_size)
        self.size_list = deque(maxlen=max_size)
        self.size = 0
        self.rng1 = np.random.default_rng(seed)  # Not thread safe
        random.seed(int(seed))

    def push(self, rollout):
        """Add one rollout to replay buffer."""
        self.size_list.append(len(rollout))
        self.size = sum(self.size_list)
        self.buffer.append(rollout)

    def update_buffer(self, buffer):
        buffer = (
            buffer if isinstance(buffer, deque) else deque(buffer, maxlen=len(buffer))
        )
        self.buffer = buffer
        self.size_list.append(len(buffer))
        self.size = len(buffer)

    def sample(self, batch_size):
        """Sample random transitions from all available rollouts."""
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        for _ in range(batch_size):
            rollout = random.sample(self.buffer, 1)[0]
            (
                state,
                obs,
                action,
                reward,
                next_state,
                next_obs,
                done,
                info,
            ) = random.sample(rollout, 1)[0]
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)
        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def sample_sequence(self, sample_len):
        """Sample one contiguous sequence from one rollout."""
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        rollout = self.buffer[self.rng1.integers(0, len(self.buffer))]
        if len(rollout) >= sample_len:
            start = self.rng1.integers(0, len(rollout) - sample_len + 1)
            rollout_sample = rollout[start : start + sample_len]
        else:
            rollout_sample = rollout

        for transition in rollout_sample:
            (
                state,
                obs,
                action,
                reward,
                next_state,
                next_obs,
                done,
                info,
            ) = transition
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def all_data_sampler(self, batch_size, drop_last=True):
        (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        ) = self.flatten_buffer()
        sampler = self.indices_sampler(len(state_batch), batch_size, drop_last)
        for indices in sampler:
            states, obss, actions, rewards = [], [], [], []
            next_states, next_obss, dones, infos = [], [], [], []
            for i in indices:
                states.append(state_batch[i])
                obss.append(obs_batch[i])
                actions.append(action_batch[i])
                rewards.append(reward_batch[i])
                next_states.append(next_state_batch[i])
                next_obss.append(next_obs_batch[i])
                dones.append(done_batch[i])
                infos.append(info_batch[i])
            states = torch.FloatTensor(np.array(states))
            obss = torch.FloatTensor(np.array(obss))
            actions = torch.FloatTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            next_obss = torch.FloatTensor(np.array(next_obss))
            dones = torch.FloatTensor(np.array(dones))
            yield (
                states,
                obss,
                actions,
                rewards,
                next_states,
                next_obss,
                dones,
                infos,
            )

    def indices_sampler(self, total_length, batch_size, drop_last=True):
        indices = self.rng1.permutation(np.arange(total_length))
        batches = indices[: len(indices) // batch_size * batch_size].reshape(
            -1, batch_size
        )
        for batch in batches:
            yield batch
        if not drop_last:
            r = len(indices) % batch_size
            if r:
                yield indices[-r:]

    def last_rollout(self):
        """Return the latest rollout in batch form."""
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        rollout_sample = self.buffer[-1]

        for transition in rollout_sample:
            state, obs, action, reward, next_state, next_obs, done, info = transition
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def flatten_buffer(self):
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        for i in range(len(self.buffer)):
            for j in range(len(self.buffer[i])):
                (
                    state,
                    obs,
                    action,
                    reward,
                    next_state,
                    next_obs,
                    done,
                    info,
                ) = self.buffer[i][j]
                state_batch.append(state)
                obs_batch.append(obs)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                next_obs_batch.append(next_obs)
                done_batch.append(done)
                info_batch.append(info)
        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def __len__(self):
        return len(self.buffer)
