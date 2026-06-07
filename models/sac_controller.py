"""Soft Actor-Critic controller that adapts the routing threshold tau and blending weight alpha."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buf: deque[Transition] = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def push(self, transition: Transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        states = torch.from_numpy(np.stack([t.state for t in batch])).float()
        actions = torch.from_numpy(np.stack([t.action for t in batch])).float()
        rewards = torch.from_numpy(np.array([t.reward for t in batch])).float().unsqueeze(-1)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float()
        dones = torch.from_numpy(np.array([t.done for t in batch])).float().unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buf)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy producing actions in [-1, 1] then scaled to action bounds."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mu = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        z = mu + eps * std
        action = torch.tanh(z)
        log_prob = (
            -0.5 * (eps ** 2 + math.log(2 * math.pi)) - log_std - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256, 256)):
        super().__init__()
        self.q1 = self._make_q(state_dim + action_dim, hidden_layers)
        self.q2 = self._make_q(state_dim + action_dim, hidden_layers)

    @staticmethod
    def _make_q(input_dim: int, hidden_layers):
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACController:
    """SAC meta-controller for adapting (tau, alpha) of the routing pipeline.

    Action vector a_t = [delta_tau, delta_alpha] in [-0.1, 0.1]^2. After
    Stage 3 / Stage 4 training, the actor weights are frozen and the controller
    produces state-dependent (tau, alpha) updates at inference without any
    gradient computation or reward evaluation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_layers=(256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau_soft: float = 0.005,
        target_entropy: float | None = None,
        action_bounds: tuple[float, float] = (-0.1, 0.1),
        replay_capacity: int = 100_000,
        device: str = "cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau_soft = tau_soft
        self.action_low, self.action_high = action_bounds
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        self.actor = GaussianActor(state_dim, action_dim, hidden_layers).to(device)
        self.critic = TwinCritic(state_dim, action_dim, hidden_layers).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay = ReplayBuffer(replay_capacity, state_dim, action_dim)

    @property
    def entropy_alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if deterministic:
            mu, _ = self.actor(s)
            a = torch.tanh(mu)
        else:
            a, _ = self.actor.sample(s)
        return self.scale_action(a).cpu().numpy().squeeze(0)

    def update(self, batch_size: int = 256) -> dict:
        if len(self.replay) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.entropy_alpha * next_logp
            target = rewards + self.gamma * (1 - dones) * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, logp = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.entropy_alpha * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1.0 - self.tau_soft).add_(self.tau_soft * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy_alpha": self.entropy_alpha.item(),
        }

    def freeze(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False
        self.actor.eval()
        self.critic.eval()
