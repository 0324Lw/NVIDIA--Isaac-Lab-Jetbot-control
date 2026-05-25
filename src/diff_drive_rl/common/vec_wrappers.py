from __future__ import annotations

import torch


def flatten_agent_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() < 3:
        return obs
    return obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])


def unflatten_agent_actions(actions: torch.Tensor, num_envs: int, num_agents: int) -> torch.Tensor:
    return actions.reshape(num_envs, num_agents, -1)
