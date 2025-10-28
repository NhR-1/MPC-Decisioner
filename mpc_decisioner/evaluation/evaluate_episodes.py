# Portions of this file are adapted from Decision Transformer:
#   https://github.com/kzl/decision-transformer
# Original work © 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors
#   https://arxiv.org/abs/2106.01345
# Licensed under the MIT License (see third_party/decision-transformer/LICENSE).

import numpy as np
import torch
import cvxpy as cp
from scipy import sparse
import collections
import time
from mpc.mpc import run_nmpc_soft

def evaluate_nmpc(
    env,
    create_dynamics,
    terminal_param_scale,
    param_scale,
    state_dim,
    param_dim,
    model,
    max_ep_len=1000,
    K=100,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    eval_mode=1,
    custom_init=None,
    eval_type="mpc_decisioner",
):
    """
    Evaluate MPC using the given model.

    Args:
        env: Environment instance providing `reset` and `step` functions.
        create_dynamics: Function to generate the system dynamics.
        terminal_param_scale: Scaling factor for terminal cost parameters.
        param_scale: Scaling factor for stage cost parameters.
        state_dim: Dimension of the system state.
        param_dim: Dimension of the model's output parameters.
        model: Decision Transformer or similar policy model.
        max_ep_len: Maximum number of environment steps per episode.
        K: Context length used in the model (default: 100).
        scale: Return normalization factor.
        state_mean, state_std: Normalization statistics for input states.
        device: Torch device ("cuda" or "cpu").
        target_return: Desired return for conditioning.
        mode: Evaluation mode ("normal" or "noise").
        eval_mode: Integer flag controlling environment initialization.
        custom_init: Optional custom initial condition for the environment.
        eval_type: Determines whether to record attention maps ("mpc_decisioner" adds them).

    Returns:
        Tuple containing (episode_return, episode_length, episode_data)
    """

    # --------------------------
    # Setup and normalization
    # --------------------------
    model.eval()
    model.to(device=device)

    state_mean = torch.as_tensor(state_mean, device=device)
    state_std = torch.as_tensor(state_std, device=device)

    state = env.reset(eval_mode, custom_init)
    if mode == "noise":
        state += np.random.normal(0, 0.1, size=state.shape)

    # Initialize history tensors
    states = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, state_dim)
    actions = torch.zeros((0, param_dim), dtype=torch.float32, device=device)
    rewards = torch.zeros(0, dtype=torch.float32, device=device)

    ep_return = target_return
    target_return = torch.tensor(ep_return, dtype=torch.float32, device=device).reshape(1, 1)
    timesteps = torch.zeros((1, 1), dtype=torch.long, device=device)

    episode_return, episode_length = 0, 0

    # Environment parameters
    if eval_type != 'optimal':
        N = env.N
        R = np.diag(env.r_diag[:])
    else:
        N = 200
        Q_diag_base = env.q_diag
        Q_diag = np.tile(Q_diag_base, N+1)
        R = np.diag(env.r_diag[:])
    xr = env.target_state
    nx = env.nx
    dt = env.dt
    u_bounds = env.u_bounds
    x_bounds = env.x_bounds

    
    episode_data = collections.defaultdict(list)

    # --------------------------
    # Main evaluation loop
    # --------------------------
    for t in range(max_ep_len):
        # Pad tensors
        actions = torch.cat([actions, torch.zeros((1, param_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        # Compute next action
        if t == 1000:
            print("Reached iteration 1000, skipping.")
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

        # Update last action entry
        actions[-1] = action
        action = action.detach().cpu().numpy()
        if eval_type == 'optimal':
            action = np.concatenate(((1 / param_scale)*Q_diag[:nx], (1 / terminal_param_scale)*Q_diag[-nx:]), 0*Q_diag[-nx:], axis=0)


        # Extract attention if enabled
        if eval_type == "mpc_decisioner":
            attn_list = model.attn_weights  # List of [batch, heads, T, T]
            T = attn_list[0].shape[-1]
            A_rollout = torch.eye(T, device=attn_list[0].device)

            for A in attn_list:
                A_mean = A.mean(dim=1)[0]  # Average over heads
                A_resid = A_mean + torch.eye(T, device=A_mean.device)
                A_resid /= A_resid.sum(dim=-1, keepdim=True)
                A_rollout = torch.matmul(A_resid, A_rollout)

            current_attention = A_rollout[-1].detach().cpu().numpy()

        # MPC rollout
        stage = param_scale * action[:nx]
        terminal = terminal_param_scale * action[nx:2 * nx]
        t_x = action[-nx:]
        Q_diag = np.tile(stage, N + 1)
        Q_diag[-nx:] = terminal

        X_sol, U_sol, *_ = run_nmpc_soft(
            state, xr, N, Q_diag, R, u_bounds, x_bounds, dt, t, create_dynamics, t_x, t_x
        )
        u = U_sol[:, 0]

        # Step environment
        prev_state = state
        state, reward, done = env.step(u, t)

        # Update tensors
        cur_state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # Update return and timestep
        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), dtype=torch.long, device=device) * (t + 1)], dim=1
        )

        # Record episode data
        episode_return += reward
        episode_length += 1
        episode_data["observations"].append(prev_state)
        episode_data["inputs"].append(u)
        episode_data["rewards"].append(reward)
        episode_data["actions"].append(action)
        if eval_type == "mpc_decisioner":
            episode_data["attention"].append(current_attention)

        if done:
            break

    # --------------------------
    # Post-processing
    # --------------------------
    for key in episode_data:
        episode_data[key] = np.array(episode_data[key])

    # Compute return-to-go
    rtg = np.zeros_like(episode_data["rewards"])
    for t in reversed(range(len(rtg))):
        rtg[t] = episode_data["rewards"][t] + (rtg[t + 1] if t + 1 < len(rtg) else 0)
    episode_data["rtg"] = rtg

    return episode_return, episode_length, episode_data
