import os
import pickle
import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, Function, nlpsol

# === Import Environments and MPC Solver ===
from mpc_decisioner.environments.nonlinear_env import NonlinearEnv
from mpc.mpc import run_nmpc_soft


def generate_dataset(
    env_name: str = "NonlinearEnv",
    param_scale: float = 20,
    terminal_param_scale: float = 1000,
    output_file_mpc: str = "dataset_mpc_decisioner.pkl",
    output_file_cql: str = "dataset_cql.pkl",
):
    """
    Generate NMPC dataset using the specified environment and scaling parameters.

    Parameters
    ----------
    env_name : str
        Name of the environment class to use ("NonlinearEnv" by default).
    param_scale : float
        Scaling factor for non-terminal MPC parameters (default 20).
    terminal_param_scale : float
        Scaling factor for terminal MPC parameters (default 1000).
    output_file_mpc : str
        File name to save the MPC decisioner dataset.
    output_file_cql : str
        File name to save the CQL dataset.
    """

    # === Configuration ===
    seed = 42
    NUM_EPISODES = 200
    MAX_EPISODE_LENGTH = 50

    # === Environment Setup ===
    if env_name == "NonlinearEnv":
        env = NonlinearEnv()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    dt = env.dt
    nx = env.nx
    nu = env.nu
    N = env.N
    R = np.diag(env.r_diag[:])
    xr = env.target_state
    u_bounds = env.u_bounds
    x_bounds = env.x_bounds

    all_episodes_mpc_decisioner = []
    all_episodes_cql = []

    # Track which intervals have been included
    intervals_included = set()
    start, end, interval_step = 90, 190, 2
    max_intervals = (end - start) // interval_step

    # === Main Data Collection Loop ===
    np.random.seed(seed)
    for ep in range(NUM_EPISODES):
        x = env.reset(0)
        episode_data_mpc_decisioner = collections.defaultdict(list)
        episode_data_cql = collections.defaultdict(list)

        # --- Randomization for MPC parameters ---
        lows = np.array([-2, -0.5], dtype=float)
        highs = np.array([5, 1.0], dtype=float)
        q_diag = env.q_diag + np.random.uniform(low=lows, high=highs)
        q_diag_terminal = np.random.randint(10, 1000, size=nx)
        t_x = np.random.uniform(-0.5, 0.5, size=nx)

        Q_combined = np.tile(q_diag, N + 1)
        Q_combined[-nx:] = q_diag_terminal

        # --- Target parameters ---
        q_diag_target = env.q_diag + np.random.uniform(low=lows, high=highs)
        q_diag_terminal_target = np.random.randint(10, 1000, size=nx)
        t_x_target = np.random.uniform(-0.5, 0.5, size=nx)

        eta = np.random.uniform(0.0, 0.5)
        transition_timer = 0
        transition_time = np.random.uniform(8, 14)

        # === Episode Simulation ===
        for step in range(MAX_EPISODE_LENGTH):
            if transition_timer > transition_time:
                transition_timer = 0
                transition_time = np.random.uniform(8, 14)
                q_diag_target = env.q_diag + np.random.uniform(low=lows, high=highs)
                q_diag_terminal_target = np.random.randint(10, 1000, size=nx)
                t_x_target = np.random.uniform(-0.5, 0.5, size=nx)

            # --- Run NMPC ---
            X_sol, U_sol, *_ = run_nmpc_soft(
                x_init=x,
                xr=xr,
                N=N,
                Q_diag=Q_combined,
                R=R,
                u_bounds=u_bounds,
                x_bounds=x_bounds,
                dt=dt,
                step=step,
                create_dynamics=env.linear_dynamics,
                t_x=t_x,
                t_xN=t_x
            )

            u = U_sol[:, 0]
            x_next, reward, done = env.step(u, step)
            norm_Q_combined = (1 / param_scale) * np.tile(q_diag, 2)
            norm_Q_combined[-nx:] = (1 / terminal_param_scale) * q_diag_terminal

            # --- Store data ---
            episode_data_mpc_decisioner["observations"].append(x)
            episode_data_mpc_decisioner["inputs"].append(u)
            episode_data_mpc_decisioner["rewards"].append(reward)
            episode_data_mpc_decisioner["next_observations"].append(x_next)
            episode_data_mpc_decisioner["actions"].append(np.concatenate([norm_Q_combined, t_x]))
            episode_data_mpc_decisioner["terminals"].append(step == MAX_EPISODE_LENGTH - 1)

            episode_data_cql["observations"].append(x)
            episode_data_cql["inputs"].append(u)
            episode_data_cql["rewards"].append(-reward)
            episode_data_cql["next_observations"].append(x_next)
            episode_data_cql["actions"].append(np.concatenate([norm_Q_combined, t_x]))
            episode_data_cql["terminals"].append(step == MAX_EPISODE_LENGTH - 1)

            # --- Smooth transition between parameters ---
            eta = 3 * (transition_timer / transition_time) ** 2 - 2 * (transition_timer / transition_time) ** 3
            q_diag = (1 - eta) * q_diag + eta * q_diag_target
            q_diag_terminal = (1 - eta) * q_diag_terminal + eta * q_diag_terminal_target
            t_x = (1 - eta) * t_x + eta * t_x_target

            Q_combined = np.tile(q_diag, N + 1)
            Q_combined[-nx:] = q_diag_terminal

            transition_timer += 1
            x = x_next

        # === Convert lists to arrays ===
        for key in episode_data_mpc_decisioner:
            episode_data_mpc_decisioner[key] = np.array(episode_data_mpc_decisioner[key])
        for key in episode_data_cql:
            episode_data_cql[key] = np.array(episode_data_cql[key])

        # === Compute Reward-to-Go (RTG) ===
        rtg = np.zeros_like(episode_data_mpc_decisioner["rewards"])
        for t in reversed(range(len(rtg))):
            rtg[t] = episode_data_mpc_decisioner["rewards"][t] + (rtg[t + 1] if t + 1 < len(rtg) else 0)
        episode_data_mpc_decisioner["rtg"] = rtg

        rtg0 = rtg[0]

        # === Select episodes based on return intervals ===
        interval_key = None
        for low in range(start, end, interval_step):
            high = low + interval_step
            if low <= rtg0 < high:
                interval_key = f"{low}_{high}"
                break

        if interval_key is not None and interval_key not in intervals_included:
            all_episodes_mpc_decisioner.append(episode_data_mpc_decisioner)
            all_episodes_cql.append(episode_data_cql)
            intervals_included.add(interval_key)

        # Stop early if all intervals are filled
        if len(intervals_included) >= max_intervals:
            print("\nAll return intervals covered — stopping early.")
            break

    # === Save Datasets ===
    with open(output_file_mpc, "wb") as f:
        pickle.dump(all_episodes_mpc_decisioner, f)

    with open(output_file_cql, "wb") as f:
        pickle.dump(all_episodes_cql, f)

    # === Summary Statistics ===
    returns = np.array([np.sum(p['rewards']) for p in all_episodes_mpc_decisioner])
    num_samples = np.sum([p['rewards'].shape[0] for p in all_episodes_mpc_decisioner])

    print(f"\nNumber of samples collected: {num_samples}")
    print(f"Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, "
          f"max = {np.max(returns):.2f}, min = {np.min(returns):.2f}")
    print(f"\nSaved datasets to:\n - {output_file_mpc}\n - {output_file_cql}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NMPC training datasets.")
    parser.add_argument("--env_name", type=str, default="NonlinearEnv", help="Environment name (default: NonlinearEnv)")
    parser.add_argument("--param_scale", type=float, default=20, help="Scaling factor for MPC parameters (default: 20)")
    parser.add_argument("--terminal_param_scale", type=float, default=1000, help="Scaling factor for terminal MPC parameters (default: 1000)")
    parser.add_argument("--output_file_mpc", type=str, default="dataset_mpc_decisioner.pkl", help="Output file name for MPC dataset")
    parser.add_argument("--output_file_cql", type=str, default="dataset_cql.pkl", help="Output file name for CQL dataset")
    args = parser.parse_args()

    generate_dataset(
        env_name=args.env_name,
        param_scale=args.param_scale,
        terminal_param_scale=args.terminal_param_scale,
        output_file_mpc=args.output_file_mpc,
        output_file_cql=args.output_file_cql
    )
