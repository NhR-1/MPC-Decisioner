# Some portions of this file are adapted from Decision Transformer:
#   https://github.com/kzl/decision-transformer
# Original work © 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors
#   https://arxiv.org/abs/2106.01345
# Licensed under the MIT License (see third_party/decision-transformer/LICENSE).

import os
import sys
import gymnasium as gym
import random
import pickle
import torch
import wandb
import argparse
import numpy as np

from mpc_decisioner.evaluation.evaluate_episodes import evaluate_nmpc
from mpc_decisioner.models.decision_transformer import DecisionTransformer
from mpc_decisioner.training.seq_trainer import SequenceTrainer
from mpc_decisioner.environments.nonlinear_env import NonlinearEnv


def discount_cumsum(x, gamma):
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def experiment(exp_prefix, variant):
    # --------------------------
    # Environment and device setup
    # --------------------------
    if "cuda_devices" in variant and variant["cuda_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(variant["cuda_devices"])
        print(f"Using CUDA_VISIBLE_DEVICES={variant['cuda_devices']}")
    else:
        print("Using default CUDA device visibility.")

    device = variant["device"]
    log_to_wandb = variant["log_to_wandb"]

    env_name = variant["env"]
    resume_iter = variant["resume_iter"]

    exp_id = random.randint(int(1e5), int(1e6) - 1)
    exp_name = f"{exp_prefix}-{env_name}-{exp_id}"

    # --------------------------
    # Environment setup
    # --------------------------
    if env_name == "nonlinear":
        env = NonlinearEnv()
        max_ep_len = 50
        env_targets = variant["target_returns"]
        scale = variant["scale"]
        terminal_param_scale = variant["terminal_param_scale"]
        param_scale = variant["param_scale"]
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    state_dim = env.observation_space.shape[0]

    # --------------------------
    # Parameter dimension setup 
    # --------------------------
    if variant["param_dim"] is not None:
        param_dim = variant["param_dim"]
        print(f"Using explicit param_dim = {param_dim}")
    else:
        param_dim = state_dim * variant["param_dim_multiplier"]
        print(f"Using param_dim = state_dim * {variant['param_dim_multiplier']} = {param_dim}")

    # --------------------------
    # Dataset loading
    # --------------------------
    dataset_path = variant["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    print("=" * 50)
    print(f"Environment: {env_name}")
    print(f"Trajectories: {len(traj_lens)} | Timesteps: {sum(traj_lens)}")
    print(f"Return mean: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Return range: {np.min(returns):.2f} - {np.max(returns):.2f}")
    print("=" * 50)

    # --------------------------
    # Model saving setup
    # --------------------------
    save_dir = variant["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = (
        os.path.join(save_dir, f"{env_name}_model_iter_{resume_iter}.pth")
        if resume_iter > 0
        else None
    )

    # --------------------------
    # Prepare training data
    # --------------------------
    K = variant["K"]
    batch_size = variant["batch_size"]
    pct_traj = variant["pct_traj"]

    num_timesteps = max(int(pct_traj * sum(traj_lens)), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories, timesteps = 1, traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=batch_size, max_len=K):
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        for _ in range(batch_size):
            traj = trajectories[int(np.random.choice(sorted_inds, p=p_sample))]
            si = random.randint(0, traj["rewards"].shape[0] - 1)
            s_t = traj["observations"][si : si + max_len]
            tlen = len(s_t)

            s_t = np.concatenate([np.zeros((max_len - tlen, state_dim)), s_t], axis=0)
            a_t = np.concatenate(
                [np.ones((max_len - tlen, param_dim)) * -10.0, traj["actions"][si : si + max_len]],
                axis=0,
            )
            r_t = np.concatenate(
                [np.zeros((max_len - tlen, 1)), traj["rewards"][si : si + max_len].reshape(-1, 1)],
                axis=0,
            )
            rtg_t = discount_cumsum(traj["rewards"][si:], gamma=1.0)[:tlen + 1].reshape(-1, 1)
            if rtg_t.shape[0] <= tlen:
                rtg_t = np.concatenate([rtg_t, np.zeros((1, 1))], axis=0)
            rtg_t = np.concatenate([np.zeros((max_len - tlen, 1)), rtg_t], axis=0) / scale
            t_t = np.concatenate([np.zeros(max_len - tlen), np.arange(si, si + tlen)], axis=0)
            m_t = np.concatenate([np.zeros(max_len - tlen), np.ones(tlen)], axis=0)

            s.append(((s_t - state_mean) / state_std)[None])
            a.append(a_t[None])
            r.append(r_t[None])
            rtg.append(rtg_t[None])
            timesteps.append(t_t[None])
            mask.append(m_t[None])

        to_torch = lambda x, dtype: torch.from_numpy(np.concatenate(x, axis=0)).to(dtype=dtype, device=device)
        return (
            to_torch(s, torch.float32),
            to_torch(a, torch.float32),
            to_torch(r, torch.float32),
            None,
            to_torch(rtg, torch.float32),
            to_torch(timesteps, torch.long),
            to_torch(mask, torch.bool),
        )

    # --------------------------
    # Evaluation function
    # --------------------------
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(variant["num_eval_episodes"]):
                with torch.no_grad():
                    ret, length, _  = evaluate_nmpc(
                        env,
                        env.linear_dynamics,
                        terminal_param_scale,
                        param_scale,
                        state_dim,
                        param_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
            }
        return fn

    # --------------------------
    # Model setup
    # --------------------------
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=param_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=variant["learning_rate"], weight_decay=variant["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1))

    # --------------------------
    # Resume checkpoint if available
    # --------------------------
    if resume_iter > 0 and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif resume_iter > 0:
        print(f"Warning: No checkpoint found for iter {resume_iter}, starting fresh.")

    # --------------------------
    # Trainer setup
    # --------------------------
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    if log_to_wandb:
        wandb.init(name=exp_name, project="mpc-decisioner", config=variant)

    # --------------------------
    # Training loop
    # --------------------------
    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"],
            iter_num=iter + 1,
            print_logs=True
        )

        if log_to_wandb:
            wandb.log(outputs)

        ckpt_path = os.path.join(save_dir, f"{env_name}_iter_{iter + resume_iter + 1}.pth")
        torch.save(
            {
                "iter_num": iter + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved at: {ckpt_path}")

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="nonlinear")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset pickle file")
    parser.add_argument("--save_dir", type=str, default="decisioner_logs")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=15)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_devices", type=str, default=None, help="Comma-separated list of GPU IDs (e.g., '0,1')")
    parser.add_argument("--scale", type=float, default=200.0)
    parser.add_argument("--target_returns", type=float, nargs="+", default=[40, 60, 80, 100, 120, 140])
    parser.add_argument("--terminal_param_scale", type=float, default=1000)
    parser.add_argument("--param_scale", type=float, default=20)
    parser.add_argument("--param_dim_multiplier", type=int, default=3,
                        help="Multiplier for state_dim to determine param_dim (used if --param_dim not given)")
    parser.add_argument("--param_dim", type=int, default=None,
                        help="Directly set action dimension (overrides multiplier)")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--resume_iter", type=int, default=0)

    args = parser.parse_args()
    experiment("mpc-exp", variant=vars(args))

