import os
import random
import pickle
import argparse
import numpy as np
import torch
import d3rlpy

from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset

from mpc_decisioner.evaluation.evaluate_episodes import evaluate_nmpc
from mpc_decisioner.environments.nonlinear_env import NonlinearEnv

def set_global_seed(seed, use_cuda=True):
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA deterministic mode

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if use_cuda and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------- Helpers ----------------------

class CQLasDT:
    """Adapter that lets CQL behave like a Decision Transformer-style policy."""
    def __init__(self, cql, state_mean, state_std, device="cpu"):
        self.cql = cql
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.device = device
        self._training = False

    def eval(self):
        self._training = False
        return self

    def train(self):
        self._training = True
        return self

    def to(self, device=None):
        if device is not None:
            self.device = device
        return self

    def state_dict(self):
        return {}

    def parameters(self):
        return []

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor = None,
        rewards: torch.Tensor = None,
        target_return: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        **kwargs,
    ):
        """Predict an action given normalized states."""
        if states.ndim == 1:
            s_t = states
        elif states.ndim == 2:
            s_t = states[-1]
        else:  # (B, T, D)
            s_t = states[0, -1]

        s = s_t.detach().cpu().numpy().astype(np.float32)
        s_batched = np.expand_dims(s, axis=0)
        a = self.cql.predict(s_batched)[0]
        return torch.from_numpy(a).to(states.device, dtype=torch.float32)


def make_env(env_name):
    """Create environment based on name."""
    if env_name == "nonlinear":
        env = NonlinearEnv()
        max_ep_len = 50
        scale = 200.0
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env, max_ep_len, scale


def set_seed(seed, env):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass


def ensure_terminals(traj_len, traj_dict):
    """Ensure trajectory has terminal flags."""
    if "terminals" in traj_dict:
        return traj_dict["terminals"].astype(bool)
    if "dones" in traj_dict:
        return traj_dict["dones"].astype(bool)
    term = np.zeros((traj_len,), dtype=bool)
    term[-1] = True
    return term


def compute_state_stats(trajectories):
    """Compute mean and std of all observations."""
    states = np.concatenate([t["observations"] for t in trajectories], axis=0)
    mean = states.mean(axis=0).astype(np.float32)
    std = (states.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def build_mdp_from_pickle(trajectories, state_mean, state_std):
    """Convert trajectory list (DT style) into MDPDataset for d3rlpy."""
    obs_list, act_list, rew_list, term_list = [], [], [], []
    for traj in trajectories:
        o = traj["observations"].astype(np.float32)
        a = traj["actions"].astype(np.float32)
        r = traj["rewards"].astype(np.float32).reshape(-1)
        T = o.shape[0]
        t = ensure_terminals(T, traj)
        o = (o - state_mean) / state_std
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        term_list.append(t)

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    terminals = np.concatenate(term_list, axis=0)

    return MDPDataset(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rewards.astype(np.float32),
        terminals=terminals.astype(np.bool_),
    )


# ---------------------- Main Experiment ----------------------

def run_cql_experiment(args):
    # === Full reproducibility ===
    set_global_seed(args.seed, use_cuda=args.gpu)
    d3rlpy.seed(args.seed)

    # === Environment setup ===
    env, max_ep_len, scale = make_env(args.env)
    set_seed(args.seed, env)

    env_targets = args.target_returns
    action_gain = args.terminal_param_scale
    action_q_gain = args.param_scale

    # === Dataset loading ===
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    with open(args.dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # === Normalization ===
    state_mean, state_std = compute_state_stats(trajectories)
    mdp = build_mdp_from_pickle(trajectories, state_mean, state_std)

    # === Device selection ===
    device = "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu"

    # === CQL configuration (matches published table) ===
    cfg = CQLConfig(
        batch_size=args.batch_size,            
        gamma=args.gamma,                      
        actor_learning_rate=args.lr,           
        critic_learning_rate=args.lr,          
        temp_learning_rate=args.temp_lr,       
        initial_temperature=args.temperature,  
        tau=args.tau,                          
        n_critics=args.n_critics,              
        conservative_weight=args.alpha,        
        n_action_samples=64,
        action_scaler=None,
        reward_scaler=None,
    )

    # Define network architecture
    cfg.q_func_factory_kwargs = dict(hidden_units=[256, 256], activation="relu")

    cql = cfg.create(device=device)
    adapter = CQLasDT(cql, state_mean, state_std, device=device)

    # === Directories ===
    os.makedirs(args.save_dir, exist_ok=True)
    exp_dir = os.path.join(args.save_dir, f"{args.env}_seed{args.seed}")
    os.makedirs(exp_dir, exist_ok=True)

    total_steps = args.steps
    steps_per_epoch = args.n_steps_per_epoch
    n_full_epochs = total_steps // steps_per_epoch
    last_chunk = total_steps - n_full_epochs * steps_per_epoch
    n_epochs = n_full_epochs + (1 if last_chunk > 0 else 0)

    state_dim = env.observation_space.shape[0]
    if args.param_dim is not None:
        param_dim = args.param_dim
        print(f"Using explicit param_dim = {param_dim}")
    else:
        param_dim = state_dim * args.param_dim_multiplier
        print(f"Using param_dim = state_dim * {args.param_dim_multiplier} = {param_dim}")
    device_str = "cuda" if device.startswith("cuda") else "cpu"

    print(f"\nStarting deterministic CQL training on {args.env.upper()} [{device_str}]")
    for ep in range(1, n_epochs + 1):
        steps_this = steps_per_epoch if ep <= n_full_epochs else last_chunk

        cql.fit(
            mdp,
            n_steps=steps_this,
            n_steps_per_epoch=steps_this,
            experiment_name=exp_dir,
            with_timestamp=False,
        )

        # --- Evaluation after each epoch ---
        logs = {}
        for tar in env_targets:
            rets = []
            for _ in range(args.eval_episodes):
                ret, _, _ = evaluate_nmpc(
                    env, env.linear_dynamics, action_gain, action_q_gain,
                    state_dim, param_dim, adapter,
                    max_ep_len=max_ep_len, scale=scale,
                    state_mean=state_mean, state_std=state_std,
                    device=device_str, target_return=tar / scale,
                    eval_mode=1, eval_type="cql",
                )
                rets.append(ret)
            logs[f"target_{tar}_return_mean"] = float(np.mean(rets))

        print(f"\nEvaluation after epoch {ep}/{n_epochs}:")
        for k, v in logs.items():
            print(f"  {k}: {v:.3f}")

        # --- Save checkpoint ---
        ep_dir = os.path.join(exp_dir, f"epoch_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        cql.save_model(os.path.join(ep_dir, "cql_model.d3"))

    # === Final save ===
    cql.save_model(os.path.join(exp_dir, "cql_model_last.d3"))
    np.savez(os.path.join(exp_dir, "norm_stats.npz"), state_mean=state_mean, state_std=state_std)
    print(f"\nTraining completed. Results saved to {exp_dir}\n")


# ---------------------- CLI Entry ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deterministic CQL training for MPC Decisioner datasets.")
    parser.add_argument("--env", type=str, default="nonlinear", choices=["nonlinear"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset pickle file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--n_steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for actor and critic.")
    parser.add_argument("--temp_lr", type=float, default=3e-5, help="Temperature learning rate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Initial temperature value.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate.")
    parser.add_argument("--n_critics", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=5.0, help="CQL penalty strength.")
    parser.add_argument("--save_dir", type=str, default="cql_logs")
    parser.add_argument("--terminal_param_scale", type=float, default=1000)
    parser.add_argument("--param_scale", type=float, default=20)
    parser.add_argument("--target_returns", type=float, nargs="+", default=[0])
    parser.add_argument("--param_dim_multiplier", type=int, default=3,
                        help="Multiplier for state_dim to determine param_dim (used if --param_dim not given)")
    parser.add_argument("--param_dim", type=int, default=None,
                        help="Directly set action dimension (overrides multiplier)")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    # === Default: use GPU if available ===
    args.gpu = not args.no_gpu and torch.cuda.is_available()

    run_cql_experiment(args)
