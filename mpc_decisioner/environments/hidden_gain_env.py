"""
hidden_gain_env.py  —  copy to mpc_decisioner/environments/

Jacketed CSTR with unknown heat transfer coefficient (fouling mode).

Physical setup:
    A -> B reaction in a stirred tank with a cooling jacket.
    The jacket heat transfer coefficient g switches between episodes:
        g = G_LO = 0.3  (fouled jacket: weak cooling)
        g = G_HI = 3.0  (clean jacket: strong cooling)
    The mode is unknown to the controller - it only sees x = [T_dev, C_dev].

Why DT > CQL and DT > RS:
    - RS  : fixed Q, must compromise between both modes -> suboptimal for both
    - CQL : at t=0 x2=0 always, modes are indistinguishable -> reacts too late
    - DT  : after K=5 context steps, history of (x, u, r) reveals g clearly
              -> switches to mode-optimal Q immediately

True dynamics  (nonlinear, hidden g):
    dx1 = x2
    dx2 = -(x1 + 0.01*t)^3  +  g * u

Nominal model  (linear, g=1, no t-dependency -- standard linearization):
    dx1 = x2
    dx2 = u
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from casadi import SX, vertcat, Function

G_LO = 0.3    # fouled jacket
G_HI = 3.0    # clean jacket


class HiddenGainEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.N        = 5
        self.nx       = 2
        self.nu       = 1
        self.dt       = 0.05
        self.q_diag   = np.array([10., 1.])
        self.r_diag   = np.array([0.1])
        self.u_bounds = np.array([-4., 4.])
        self.x_bounds = np.array([-2., 2.])
        self.w_x      = 10. * np.ones(self.nx)
        self.w_u      = 10. * np.ones(self.nu)

        self.action_space = spaces.Box(
            low=np.array([self.u_bounds[0]]),
            high=np.array([self.u_bounds[1]]),
            dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.full(self.nx, self.x_bounds[0]),
            high=np.full(self.nx, self.x_bounds[1]),
            dtype=np.float64)

        self.target_state = np.array([0., 0.])
        self.state = None
        self.g     = G_HI

    def _dynamics(self, x, u, t):
        x1, x2 = x
        dx1 = x2
        dx2 = -(x1 + 0.01 * t)**3 + self.g * u[0]
        return np.array([dx1, dx2])

    def linear_dynamics(self, t):
        """Double-integrator nominal model: dx1=x2, dx2=u."""
        nx, nu = self.nx, self.nu
        x = SX.sym('x', nx)
        u = SX.sym('u', nu)
        rhs = vertcat(x[1], u[0])
        return Function('f', [x, u], [rhs]), nx, nu

    def reset(self, eval_mode=0, custom_init=None):
        self.g = np.random.choice([G_LO, G_HI])
        base   = np.array([1.0, 0.0])
        noise  = np.random.uniform(-0.2, 0.2, 2)
        if eval_mode == 1:
            self.state = base.copy()
        elif eval_mode == 2:
            self.state = np.array(custom_init, dtype=float)
        else:
            self.state = base + noise
        return np.array(self.state, dtype=np.float64)

    def step(self, action, time_step):
        dx = self._dynamics(self.state, action, time_step)
        self.state = self.state + self.dt * dx
        reward = (self.q_diag[0] * (self.state[0] - self.target_state[0])**2
                + self.q_diag[1] * (self.state[1] - self.target_state[1])**2
                + self.r_diag[0] * action[0]**2)
        done = bool(
            self.state[0] < self.x_bounds[0] or
            self.state[0] > self.x_bounds[1] or
            self.state[1] < self.x_bounds[0] or
            self.state[1] > self.x_bounds[1])
        if done:
            reward += 1e6
        return np.array(self.state, dtype=np.float64), reward, done

    def render(self, mode='human'):
        print(f"state={self.state}  g={self.g}")

    def close(self):
        pass
