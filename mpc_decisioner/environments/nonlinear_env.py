import numpy as np
import gymnasium as gym
from gymnasium import spaces
from casadi import SX, vertcat, Function, nlpsol, sin, cos, if_else


class NonlinearEnv(gym.Env):

    def __init__(self):
        super(NonlinearEnv, self).__init__()       
        self.N = 5  # Horizon length
        self.nx = 2 # Number of states
        self.nu = 1 # Number of control inputs
        # State and input weighting for cost
        self.q_diag = np.array([10, 1])
        self.r_diag = np.array([0.1])
        self.dt = 0.05 # Sampling time
        # Input and state constraints
        self.u_bounds = np.array([-4, 4])
        self.x_bounds = np.array([-2, 2])
        # Additional weights for constraint violation
        self.w_x = 10 * np.ones(self.nx)
        self.w_u = 10 * np.ones(self.nu)
        
        # Action space: 
        self.action_space = spaces.Box(low=np.array([self.u_bounds[0]]),  
                                       high=np.array([self.u_bounds[1]]),
                                       dtype=np.float64)
        # Observation space: 
        self.observation_space = spaces.Box(low=np.array([self.x_bounds[0], self.x_bounds[0]]),
                                            high=np.array([self.x_bounds[1], self.x_bounds[1]]),  
                                            dtype=np.float64)

        # Target state (regulation)
        self.target_state = np.array([0.0, 0.0])

        self.state = None


    def _dynamics(self, x, u, time_step):
        """Nonlinear dynamics"""
        x1, x2 = x
        if time_step <20:
            dx1 = x2 
            dx2 = -(x1+0.01*time_step)**3 + 1.5*u[0] 
        elif time_step >= 20 and time_step < 35:
            dx1 = x2 
            dx2 = -(x1+0.01*time_step)**3 - 0.1*u[0]  
        else:
            dx1 = x2 
            dx2 = -(x1+0.01*time_step)**3 + 1.5*u[0]        
        return np.array([dx1, dx2])

    def linear_dynamics(self,time_step):
        nx = self.nx
        nu = self.nu
        x = SX.sym("x", nx)
        u = SX.sym("u", nu)
        dx1 = x[1]
        dx2 = u[0]
        rhs = vertcat(dx1, dx2)
        f = Function("f", [x, u], [rhs])
        return f, nx, nu
    
    def original_dynamics(self,time_step):
        nx = self.nx
        nu = self.nu
        x = SX.sym("x", nx)
        u = SX.sym("u", nu)
        step = SX.sym("step")  
    
        dx1 = x[1]
        dx2_case1 = -(x[0]+0.01*step )**3+ (1.5) * u[0]
        dx2_case2 = -(x[0]+0.01*step )**3 - (0.1) * u[0]
    
        dx2 = if_else(step < 20,
                      dx2_case1,
                      if_else(step < 35, dx2_case2, dx2_case1))
    
        rhs = vertcat(dx1, dx2)
        f = Function("f", [x, u, step], [rhs])
        return f, nx, nu

    def step(self, action, time_step):
        dx = self._dynamics(self.state, action, time_step)
        self.state = self.state + self.dt * dx
        reward = self.q_diag[0] * (self.state[0]-self.target_state[0])**2 + self.q_diag[1] * (self.state[1]-self.target_state[1])**2 + self.r_diag[0] * action[0]**2
        done = bool(self.state[0] < self.x_bounds[0] or self.state[0] > self.x_bounds[1]  or self.state[1] < self.x_bounds[0] or self.state[1] > self.x_bounds[1] or action[0] < self.u_bounds[0] or action[0] > self.u_bounds[1])
        if done:
            reward += 1e6  
        return np.array(self.state, dtype=np.float64), reward, done

    def reset(self, eval_mode,custom_init=None):
        self.init_state = 0.5 * np.array([2.0, 0.0])  
        noise = np.random.uniform(low=-0.2, high=0.2, size=(2,))
        if eval_mode == 1:
            self.state = self.init_state 
        elif eval_mode == 2:
            self.state = custom_init
        else:
            self.state = self.init_state + noise   
        return np.array(self.state, dtype=np.float64)

    def render(self, mode='human'):
        x1, x2 = self.state
        print(f"Concentration x1: {x1:.3f}, Temperature x2: {x2:.3f}")

    def close(self):
        pass
