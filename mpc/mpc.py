from casadi import Function, SX, DM, dot, vertcat, nlpsol
import numpy as np

def run_nmpc_soft(
    x_init,
    xr,
    N,
    Q_diag,
    R,
    u_bounds,
    x_bounds,
    dt,
    step,
    create_dynamics,
    t_x=None,
    t_xN=None,
    t_u=None,
    w_x=None,
    w_xN=None,
    w_u=None,
):
    """
    Nonlinear Model Predictive Control (NMPC) with soft state and input constraints.

    Formulation uses slack variables to soften state and input constraints, adding
    linear penalties to the cost function. The objective function includes:

        J = Σ (x_t - x_r)^T Q_t (x_t - x_r) + u_t^T R u_t
            + w_x^T (σ_x_lower + σ_x_upper)
            + w_u^T (σ_u_lower + σ_u_upper)
            + terminal cost + terminal slack penalties

    Args:
        x_init: Initial state vector.
        xr: Reference (target) state.
        N: Prediction horizon.
        Q_diag: Flattened sequence of state cost weights.
        R: Control input cost matrix.
        u_bounds: Tuple/list of (lower, upper) input bounds.
        x_bounds: Tuple/list of (lower, upper) state bounds.
        dt: Integration step size.
        step: Current time step index.
        create_dynamics: Function returning system dynamics (f, nx, nu).
        t_x, t_xN, t_u: Optional bias terms for constraints.
        w_x, w_xN, w_u: Slack variable penalty weights.

    Returns:
        Tuple containing:
            (X_sol, U_sol,
             Sigma_x_lower_sol, Sigma_x_upper_sol,
             Sigma_u_lower_sol, Sigma_u_upper_sol,
             Sigma_xN_lower_sol, Sigma_xN_upper_sol,
             cost_value)
    """

    # --------------------------
    # Setup and parameters
    # --------------------------
    f, nx, nu = create_dynamics(step)
    X = SX.sym("X", nx, N + 1)
    U = SX.sym("U", nu, N)

    # Slack variables
    Sigma_x_lower = SX.sym("Sigma_x_lower", nx, N)
    Sigma_x_upper = SX.sym("Sigma_x_upper", nx, N)
    Sigma_u_lower = SX.sym("Sigma_u_lower", nu, N)
    Sigma_u_upper = SX.sym("Sigma_u_upper", nu, N)
    Sigma_xN_lower = SX.sym("Sigma_xN_lower", nx)
    Sigma_xN_upper = SX.sym("Sigma_xN_upper", nx)

    # Default penalty weights and bias terms
    if w_x is None:
        w_x = 10 * np.ones(nx)
    if w_u is None:
        w_u = 10 * np.ones(nu)
    if w_xN is None:
        w_xN = 10 * np.ones(nx)
    if t_x is None:
        t_x = np.zeros(nx)
    if t_u is None:
        t_u = np.zeros(nu)
    if t_xN is None:
        t_xN = np.zeros(nx)

    # Convert to CasADi DM format
    w_x, w_u, w_xN = DM(w_x), DM(w_u), DM(w_xN)
    t_x, t_u, t_xN = DM(t_x), DM(t_u), DM(t_xN)

    # --------------------------
    # Optimization formulation
    # --------------------------
    cost = 0
    g, lbg, ubg = [], [], []
    BIG = 1e6  # Large upper bound constant

    # Initial condition (equality constraint)
    g.append(X[:, 0] - x_init)
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    # --------------------------
    # Stage loop
    # --------------------------
    for t in range(N):
        Qt = DM(np.diag(Q_diag[t * nx : (t + 1) * nx]))
        xt_err = X[:, t] - xr

        # Quadratic stage cost
        cost += xt_err.T @ Qt @ xt_err + U[:, t].T @ R @ U[:, t]

        # Slack penalties
        cost += dot(w_x, Sigma_x_lower[:, t]) + dot(w_x, Sigma_x_upper[:, t])
        cost += dot(w_u, Sigma_u_lower[:, t]) + dot(w_u, Sigma_u_upper[:, t])

        # Dynamics constraint
        x_next = X[:, t] + dt * f(X[:, t], U[:, t])
        g.append(X[:, t + 1] - x_next)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # State bounds (soft)
        g.append(X[:, t] - x_bounds[0] + Sigma_x_lower[:, t] + t_x)
        g.append(x_bounds[1] - X[:, t] + Sigma_x_upper[:, t] + t_x)
        lbg += [0.0] * (2 * nx)
        ubg += [BIG] * (2 * nx)

        # Input bounds (soft)
        g.append(U[:, t] - u_bounds[0] + Sigma_u_lower[:, t])
        g.append(u_bounds[1] - U[:, t] + Sigma_u_upper[:, t])
        lbg += [0.0] * (2 * nu)
        ubg += [BIG] * (2 * nu)

    # --------------------------
    # Terminal cost and constraints
    # --------------------------
    QN = DM(np.diag(Q_diag[N * nx : (N + 1) * nx]))
    xN_err = X[:, N] - xr
    cost += xN_err.T @ QN @ xN_err
    cost += dot(w_xN, Sigma_xN_lower) + dot(w_xN, Sigma_xN_upper)

    # Terminal bounds (soft)
    g.append(X[:, N] - x_bounds[0] + Sigma_xN_lower + t_xN)
    g.append(x_bounds[1] - X[:, N] + Sigma_xN_upper + t_xN)
    lbg += [0.0] * (2 * nx)
    ubg += [BIG] * (2 * nx)

    # --------------------------
    # Decision variables and solver setup
    # --------------------------
    vars = vertcat(
        X.reshape((-1, 1)),
        U.reshape((-1, 1)),
        Sigma_x_lower.reshape((-1, 1)),
        Sigma_x_upper.reshape((-1, 1)),
        Sigma_u_lower.reshape((-1, 1)),
        Sigma_u_upper.reshape((-1, 1)),
        Sigma_xN_lower.reshape((-1, 1)),
        Sigma_xN_upper.reshape((-1, 1)),
    )

    g_flat = vertcat(*g)

    # Variable bounds
    lbx = [-1e6] * (nx * (N + 1) + nu * N)
    ubx = [1e6] * (nx * (N + 1) + nu * N)

    # Slack variable bounds (>= 0)
    n_slack = (nx * N) * 2 + (nu * N) * 2 + 2 * nx
    lbx += [0.0] * n_slack
    ubx += [1e6] * n_slack

    # IPOPT solver setup
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-6}
    solver = nlpsol("solver", "ipopt", {"x": vars, "f": cost, "g": g_flat}, opts)

    # --------------------------
    # Solve optimization problem
    # --------------------------
    sol = solver(x0=np.zeros(vars.shape[0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    sol_vars = np.array(sol["x"]).flatten()

    # --------------------------
    # Extract and reshape solution
    # --------------------------
    nX = nx * (N + 1)
    nU = nu * N
    nSxl, nSxu, nSul, nSuu = nx * N, nx * N, nu * N, nu * N
    nSxNl, nSxNu = nx, nx

    X_sol = sol_vars[:nX].reshape((nx, N + 1), order="F")
    U_sol = sol_vars[nX : nX + nU].reshape((nu, N), order="F")

    offset = nX + nU
    Sigma_x_lower_sol = sol_vars[offset : offset + nSxl].reshape((nx, N), order="F")
    Sigma_x_upper_sol = sol_vars[offset + nSxl : offset + nSxl + nSxu].reshape((nx, N), order="F")
    Sigma_u_lower_sol = sol_vars[offset + nSxl + nSxu : offset + nSxl + nSxu + nSul].reshape((nu, N), order="F")
    Sigma_u_upper_sol = sol_vars[offset + nSxl + nSxu + nSul : offset + nSxl + nSxu + nSul + nSuu].reshape((nu, N), order="F")

    Sigma_xN_lower_sol = sol_vars[-(nSxNl + nSxNu) : -nSxNu].reshape((nx, 1))
    Sigma_xN_upper_sol = sol_vars[-nSxNu:].reshape((nx, 1))

    # --------------------------
    # Return results
    # --------------------------
    return (
        X_sol,
        U_sol,
        Sigma_x_lower_sol,
        Sigma_x_upper_sol,
        Sigma_u_lower_sol,
        Sigma_u_upper_sol,
        Sigma_xN_lower_sol,
        Sigma_xN_upper_sol,
        float(sol["f"]),
    )
