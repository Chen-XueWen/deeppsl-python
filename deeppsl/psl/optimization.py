import cvxpy as cp

def create_psl_problem(nc, ny, np_):
    """Creates a CVXPY problem for HL-MRF inference."""
    # Variables: unobserved ground atoms
    y = cp.Variable(ny)
    
    # Parameters: observed ground atoms, matrices, and weights
    p = cp.Parameter(np_)
    Ay = cp.Parameter((nc, ny))
    Ap = cp.Parameter((nc, np_))
    b = cp.Parameter(nc)
    w = cp.Parameter(nc, nonneg=True)
    
    # Clause violation: [Ay*y + Ap*p + b]_+
    violation = cp.pos(Ay @ y + Ap @ p + b)
    
    # Objective: Weighted Hinge Loss + Small L2 Regularization for stability
    objective = cp.Minimize(
        cp.sum(cp.multiply(w, cp.square(violation))) + 0.01 * cp.sum_squares(y)
    )
    
    # Constraints: truth values in [0, 1]
    constraints = [y >= 0, y <= 1]
    
    problem = cp.Problem(objective, constraints)
    return problem, [p, Ay, Ap, b, w], y
