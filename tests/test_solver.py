import torch
import numpy as np
from deeppsl.psl.solver import PSLSolver

def test_solver_simple():
    # Rule: p1 -> p2 (weight 100)
    # Ay*y + Ap*p + b = -1*p2 + 1*p1 + 0
    Ay = np.array([[-1.0]])
    Ap = np.array([[1.0]])
    b = np.array([0.0])
    weights = np.array([100.0])
    
    solver = PSLSolver(Ay, Ap, b, weights)
    
    # If p1 is high, p2 should be high
    p = torch.tensor([[0.8]], dtype=torch.float32)
    y = solver(p)
    assert y.item() > 0.75
    
    # If p1 is low, p2 could be anything (but QP will likely push to 0 if no other rules)
    p = torch.tensor([[0.1]], dtype=torch.float32)
    y = solver(p)
    assert y.item() < 0.2
