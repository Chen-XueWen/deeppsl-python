import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from deeppsl.psl.optimization import create_psl_problem

try:
    from cvxpylayers.torch import CvxpyLayer
    HAS_CVXPYLAYERS = True
except ImportError:
    HAS_CVXPYLAYERS = False

class PSLSolver(nn.Module):
    """
    Differentiable PSL solver. 
    Falls back to non-differentiable CVXPY if cvxpylayers is missing.
    """
    def __init__(self, Ay, Ap, b, weights):
        super().__init__()
        self.nc, self.ny = Ay.shape
        _, self.np_ = Ap.shape
        
        self.problem, self.params, self.y_var = create_psl_problem(self.nc, self.ny, self.np_)
        
        if HAS_CVXPYLAYERS:
            self.cvxpy_layer = CvxpyLayer(self.problem, parameters=self.params, variables=[self.y_var])
        else:
            print("Warning: cvxpylayers not found. Backprop through solver is disabled.")
            self.cvxpy_layer = None
        
        self.register_buffer('Ay', torch.tensor(Ay, dtype=torch.float32))
        self.register_buffer('Ap', torch.tensor(Ap, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        self.raw_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, p):
        batch_size = p.shape[0]
        weights = torch.abs(self.raw_weights)
        
        if HAS_CVXPYLAYERS:
            Ay_b = self.Ay.unsqueeze(0).expand(batch_size, -1, -1)
            Ap_b = self.Ap.unsqueeze(0).expand(batch_size, -1, -1)
            b_b = self.b.unsqueeze(0).expand(batch_size, -1)
            weights_b = weights.unsqueeze(0).expand(batch_size, -1)
            y, = self.cvxpy_layer(p, Ay_b, Ap_b, b_b, weights_b)
            return y
        else:
            # Non-differentiable fallback
            y_results = []
            p_np = p.detach().cpu().numpy()
            Ay_np = self.Ay.cpu().numpy()
            Ap_np = self.Ap.cpu().numpy()
            b_np = self.b.cpu().numpy()
            w_np = weights.detach().cpu().numpy()
            
            # Set fixed parameters
            self.params[1].value = Ay_np
            self.params[2].value = Ap_np
            self.params[3].value = b_np
            self.params[4].value = w_np
            
            for i in range(batch_size):
                self.params[0].value = p_np[i]
                self.problem.solve()
                # If solver fails, return zeros
                if self.y_var.value is None:
                    y_results.append(np.zeros(self.ny))
                else:
                    y_results.append(self.y_var.value)
            
            # Create a tensor that allows gradients (though solver part is detached)
            y_tensor = torch.tensor(np.array(y_results), dtype=torch.float32, device=p.device)
            return y_tensor
