import torch.nn as nn

class DeepPSLModel(nn.Module):
    """Combines perception and PSL reasoning into an end-to-end model."""
    def __init__(self, perception, solver):
        super().__init__()
        self.perception = perception
        self.solver = solver

    def forward(self, x):
        # 1. Perception: x -> p (observed atoms)
        p = self.perception(x)
        # 2. Reasoning: p -> y (unobserved atoms)
        y = self.solver(p)
        return y
