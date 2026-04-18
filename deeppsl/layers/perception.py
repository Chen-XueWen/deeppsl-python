import torch
import torch.nn as nn

class MLPPerception(nn.Module):
    """Simple MLP to predict atom truth values from input features."""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        layers.append(nn.Sigmoid()) # Truth values in [0, 1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNNPerception(nn.Module):
    """Simple CNN for image-based perception."""
    def __init__(self, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Assuming 28x28 input (MNIST-like)
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)
