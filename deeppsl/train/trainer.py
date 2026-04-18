import torch
import torch.nn as nn
import torch.optim as optim

class DeepPSLTrainer:
    """Utility to train DeepPSL models."""
    def __init__(self, model, lr=1e-3, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss() # Since truth values are in [0, 1]

    def train_step(self, x, y_true):
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        count = 0
        with torch.no_grad():
            for x, y_true in loader:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true)
                total_loss += loss.item()
                preds = (y_pred > 0.5).float()
                total_acc += (preds == y_true).float().mean().item()
                count += 1
        return total_loss / count, total_acc / count
