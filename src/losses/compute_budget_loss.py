import torch
import torch.nn as nn

class ComputeBudgetLoss(nn.Module):
    def __init__(self, cost_weights=[0.0, 0.5, 1.0], target_budget=0.3):
        super().__init__()
        # cost_weights: Relative computational cost of each branch (0=CNN, 1=Light, 2=Deep)
        self.cost_weights = torch.tensor(cost_weights, dtype=torch.float32)
        self.target_budget = target_budget

    def forward(self, route_probs):
        """
        route_probs: (B, 3, H, W) probabilities of routing paths
        """
        self.cost_weights = self.cost_weights.to(route_probs.device)
        
        # Reshape cost_weights to (1, 3, 1, 1) to broadcast over spatial dimensions
        cost_weights = self.cost_weights.view(1, 3, 1, 1)
        
        # Expected cost per sample/region in batch
        expected_cost = torch.sum(route_probs * cost_weights, dim=1).mean()
        
        # Symmetric budget-deviation objective
        budget_loss = (expected_cost - self.target_budget) ** 2
        return budget_loss
