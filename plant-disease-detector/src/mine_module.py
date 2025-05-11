import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_size=128):
        super().__init__()
        self.fc1_x = nn.Linear(dim_x, hidden_size)
        self.fc1_y = nn.Linear(dim_y, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        h = self.relu(self.fc1_x(x) + self.fc1_y(y))
        return self.fc2(h)

def mine_loss(joint, marginal):
    joint_mean = torch.mean(joint)
    marg_exp = torch.mean(torch.exp(marginal))
    return -(joint_mean - torch.log(marg_exp + 1e-6))
