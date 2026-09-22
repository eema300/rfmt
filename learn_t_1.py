'''
this is the mlp model that will learn t from z_t
'''

import torch
import torch.nn as nn


class RefinementT(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        num_hidden_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []

        input_dim = feature_dim + 1 # feature size for codes? and t

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_features, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x = torch.cat([x_features, t], dim=-1)

        return self.network(x)