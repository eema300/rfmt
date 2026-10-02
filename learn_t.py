'''
this is the mlp model that will learn t from z_t
* first iteration: vanila mlp
# second iteration: resnet
'''

import torch
import torch.nn as nn
import os


class RefinementT(nn.Module):

    def __init__(self, feature_dim, fabric, config,
                 hidden_dim=256, num_hidden_layers=3, dropout=0.1,):
        
        super().__init__()

        self.fabric = fabric
        self.config = config

        self.obj = nn.MSELoss()

        layers = []

        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

    def run_batch(self, X, t, opt):
        self.train(True)

        X = X.to(self.device)
        t = t.to(self.device)

        t_pred = self(X)
        loss = self.obj(t_pred, t) 

        opt.zero_grad()
        # loss.backward()
        self.fabric.backward(loss)
        opt.step()

        return loss.item()

    def save_checkpoint(self, loss, model_type):
        torch.save(self.state_dict(), os.path.join(self.config["mlp_out_dir"], f"{model_type}-best_rfmt_t.pt"))
        print(f" * >> saved new best model with loss: {loss}")

        


# normal MSE
# make sure to print in the training loop to keep track


class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        return self.relu(out)


class RefinementTResNet(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        num_hidden_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []

        input_dim = feature_dim + 1

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