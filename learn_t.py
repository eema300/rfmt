'''
this is the mlp model that will learn t from z_t
* first iteration: vanila mlp
# second iteration: resnet
'''

import torch
import os
import copy
import csv
import torch.nn as nn
from tqdm import tqdm


class RefinementT(nn.Module):

    def __init__(self, feature_dim, fabric, config,
                 hidden_dim=256, num_hidden_layers=3, dropout=0.1,):
        
        super().__init__()

        self.fabric = fabric
        self.config = config

        self.obj = nn.MSELoss()
        self.is_fit = False
        self.best_train_loss = 0.0
        self.plot_mse = {}

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

    # collection_type: "interpolate" or "sample"
    def run_batch(self, X, t, opt, collection_type: str, epoch=None):
        self.train(True)

        X = X.to(self.fabric.device)
        t = t.to(self.fabric.device)

        t_pred = self(X)
        self.print_pred(t_pred=t_pred, t_actual=t,
                        path_name=self.config["csv_dir"],
                        filename=f"{collection_type}_train.csv", epoch=epoch)
        loss = self.obj(t_pred, t) 

        opt.zero_grad()
        self.fabric.backward(loss)
        opt.step()

        # doing .item() prevents memory leaks
        return loss.item()

    def save_checkpoint(self, loss, model_type):
        torch.save(self.state_dict(), os.path.join(self.config["mlp_out_dir"], f"{model_type}-best_rfmt_t.pt"))
        print(f" * >> saved new best model with loss: {loss}")

    # if training data readily available as a whole (sample_collect.py)
    def fit(self, train_data_loader, epochs: int, opt: torch.optim.Optimizer):
        best_loss = float("inf")
        best_model = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_examples = 0
            new_best_flag = False

            for batch in tqdm(train_data_loader):
                batch_loss = self.run_batch(X=batch[0], t=batch[1], opt=opt,
                                            collection_type="sample", epoch=epoch)
                epoch_loss += batch_loss * batch[0].shape[0]
                num_examples += batch[0].shape[0]

            if num_examples != 0:
                epoch_loss /= num_examples
            print(f"epoch {epoch} loss: {epoch_loss}")

            if epoch_loss < best_loss:
                self.save_checkpoint(loss=epoch_loss, model_type="sample")
                best_model = copy.deepcopy(self.state_dict())
                best_loss = epoch_loss
                new_best_flag = True

            # for plotting
            self.plot_mse[epoch] = (epoch_loss, new_best_flag)
                
        self.is_fit = True
        self.best_train_loss = best_loss

        self.load_state_dict(best_model)
        return self

    # use on either sample_collect.py or data_collect.py 
    # TODO: rename data_collect.py -> interpolate_collect.py
    # collection_type: "interpolate" or "sample"
    def evaluate(self, val_data_loader, collection_type: str):
        total_loss = 0.0
        num_examples = 0

        # set model to eval mode
        self.eval()

        with torch.no_grad():
            for batch in tqdm(val_data_loader):
                # set to device
                batch[0] = batch[0].to(self.fabric.device)
                batch[1] = batch[1].to(self.fabric.device)

                # make prediction
                t_pred = self(batch[0])
                self.print_pred(t_pred=t_pred, t_actual=batch[1],
                                path_name=self.config["csv_dir"],
                                filename=f"{collection_type}_validation.csv")
                
                # accumulate loss
                loss = (self.obj(t_pred, batch[1])).item()
                total_loss += loss * batch[0].shape[0]
                num_examples += batch[0].shape[0]

        total_loss = total_loss / num_examples

        return total_loss

    @staticmethod
    def print_pred(t_pred, t_actual, path_name, filename, epoch=None):
        full_path = os.path.join(path_name, filename)
        
        # make the dim (batch_size, 2)
        data = torch.cat((t_pred, t_actual), dim=1)

        # convert to list
        data = data.detach().cpu().tolist()

        file_exists = os.path.exists(full_path) and os.path.getsize(full_path) > 0

        with open(full_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                if epoch is None:
                    writer.writerow(["t_pred", "t_actual"])
                else:
                    writer.writerow(["epoch", "t_pred", "t_actual"])

            for t_p, t_a in data:
                if epoch is None:
                    writer.writerow([t_p, t_a])
                else:
                    writer.writerow([epoch, t_p, t_a])


def load_checkpoint(chkpt_path, feat_dim, fabric, config, hidden_dim=256,
                    num_hidden_layers=3, dropout=0.1):

    model = RefinementT(
        feature_dim=feat_dim,
        fabric=fabric,
        config=config,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    state_dict = torch.load(chkpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


# class ResNetBlock(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, x):
#         residual = x
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out += residual
#         return self.relu(out)


# class RefinementTResNet(nn.Module):

#     def __init__(
#         self,
#         feature_dim,
#         hidden_dim=256,
#         num_hidden_layers=3,
#         dropout=0.1,
#     ):
#         super().__init__()

#         layers = []

#         input_dim = feature_dim + 1

#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))

#         for _ in range(num_hidden_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))


#         layers.append(nn.Linear(hidden_dim, 1))

#         self.network = nn.Sequential(*layers)

#     def forward(self, X):

#         return self.network(X)