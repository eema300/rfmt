import hydra
import torch
import os
import sys
import omegaconf
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_fm import load_checkpoint_fm
from funcmol.models.nffm import create_nffm, sample_normal
from torch.utils.data import DataLoader
sys.path.append("/net/dali/home/mscbio/emg386/rfmt")
from learn_t import RefinementT
from tz_dataset import TZ_Dataset
from data_collect import assert_dir
from plot_mse import plot_mse_with_best_dict

# creates a dataset of length = num_samples x num_steps x num_epochs
def collect_data(dataset: TZ_Dataset, nffm_model, width: int, num_epochs: int, 
                 num_steps: int, num_samples: int, lr: float) -> None:

    for _ in range(num_epochs):

        # sample Gaussian dist num_samples times -> x_0 vector
        x0 = sample_normal(num_samples, width)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)

        # calculate step size
        t0 = 0.0
        t1 = 1.0
        dt = (t1 - t0) / float(num_steps)

        # x_t vector starts at x_0
        x = x0.to(nffm_model.device)

        # sampling
        # # training dataset size = num_samples × num_steps
        for i in range(num_steps):
            # new_best_flag = False

            # take a time step for every x in x_t
            t = t0 + i * dt
            t_batch = torch.full((x.shape[0], 1), t, device=nffm_model.device)

            with torch.no_grad(): # need this here or else x links back through foward computation and prev iteration x's
                # predict x_1
                x1_hat = nffm_model(x, t_batch)
                # calc velocity field based on x_1
                v = (x1_hat - x) / max(1.0 - float(t), lr)
                # update x_t based on velocity
                x = x + dt * v
                
                # save x_t, t in the dataset
                dataset.add_data(x_t=x, t=t_batch)


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    assert_dir(config, "out_dir")
    assert_dir(config, "mlp_out_dir")

    fabric = setup_fabric(config)

    # load the nffm checkpoint & update the model config to meet collect_codes_t config
    checkpoint_nffm = fabric.load(os.path.join(config["nffm_pretrained_path"], "checkpoint.pth.tar"))
    config_ckpt = checkpoint_nffm["config"]
    for key in config.keys():
        if key in config_ckpt and isinstance(config_ckpt[key], omegaconf.dictconfig.DictConfig):
            config_ckpt[key].update(config[key])
        else:
            config_ckpt[key] = config[key]
    config = config_ckpt
    fabric.print(f"updated config: {config}")

    # load nffm model from checkpoint
    nffm = create_nffm(config, fabric)
    nffm, code_stats, _ = load_checkpoint_fm(nffm, config["nffm_pretrained_path"], fabric=fabric)
    nffm = fabric.setup_module(nffm)

    # set up vars for sampling (generating data)
    num_epochs_train = config["n_epochs_train"]
    num_epochs_val = config["n_epochs_val"]
    num_epochs_mlp = config["n_epochs"]
    num_samples = config["num_samples"]
    num_steps = config["num_steps"]
    width = config["decoder"]["code_dim"]
    lr = 1e-4

    # initialize empty datasets
    train_mlp_dset = TZ_Dataset(torch.empty(0, width), torch.empty(0, 1))
    val_mlp_dset = TZ_Dataset(torch.empty(0, width), torch.empty(0, 1))

    # collect data for each dataset
    collect_data(train_mlp_dset, nffm, width, num_epochs_train, num_steps, num_samples, lr)
    collect_data(val_mlp_dset, nffm, width, num_epochs_val, num_steps, num_samples, lr)

    # data loaders
    batch_size = config["dset"]["batch_size"]
    train_mlp_loader = DataLoader(train_mlp_dset, batch_size, shuffle=True)
    val_mlp_loader   = DataLoader(val_mlp_dset,   batch_size, shuffle=True)

    # initialize model and optimizer
    rfmt_t = RefinementT(feature_dim=width, 
                         fabric=fabric, config=config)
    opt = torch.optim.Adam(rfmt_t.parameters(), lr=lr)
    rfmt_t, opt = fabric.setup(rfmt_t, opt)
    
    # train mlp model to predict t given x_t
    rfmt_t = rfmt_t.fit(train_data_loader=train_mlp_loader, epochs=num_epochs_mlp, opt=opt)

    # evaluate fit mlp model
    val_loss = rfmt_t.evaluate(val_mlp_loader, collection_type="sample")
    print(f"validation loss: {val_loss}")

    # plot training mse
    plot_mse_with_best_dict(mse_dict=rfmt_t.plot_mse, save_path=config["plots_path"])


if __name__ == "__main__":
    main()