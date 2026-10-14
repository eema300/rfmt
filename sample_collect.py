import hydra
import torch
import os
import omegaconf
from funcmol.utils.utils_base import setup_fabric
from torch.utils.data import DataLoader
from learn_t import RefinementT
from tz_dataset import TZ_Dataset, NFFMStream
from plot_mse import plot_mse_with_best_dict
from utils_collect import assert_dir, import_nffm, collect_data


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    assert_dir(config, "plots")
    assert_dir(config, "t_preds_path")
    assert_dir(config, "losses_path")
    assert_dir(config, "mlp_pretrained_path")

    fabric = setup_fabric(config)

    # load the model from checkpoint
    nffm = import_nffm(config, fabric)

    # set up vars for sampling (generating data)
    num_epochs_val = config["n_epochs_val"]
    num_samples = config["num_samples"]
    batch_size = config["mlp_batch_size"]
    num_steps = config["num_steps"]
    width = config["decoder"]["code_dim"]
    lr = 1e-4

    # generate the val set, size is num_samples x num_steps x num_epochs_val
    val_mlp_dset = TZ_Dataset(torch.empty(0, width), torch.empty(0, 1))
    collect_data(val_mlp_dset, nffm, width, num_epochs_val, num_steps, num_samples, lr)
    val_mlp_loader = DataLoader(val_mlp_dset, batch_size, shuffle=True)

    # infinite training data
    train_mlp_loader = DataLoader(NFFMStream(nffm, width, batch_size, num_samples=batch_size), batch_size=None, num_workers=0)

    # initialize model and optimizer
    rfmt_t = RefinementT(feature_dim=width, 
                         fabric=fabric, config=config)
    opt = torch.optim.Adam(rfmt_t.parameters(), lr=lr)
    rfmt_t, opt = fabric.setup(rfmt_t, opt)
    
    # train mlp model to predict t given x_t
    num_epochs_train = config["n_epochs"]
    steps_per_epoch = config["n_steps_per_epoch"]
    rfmt_t = rfmt_t.fit(train_mlp_loader, val_mlp_loader, epochs=num_epochs_train, steps_per_epoch=steps_per_epoch,
                        opt=opt, collection_type="sample")
    
    # plot training mse
    save_plot = os.path.join(config["plots_path"], "sample_train_mse_graph_vanilla.png")
    plot_mse_with_best_dict(mse_dict=rfmt_t.plot_train_mse, save_path=save_plot)
    
    # plot val mse
    save_plot = os.path.join(config["plots_path"], "sample_val_mse_graph_vanilla.png")
    plot_mse_with_best_dict(mse_dict=rfmt_t.plot_val_mse, save_path=save_plot)


if __name__ == "__main__":
    main()