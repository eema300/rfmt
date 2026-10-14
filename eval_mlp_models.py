import hydra
import torch
import omegaconf
from funcmol.utils.utils_base import setup_fabric
from torch.utils.data import DataLoader
from utils_collect import collect_data, import_nffm, load_mlp_model
from tz_dataset import TZ_Dataset


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    config = omegaconf.OmegaConf.to_container(config, resolve=True)

    fabric = setup_fabric(config)

    # load nffm model from checkpoint
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

    # load models from checkpoints
    rfmt_t_interpolate = load_mlp_model(config, fabric, width, lr, "interp_pt_van_path")
    rfmt_t_sample      = load_mlp_model(config, fabric, width, lr, "sample_pt_van_path")

    # evaluate models on the validation set
    val_loss_interpolate = rfmt_t_interpolate.evaluate(val_mlp_loader, collection_type="interpolate")
    print(f"interpolate model test loss: {val_loss_interpolate}")

    val_loss_sample = rfmt_t_sample.evaluate(val_mlp_loader, collection_type="sample")
    print(f"sample model test loss: {val_loss_sample}")


if __name__ == "__main__":
    main()