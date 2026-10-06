import hydra
import torch
import os
import sys
import omegaconf
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_fm import load_checkpoint_fm
from funcmol.models.nffm import create_nffm, sample_normal
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append("/net/dali/home/mscbio/emg386/rfmt")
from data_collect import assert_dir
from sample_collect import collect_data
from tz_dataset import TZ_Dataset
from learn_t import RefinementT, load_checkpoint


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
        
    assert_dir(config, "out_dir")
    assert_dir(config, "mlp_out_dir")

    fabric = setup_fabric(config)

    # collect the validation data by sampling the nffm model
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
    num_epochs_val = config["n_epochs_val"]
    num_samples = config["num_samples"]
    num_steps = config["num_steps"]
    width = config["decoder"]["code_dim"]
    lr = 1e-4

    # initialize empty val dataset
    val_mlp_dset = TZ_Dataset(torch.empty(0, width), torch.empty(0, 1))

    # collect validation dataset
    collect_data(val_mlp_dset, nffm, width, num_epochs_val, num_steps, num_samples, lr)

    # data loader
    batch_size = config["dset"]["batch_size"]
    code_dim = config["decoder"]["code_dim"]
    val_mlp_loader = DataLoader(val_mlp_dset, batch_size, shuffle=True)

    # load models from checkpoints
    rfmt_t_interpolate = load_checkpoint(chkpt_path=config["interpolate_pt_path"], feat_dim=code_dim,
                                         fabric=fabric, config=config)
    rfmt_t_sample      = load_checkpoint(chkpt_path=config["sample_pt_path"], feat_dim=code_dim,
                                         fabric=fabric, config=config)
    
    opt_interpolate = torch.optim.Adam(rfmt_t_interpolate.parameters(), lr=lr)
    rfmt_t_interpolate, opt_interpolate = fabric.setup(rfmt_t_interpolate, opt_interpolate)
    
    opt_sample = torch.optim.Adam(rfmt_t_sample.parameters(), lr=lr)
    rfmt_t_sample, opt_sample = fabric.setup(rfmt_t_sample, opt_sample)

    # evaluate models on the validation set
    val_loss_interpolate = rfmt_t_interpolate.evaluate(val_mlp_loader, collection_type="interpolate")
    print(f"interpolate model validation loss: {val_loss_interpolate}")

    val_loss_sample = rfmt_t_sample.evaluate(val_mlp_loader, collection_type="sample")
    print(f"sample model validation loss: {val_loss_sample}")


if __name__ == "__main__":
    main()