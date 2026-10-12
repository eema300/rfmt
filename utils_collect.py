import os
import omegaconf
import torch
import sys
from funcmol.models.nffm import create_nffm, sample_normal
from funcmol.utils.utils_fm import load_checkpoint_fm
sys.path.append("/net/dali/home/mscbio/emg386/rfmt")
from tz_dataset import TZ_Dataset
from learn_t import load_checkpoint


def assert_dir(config, key):
    dir_name = config[key]
    assert "${" not in dir_name, f"{dir_name} was not resolved: {dir_name}"
    assert os.path.isabs(dir_name), f"{dir_name} is not an absolute path: {dir_name}"
    os.makedirs(dir_name, exist_ok=True)

    # this is just so i don't waste my time
    test_file = os.path.join(dir_name, ".write_test")
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)


def import_nffm(config, fabric):
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
    nffm, _, _ = load_checkpoint_fm(nffm, config["nffm_pretrained_path"], fabric=fabric)
    nffm = fabric.setup_module(nffm)
    nffm.eval()

    return nffm


def load_mlp_model(config, fabric, width, lr, checkpoint_path):
    rfmt_t = load_checkpoint(chkpt_path=config[checkpoint_path], 
                             feat_dim=width, fabric=fabric, config=config)
    
    opt = torch.optim.Adam(rfmt_t.parameters(), lr=lr)
    rfmt_t, _ = fabric.setup(rfmt_t, opt)

    return rfmt_t


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

            # save x_t, t in the dataset
            dataset.add_data(x_t=x, t=t)

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

        # save the last x_t, t pair
        dataset.add_data(x_t=x, t=t)