import hydra
import torch
import os
import omegaconf
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_fm import load_checkpoint_fm
from funcmol.models.nffm import create_nffm, sample_normal
from learn_t import RefinementT
from data_collect import assert_dir


@hydra.main(config_path="configs", config_name="sample_nffm", version_base=None)
def main(config):
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    assert_dir(config, "out_dir")
    assert_dir(config, "mlp_out_dir")

    fabric = setup_fabric(config)

    checkpoint_nffm = fabric.load(os.path.join(config["nffm_pretrained_path"], "checkpoint.pth.tar"))
    config_ckpt = checkpoint_nffm["config"]
    for key in config.keys():
        if key in config_ckpt and isinstance(config_ckpt[key], omegaconf.dictconfig.DictConfig):
            config_ckpt[key].update(config[key])
        else:
            config_ckpt[key] = config[key]
    config = config_ckpt
    fabric.print(f"updated config: {config}")

    # load checkpoint
    # with torch.no_grad():
    nffm = create_nffm(config, fabric)
    nffm, code_stats, _ = load_checkpoint_fm(nffm, config["nffm_pretrained_path"], fabric=fabric)
    nffm = fabric.setup_module(nffm)

    # set up for sampling
    num_samples = config["num_samples"]
    num_steps = config["num_steps"]
    eps = 1e-4 # from train_nffm.py

    width = config["decoder"]["code_dim"]

    x0 = sample_normal(num_samples, width)

    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    t0 = 0.0
    t1 = 1.0
    dt = (t1 - t0) / float(num_steps)

    x = x0.to(nffm.device)

    rfmt_t = RefinementT(feature_dim=config["decoder"]["code_dim"], 
                         fabric=fabric, config=config)
    opt = torch.optim.Adam(rfmt_t.parameters(), lr=1e-4)
    rfmt_t, opt = fabric.setup(rfmt_t, opt)
    
    best_loss = float("inf")

    # sampling
    # training dataset size = num_samples × num_steps
    for i in range(num_steps):
        t = t0 + i * dt
        t_batch = torch.full((x.shape[0], 1), t, device=nffm.device)

        # predict x_1
        x1_hat = nffm(x, t_batch)

        # calc velocity field based on x_1
        v = (x1_hat - x) / max(1.0 - float(t), eps)

        # update using velocity
        x = x + dt * v

        # use (x, t) as training batch & print mse
        loss = rfmt_t.run_batch(X=x, t=t_batch, opt=opt)

        # save the checkpt
        if loss < best_loss:
            rfmt_t.save_checkpoint(loss=loss, model_type="sample")