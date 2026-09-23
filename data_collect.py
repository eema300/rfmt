'''
this will build the (z_t, t) dataset by interpolating many times
'''

import hydra
import torch
import os
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_fm import compute_codes
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.utils.utils_nf import infer_codes_occs_batch
from funcmol.dataset.field_maker import FieldMaker
from funcmol.models.nffm import sample_normal, sample_time


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    with torch.no_grad():
        # load the neural field (encoder, decoder)
        print(">> loading nf checkpoint")
        nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
        config_nf = nf_checkpoint["config"]
        config_nf["dset"] = config["dset"]
        config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]

        enc, dec = load_neural_field(nf_checkpoint, fabric, config=config_nf)
        dec_module = dec.module if hasattr(dec, "module") else dec

    # create the molecular occupancy fields
    print(">> creating molecular occupancy fields")
    field_maker = FieldMaker(config, sample_points=False) # should be false since i am not retraining neural fields
    field_maker = field_maker.to(fabric.device)

    # data loaders for neural field
    print(">> creating neural field data loader")
    loader_train = create_field_loaders(config, fabric=fabric) # train by default and this is good to start with since there are 10,000 obs in there

    # encode the molecular fields into their latent codes & get the stats
    print(">> encoding latent codes")
    _, code_stats = compute_codes(
        loader_train, enc, config_nf, "train", fabric, config["normalize_codes"],
        field_maker=field_maker, code_stats=None
    )
    dec_module.set_code_stats(code_stats)

    print(">> generating data")
    latent_dim = config["decoder"]["code_dim"]
    with torch.no_grad():
        for epoch in range(config["n_epochs"]):
            zt_shard, t_shard = [], []

            for batch in loader_train:
                # encode to latent codes for this batch
                codes, _ = infer_codes_occs_batch(
                    batch, enc, config, to_cpu=False, field_maker=field_maker,
                    code_stats=dec_module.code_stats if config["normalize_codes"] else None
                )

                z1 = codes
                z0 = sample_normal(z1.shape[0], latent_dim).to(fabric.device)
                t = sample_time(z1.shape[0]).to(fabric.device)

                # interpolate
                z_t = (1.0 - t) * z0 + t * z1

                # save the current z_t, t pair
                zt_shard.append(z_t.detach().cpu())
                t_shard.append(t.detach().cpu())

            zt_shard = torch.cat(zt_shard, dim=0)
            t_shard = torch.cat(t_shard, dim=0)

            out_path = os.path.join(config["out_dir"], f"t_data_{epoch:04d}.pt")
            torch.save({"z_t": zt_shard, "t": t_shard}, out_path)
            print(f">> saved {zt_shard.shape[0]} pairs to {out_path}")


if __name__ == "__main__":
    main()