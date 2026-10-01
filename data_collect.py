'''
this will build the (z_t, t) dataset by interpolating num_epoch times per molecule
'''

import hydra
import torch
import os
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import load_neural_field, normalize_code
from funcmol.utils.utils_fm import compute_codes, compute_code_stats_offline
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.field_maker import FieldMaker
from funcmol.models.nffm import sample_normal, sample_time
from omegaconf import OmegaConf
from learn_t import RefinementT

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


@hydra.main(config_path="configs", config_name="collect_codes_t", version_base=None)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    assert_dir(config, "out_dir")
    assert_dir(config, "codes_dir")
    assert_dir(config, "mle_out_dir")

    fabric = setup_fabric(config)

    # need to encode the molecules
    if config["on_the_fly"]:
        print(">> encoding molecules")

        with torch.no_grad():
            # load the neural field (encoder, decoder)
            print(">> loading nf checkpoint")
            nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
            config_nf = nf_checkpoint["config"]
            config_nf["dset"] = config["dset"]
            config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]

            enc, _ = load_neural_field(nf_checkpoint, fabric, config=config_nf)
            # dec_module = dec.module if hasattr(dec, "module") else dec

            print(">> creating molecular occupancy fields")
            field_maker = FieldMaker(config, sample_points=False) # should be false since i am not retraining neural fields
            field_maker = field_maker.to(fabric.device)

            # data loaders for neural field
            print(">> creating neural field data loader")
            loader_field = create_field_loaders(config, fabric=fabric) # train by default and this is good to start with since there are 10,000 obs in there

            # encode the molecular fields into their latent codes & get the stats
            print(">> encoding latent codes")
            codes_raw, _ = compute_codes(
                loader_field, enc, config_nf, "train", fabric, config["normalize_codes"],
                field_maker=field_maker, code_stats=None
            )
            # dec_module.set_code_stats(code_stats)
            codes_raw = codes_raw.detach().cpu()

            code_path = os.path.join(config["codes_dir"], "codes_0000.pt")
            torch.save(codes_raw, code_path) # TODO so this is a tensor and not a dict since the CodeDataset class expects this
            print(f">> wrote {codes_raw.shape[0]} raw codes to {code_path}")

    else:
        print(">> using existing codes")
        existing = [f for f in os.listdir(os.path.join(config["codes_dir"], "train")) if f.startswith("codes") and f.endswith(".pt")]
        assert existing, f"on_the_fly=False but no codes found in {config['codes_dir']}"

    # get the computed codes
    loader_train = create_code_loaders(config, split="train", fabric=fabric)
    code_stats = compute_code_stats_offline(loader_train, "train", fabric, config["normalize_codes"])
    print(f">> code_stats: {code_stats}")

    latent_dim = config["decoder"]["code_dim"]

    rfmt_t = RefinementT(feature_dim=config["decoder"]["code_dim"], fabric=fabric)
    best_loss = float("inf")

    # training dataset size = n_molecules × n_epochs
    print(">> generating data and training mlp")
    for epoch in range(config["n_epochs"]):

        total_loss, n_examples, = 0.0, 0

        for batch in loader_train:
            # get actual z1 code, sample z0, sample a t
            z1 = normalize_code(batch, code_stats)
            z0 = sample_normal(z1.shape[0], latent_dim).to(fabric.device)
            t = sample_time(z1.shape[0]).to(fabric.device)

            # interpolate
            z_t = (1.0 - t) * z0 + t * z1

            # use (x, t) as training batch
            batch_loss = rfmt_t.run_batch(X=z_t, t=t, epoch=epoch)

            # loss in batch so far
            total_loss += batch_loss * z_t.shape[0] # normalizes the loss per batch
            n_examples += z_t.shape[0]

        # get the loss for this epoch across the batches
        epoch_loss = total_loss / n_examples
        print(f" * epoch {epoch} >> avg loss: {epoch_loss:.6f}") # if time wandb

        # save model checkpt
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            rfmt_t.save_checkpoint(loss=epoch_loss, model_type="interpolate")


if __name__ == "__main__":
    main()

