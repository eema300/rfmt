import os
import hydra
from omegaconf import OmegaConf
import torch
from funcmol.models.nffm import sample_normal, sample_time
from funcmol.utils.utils_fm import compute_code_stats_offline, compute_ot_fast
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import normalize_code
from funcmol.dataset.dataset_code import create_code_loaders


@hydra.main(config_path="configs", config_name="train_nffm_qm9", version_base=None)
def main(config):
    fabric = setup_fabric(config)
    config = OmegaConf.to_container(config)

    out_dir = config.get("out_dir", os.path.join(config["dirname"], "t_data"))
    os.makedirs(out_dir, exist_ok=True)
    n_iterations = config.get("n_iterations", 10)

    loader_train = create_code_loaders(config, split="train", fabric=fabric)
    code_stats = compute_code_stats_offline(loader_train, "train", fabric, config["normalize_codes"])
    fabric.print(f">> code_stats: {code_stats}")

    latent_dim = config["decoder"]["code_dim"]
    use_ot = config.get("nffm", {}).get("use_ot", False)
    fabric.print(f">> use_ot: {use_ot}")

    for it in range(n_iterations):
        aug_idx = torch.randint(0, loader_train.dataset.num_augmentations, [1])[0].item()
        loader_train.dataset.load_codes(aug_idx)

        xt_shard, t_shard = [], []

        for batch in loader_train:
            codes = normalize_code(batch, code_stats)
            x1 = codes
            x0 = sample_normal(x1.shape[0], latent_dim).to(fabric.device)

            if use_ot:
                perm = compute_ot_fast(x0, x1)
                x0 = x0[perm]

            t = sample_time(x1.shape[0]).to(fabric.device)
            x_t = (1.0 - t) * x0 + t * x1

            xt_shard.append(x_t.detach().cpu())
            t_shard.append(t.detach().cpu())

        xt_shard = torch.cat(xt_shard, dim=0)
        t_shard = torch.cat(t_shard, dim=0)

        out_path = os.path.join(out_dir, f"t_data_{it:04d}.pt")
        torch.save({"xt": xt_shard, "t": t_shard}, out_path)
        fabric.print(f">> saved {xt_shard.shape[0]} (xt, t) pairs to {out_path}")


if __name__ == "__main__":
    main()