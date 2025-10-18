import os
import wget
import yaml

def ensure_sam2_weights(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = f"checkpoints/{os.path.basename(cfg['checkpoint_url'])}"
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(ckpt_path):
        print(f"Downloading SAM2 weights to {ckpt_path} ...")
        wget.download(cfg["checkpoint_url"], ckpt_path)
    return ckpt_path, cfg["config_path"], cfg["device"]

