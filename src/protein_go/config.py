from pathlib import Path
import yaml

def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    cfg = load_config()
    print("config loaded sucessfully")
    print("Keys:", list(cfg.keys()))
