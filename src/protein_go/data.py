from pathlib import Path
import numpy as np
from src.protein_go.config import load_config

def load_feature_matrix(feature_set: str="pes"):
    config = load_config()
    data_dir = Path(config['paths']['data_processed'])
    load_file = np.load(data_dir / f"X_{feature_set}_homology.npy")
    return load_file

def load_labels():
    config = load_config()
    data_dir = Path(config['paths']['data_processed'])
    load_file = np.load(data_dir / "Y_all.npy")
    return load_file

def load_splits(feature_set: str="pes"):
    config = load_config()
    data_dir = Path(config['paths']['data_processed'])
    load_file = np.load(data_dir / "splits_homology.npz")
    return load_file['train_idx'], load_file['val_idx'], load_file['test_idx']



if __name__ =="__main__":
    X = load_feature_matrix()
    Y = load_labels()
    train_idx, val_idx, test_idx = load_splits()
    print("Feature matrix shape:", X.shape)
    print("Labels shape:", Y.shape)
    print("Train indices shape:", train_idx.shape)
    print("Validation indices shape:", val_idx.shape)
    print("Test indices shape:", test_idx.shape)