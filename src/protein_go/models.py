import torch
import torch.nn as nn
from src.protein_go.data import load_feature_matrix, load_labels, load_splits

class MLP(nn.Module):
    """Original four-layer MLP architecture."""

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_mlp_model(input_dim, num_classes):
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    loaded_state_dict = torch.load("data/processed/models/best_homology_MLP_pes_seed42.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict)
    model.eval()
    return model





if __name__ == "__main__":
    X = load_feature_matrix("pes")
    Y = load_labels()
    train_idx, val_idx, test_idx = load_splits("pes")
    model = load_mlp_model(input_dim=X.shape[1], num_classes=Y.shape[1])
    batch = torch.tensor(X[:8], dtype=torch.float32)
    with torch.no_grad():
        output = model(batch)
    print("Output shape:", output.shape)
