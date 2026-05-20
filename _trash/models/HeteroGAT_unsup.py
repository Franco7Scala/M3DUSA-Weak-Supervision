import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=128, out_channels=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            conv_dict = {
                edge_type: GATv2Conv(
                    (-1, -1), hidden_channels,
                    add_self_loops=False,
                    dropout=dropout
                )
                for edge_type in metadata[1]
                if "metapath" not in edge_type[1].lower()
            }

            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

        # Final MLP layers for user output (for contrastive loss/clustering)
        self.user_lin = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Return transformed user embeddings and all other node embeddings
        z_user = self.user_lin(x_dict['user'])  # Only for users
        return z_user, x_dict  # x_dict can be used for other tasks or debug
