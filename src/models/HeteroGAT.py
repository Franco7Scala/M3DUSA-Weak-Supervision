import torch
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear  # <-- use Linear from PyG


class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, target_type, hidden_channels=128, out_channels=2, dropout=0, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.target_type = target_type

        # Heterogeneous GAT layers
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        # First layer
        self.convs.append(
            HeteroConv(
                {
                    edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
                    for edge_type in metadata[1]  # metadata[1] = list of edge types
                },
                aggr="sum"
            )
        )
        self.lins.append(
            torch.nn.ModuleDict(
                {ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}  # now safe, PyG Linear supports -1
            )
        )

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
                        for edge_type in metadata[1]
                    },
                    aggr="sum"
                )
            )
            self.lins.append(
                torch.nn.ModuleDict(
                    {ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}
                )
            )

        # Last embedding layer
        self.convs.append(
            HeteroConv(
                {
                    edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
                    for edge_type in metadata[1]
                },
                aggr="sum"
            )
        )
        self.lins.append(
            torch.nn.ModuleDict(
                {ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}
            )
        )

        # Final classification layer
        self.final_conv = HeteroConv(
            {
                edge_type: GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout)
                for edge_type in metadata[1]
            },
            aggr="sum"
        )
        self.final_lin = torch.nn.ModuleDict(
            {ntype: Linear(-1, out_channels) for ntype in metadata[0]}
        )

    def forward(self, x_dict, edge_index_dict, embeddings_only=False):
        # Hidden layers
        for i in range(self.num_layers - 1):
            x_dict = {
                ntype: self.convs[i](x_dict, edge_index_dict)[ntype] + self.lins[i][ntype](x.relu())
                for ntype, x in x_dict.items()
            }
            x_dict = {ntype: x.relu() for ntype, x in x_dict.items()}

        # Embeddings before final layer
        embeddings = {
            ntype: self.convs[-1](x_dict, edge_index_dict)[ntype] + self.lins[-1][ntype](x.relu())
            for ntype, x in x_dict.items()
        }

        # Final classification outputs
        out_dict = {
            ntype: self.final_conv(embeddings, edge_index_dict)[ntype] + self.final_lin[ntype](embeddings[ntype].relu())
            for ntype in embeddings.keys()
        }

        out_dict["target_type"] = self.target_type

        if embeddings_only:
            return embeddings

        return out_dict #, embeddings
