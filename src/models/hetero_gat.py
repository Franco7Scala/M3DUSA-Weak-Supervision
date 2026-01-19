import torch
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear


class HeteroGAT(torch.nn.Module):

    def __init__(self, metadata, target_type, hidden_channels=128, out_channels=2, dropout=0, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.target_type = target_type
        # Heterogeneous GAT layers
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        # First layer
        self.convs.append(HeteroConv({edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum"))
        self.lins.append(torch.nn.ModuleDict({ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}))
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(HeteroConv({edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum"))
            self.lins.append(torch.nn.ModuleDict({ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}))

        # Last embedding layer
        self.convs.append(HeteroConv({edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum"))
        self.lins.append(torch.nn.ModuleDict({ntype: Linear(-1, hidden_channels) for ntype in metadata[0]}))
        # Final classification and regression layers
        self.final_conv_ic_classification = HeteroConv({edge_type: GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum")
        self.final_ic_classification = torch.nn.ModuleDict({ntype: Linear(-1, out_channels) for ntype in metadata[0]})
        self.final_conv_ic_regression = HeteroConv({edge_type: GATv2Conv((-1, -1), 1, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum")
        self.final_ic_regression = torch.nn.ModuleDict({ntype: Linear(-1, 1) for ntype in metadata[0]})
        self.final_conv_proxy_regression = HeteroConv({edge_type: GATv2Conv((-1, -1), 1, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum")
        self.final_proxy_regression = torch.nn.ModuleDict({ntype: Linear(-1, 1) for ntype in metadata[0]})

    def forward(self, x_dict, edge_index_dict, embeddings_only=False):
        # Hidden layers
        for i in range(self.num_layers - 1):
            x_dict = {ntype: self.convs[i](x_dict, edge_index_dict)[ntype] + self.lins[i][ntype](x.relu()) for ntype, x in x_dict.items()}
            x_dict = {ntype: x.relu() for ntype, x in x_dict.items()}

        # Embeddings before final layer
        embeddings = {ntype: self.convs[-1](x_dict, edge_index_dict)[ntype] + self.lins[-1][ntype](x.relu()) for ntype, x in x_dict.items()}

        # Final classification outputs
        out_ic_classification = {ntype: self.final_conv_ic_classification(embeddings, edge_index_dict)[ntype] + self.final_ic_classification[ntype](embeddings[ntype].relu()) for ntype in embeddings.keys()}
        out_ic_regression = {ntype: self.final_conv_ic_regression(embeddings, edge_index_dict)[ntype] + self.final_ic_regression[ntype](embeddings[ntype].relu()) for ntype in embeddings.keys()}
        out_proxy_regression = {ntype: self.final_conv_proxy_regression(embeddings, edge_index_dict)[ntype] + self.final_proxy_regression[ntype](embeddings[ntype].relu()) for ntype in embeddings.keys()}

        out_ic_classification["target_type"] = self.target_type
        out_ic_regression["target_type"] = self.target_type
        out_proxy_regression["target_type"] = self.target_type

        if embeddings_only:
            return embeddings

        return out_ic_classification, out_ic_regression, out_proxy_regression
