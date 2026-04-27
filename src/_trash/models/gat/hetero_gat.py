import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear


class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, target_type, hidden_channels=128, out_channels=2, dropout=0, num_layers=2):
        super().__init__()
        self.target_type = target_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({edge_type: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout) for edge_type in metadata[1]}, aggr="sum")
            self.convs.append(conv)
            lin = torch.nn.ModuleDict({ ntype: Linear(-1, hidden_channels) for ntype in metadata[0]})
            self.lins.append(lin)

        self.head_class = nn.Sequential(
            Linear(-1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(32, out_channels)
        )
        self.head_ic_reg = nn.Sequential(
            Linear(-1, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Linear(32, 1),
            nn.Sigmoid()
        )
        self.head_proxy_reg = nn.Sequential(
            Linear(-1, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Linear(32, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        for i in range(self.num_layers):
            conv_out = self.convs[i](x_dict, edge_index_dict)
            new_x_dict = {}
            for ntype, x in x_dict.items():
                res = self.lins[i][ntype](x)
                if ntype in conv_out:
                    out = conv_out[ntype] + res

                else:
                    out = res

                out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
                new_x_dict[ntype] = out

            x_dict = new_x_dict

        target_emb = x_dict[self.target_type]
        out_class = self.head_class(target_emb)
        out_ic = self.head_ic_reg(target_emb)
        out_proxy = self.head_proxy_reg(target_emb)
        return ({self.target_type: out_class}, {self.target_type: out_ic}, {self.target_type: out_proxy})
