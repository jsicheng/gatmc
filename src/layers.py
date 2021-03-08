import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from utils import stack, split_stack
from torch_geometric.nn import GCNConv, GATConv

USER, ITEM = 'user', 'item'
GCN, GAT = 'gcn', 'gat'

# First Layer of the Encoder (implemented by Pytorch Geometric)
# Please the following repository for details.
# https://github.com/rusty1s/pytorch_geometric
class RGCLayer(MessagePassing):
    def __init__(self, config, weight_init):
        super(RGCLayer, self).__init__()
        self.in_c = config.num_nodes
        self.out_c = config.hidden_size[0]
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.num_item = config.num_nodes - config.num_users
        self.drop_prob = config.drop_prob
        self.weight_init = weight_init
        self.accum = config.accum
        self.bn = config.rgc_bn
        self.relu = config.rgc_relu

        # if config.accum == 'split_stack':
        #     # each 100 dimention has each realtion node features
        #     # user-item-weight-sharing
        #     self.base_weight = nn.Parameter(torch.Tensor(
        #         max(self.num_users, self.num_item), self.out_c))
        #     self.dropout = nn.Dropout(self.drop_prob)
        # else:
        #     # ordinal basis matrices in_c * out_c = 2625 * 500
        #     ord_basis = [nn.Parameter(torch.Tensor(1, self.in_c * self.out_c)) for r in range(self.num_relations)]

        ord_basis = [nn.Parameter(torch.Tensor(self.in_c, self.out_c)) for r in range(self.num_relations)]
        self.ord_basis = nn.ParameterList(ord_basis)

        self.relu = nn.ReLU()

        if config.accum == 'stack':
            self.bn = nn.BatchNorm1d(self.in_c * config.num_relations)
        else:
            self.bn = nn.BatchNorm1d(self.in_c)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        # if self.accum == 'split_stack':
        #     weight_init(self.base_weight, self.in_c, self.out_c)
        # else:
        #     for basis in self.ord_basis:
        #         weight_init(basis, self.in_c, self.out_c)
        for basis in self.ord_basis:
            weight_init(basis, self.in_c, self.out_c)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        # return self.propagate(self.accum, edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)
        return self.our_propagate(x, edge_index, edge_type, edge_norm)

    def our_propagate(self, x, edge_index, edge_type, edge_norm):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mu_j = torch.zeros(self.num_relations, self.in_c, self.out_c).to(device)
        for r in range(self.num_relations):
            assert type(r) is int
            mu_j[r:] += self.ord_basis[r]
        mu_j = self.our_node_dropout(mu_j)

        h = []
        for r in range(self.num_relations):
            edge_index_r_indices = (edge_type == r).nonzero().view(-1)
            edge_index_r = edge_index[:,edge_index_r_indices]
            edge_norm_r = edge_norm[edge_index_r_indices]
            mu_jr = mu_j[r]
            h.append(
                scatter_add(
                    edge_norm_r.view(-1,1)*mu_jr[edge_index_r[0]],
                    edge_index_r[1],
                    dim=0,
                    dim_size=self.in_c
                )
            )
        h = torch.sum(torch.stack(tuple(h),dim=0),dim=0)

        # post process sigmoid stuff
        if self.bn:
            h = self.bn(h.unsqueeze(0)).squeeze(0)
        if self.relu:
            h = self.relu(h)

        return h

    def our_node_dropout(self, weight):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        drop_mask = drop_mask.view(1,-1,1)
        drop_mask = drop_mask.to(device)
        weight = weight * drop_mask

        return weight

# First Layer of the Encoder
class GNNLayer(nn.Module):
    def __init__(self, config, weight_init, gnn_type, n_layers, num_users):
        super(GNNLayer, self).__init__()
        self.num_users = num_users
        self.n_layers = n_layers
        GNN_u, GNN_v, GNN_uv = [], [], []
        dims_default = [64, 32, 16, 8]  # TODO: make this proper
        for i in range(n_layers):
            # create in_dim and out_dim
            if i == 0:
                in_dim = 1
                out_dim = dims_default[i]
            elif i == n_layers - 1:
                in_dim = dims_default[i - 1]
                out_dim = config.hidden_size[0]
            else:
                in_dim = dims_default[i - 1]
                out_dim = dims_default[i]

            # form layers
            if gnn_type == GCN:
                GNN_u.append(GCNConv(in_dim, out_dim))
                GNN_v.append(GCNConv(in_dim, out_dim))
                GNN_uv.append(GCNConv(out_dim, out_dim))
            elif gnn_type == GAT:
                GNN_u.append(GATConv(in_dim, out_dim))
                GNN_v.append(GATConv(in_dim, out_dim))
                GNN_uv.append(GATConv(out_dim, out_dim))
            else:
                assert False

        self.GNN_u = nn.ModuleList(GNN_u)
        self.GNN_v = nn.ModuleList(GNN_v)
        self.GNN_uv = nn.ModuleList(GNN_uv)

    def forward(self, x, edge_index, edge_index_u, edge_index_v):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_index_u = edge_index_u.to(device)
        edge_index_v = edge_index_v.to(device)
        for i in range(self.n_layers):
            x_u = self.GNN_u[i](x, edge_index_u)[:self.num_users]
            x_v = self.GNN_v[i](x, edge_index_v)[self.num_users:]
            x = torch.cat((x_u, x_v), dim=0)
            x = self.GNN_uv[i](x, edge_index)
            # TODO: possibly add a u super node and v super node
            #  if edge_index_u and edge_index_v are highly disconnected
            if i != self.n_layers - 1:
                x = F.relu(F.dropout(x, training=self.training))
        return x

# Second Layer of the Encoder
class DenseLayer(nn.Module):
    def __init__(self, config, weight_init, bias=False):
        super(DenseLayer, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        self.bn = config.dense_bn
        self.relu = config.dense_relu
        self.weight_init = weight_init

        self.dropout = nn.Dropout(config.drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)
        if config.accum == 'stack':
            self.bn_u = nn.BatchNorm1d(config.num_users * config.num_relations)
            self.bn_i = nn.BatchNorm1d((config.num_nodes - config.num_users) * config.num_relations)
        else:
            self.bn_u = nn.BatchNorm1d(config.num_users)
            self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users)
        self.relu = nn.ReLU()

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)

        return u_features, i_features
