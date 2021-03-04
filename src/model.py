import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from layers import RGCLayer, DenseLayer
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils.convert import to_networkx


USER, ITEM = 'user', 'item'
GCN, GAT = 'gcn', 'gat'


# Main Model
class GAE(nn.Module):
    def __init__(self, config, weight_init):
        super(GAE, self).__init__()
        self.gcenc = OurGCEncoder(config, weight_init)
        self.bidec = BiDecoder(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm, data):
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm, data)
        adj_matrices = self.bidec(u_features, i_features)

        return adj_matrices


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


class OurGCEncoder(nn.Module):
    def __init__(self, config, weight_init):
        super(OurGCEncoder, self).__init__()
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.accum = config.accum

        # self.rgc_layer = RGCLayer(config, weight_init)
        self.gnn = GNNLayer(config, weight_init, GCN, 2, self.num_users)
        self.dense_layer = DenseLayer(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm, data):
        g = to_networkx(data)
        if self.accum == 'stack':
            # TODO: if stack also refers to input x:
            # num_nodes = int(x.shape[0] / self.num_relations)
            # assert num_nodes == g.number_of_nodes()
            # TODO: else:
            assert x.shape[0] == g.number_of_nodes()

        if len(x.shape) < 2:
            x = torch.unsqueeze(x, dim=1).type(torch.FloatTensor)

        edge_index_u, edge_index_v = self.separate_edges(g)

        features = self.gnn(x, edge_index, edge_index_u, edge_index_v)
        # features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)

        return u_features, i_features

    def separate_edges(self, g):
        # create map linking users to items and vice versa in bipartite graph
        u2i, i2u = {}, {}
        for nid in g.nodes():
            if self._is_user_or_item(nid, g) == USER:
                u2i[nid] = list(nx.neighbors(g, nid))
            elif self._is_user_or_item(nid, g) == ITEM:
                i2u[nid] = list(nx.neighbors(g, nid))
            else:
                assert False
        # create u2u edges and i2i edges
        user_edges = torch.LongTensor(self._multiply_edges(u2i, i2u))
        item_edges = torch.LongTensor(self._multiply_edges(i2u, u2i))

        return user_edges, item_edges

    def _multiply_edges(self, a2b, b2a):
        a_edges_u, a_edges_v = [], []
        for a_u, b_li in a2b.items():
            for b in b_li:
                a_v_li = b2a[b]
                a_edges_u.extend([a_u] * len(a_v_li))
                a_edges_v.extend(a_v_li)
        return [a_edges_u, a_edges_v]

    def _is_user_or_item(self, nid, g):
        # TODO: if stack also refers to input x:
        # if self.accum == 'stack':
        #     nid_norm = nid % g.number_of_nodes()
        # else:
        #     nid_norm = nid
        # TODO: else:
        nid_norm = nid

        if nid_norm < self.num_users:
            return USER
        else:
            return ITEM

    def separate_features(self, features):
        if self.accum == 'stack':
            num_nodes = int(features.shape[0] / self.num_relations)
            for r in range(self.num_relations):
                if r == 0:
                    u_features = features[:self.num_users]
                    i_features = features[self.num_users: (r + 1) * num_nodes]
                else:
                    u_features = torch.cat((u_features,
                                            features[r * num_nodes: r * num_nodes + self.num_users]), dim=0)
                    i_features = torch.cat((i_features,
                                            features[r * num_nodes + self.num_users: (r + 1) * num_nodes]), dim=0)

        else:
            u_features = features[:self.num_users]
            i_features = features[self.num_users:]

        return u_features, i_features


# Encoder (will be separated to two layers(RGC and Dense))
class GCEncoder(nn.Module):
    def __init__(self, config, weight_init):
        super(GCEncoder, self).__init__()
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.accum = config.accum

        self.rgc_layer = RGCLayer(config, weight_init)
        self.dense_layer = DenseLayer(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm, data):
        features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)

        return u_features, i_features

    def separate_features(self, features):
        if self.accum == 'stack':
            num_nodes = int(features.shape[0] / self.num_relations)
            for r in range(self.num_relations):
                if r == 0:
                    u_features = features[:self.num_users]
                    i_features = features[self.num_users: (r + 1) * num_nodes]
                else:
                    u_features = torch.cat((u_features,
                                            features[r * num_nodes: r * num_nodes + self.num_users]), dim=0)
                    i_features = torch.cat((i_features,
                                            features[r * num_nodes + self.num_users: (r + 1) * num_nodes]), dim=0)

        else:
            u_features = features[:self.num_users]
            i_features = features[self.num_users:]

        return u_features, i_features


# Decoder
class BiDecoder(nn.Module):
    def __init__(self, config, weight_init):
        super(BiDecoder, self).__init__()
        self.num_basis = config.num_basis
        self.num_relations = config.num_relations
        self.feature_dim = config.hidden_size[1]
        self.accum = config.accum
        self.apply_drop = config.bidec_drop

        self.dropout = nn.Dropout(config.drop_prob)
        self.basis_matrix = nn.Parameter(
            torch.Tensor(config.num_basis, self.feature_dim * self.feature_dim))
        coefs = [nn.Parameter(torch.Tensor(config.num_basis))
                 for b in range(config.num_relations)]
        self.coefs = nn.ParameterList(coefs)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        # weight_init(self.basis_matrix, self.feature_dim, self.feature_dim)
        nn.init.orthogonal_(self.basis_matrix)
        for coef in self.coefs:
            weight_init(coef, self.num_basis, self.num_relations)

    def forward(self, u_features, i_features):
        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)
        if self.accum == 'stack':
            u_features = u_features.reshape(self.num_relations, -1, self.feature_dim)
            i_features = i_features.reshape(self.num_relations, -1, self.feature_dim)
            num_users = u_features.shape[1]
            num_items = i_features.shape[1]
        else:
            num_users = u_features.shape[0]
            num_items = i_features.shape[0]

        for relation in range(self.num_relations):
            q_matrix = torch.sum(self.coefs[relation].unsqueeze(1) * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.feature_dim, self.feature_dim)
            if self.accum == 'stack':
                if relation == 0:
                    out = torch.chain_matmul(
                        u_features[relation], q_matrix,
                        i_features[relation].t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features[relation], q_matrix,
                        i_features[relation].t()).unsqueeze(-1)), dim=2)
            else:
                if relation == 0:
                    out = torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(-1)), dim=2)

        out = out.reshape(num_users * num_items, -1)

        return out

