import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from layers import RGCLayer, GNNLayer, DenseLayer
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.dropout import dropout_adj
from collections import defaultdict
USER, ITEM = 'user', 'item'
GCN, GAT = 'gcn', 'gat'

class Timer():
    def __init__(self):
        self.last_t = time.time()
    def get_time(self):
        delta = time.time()-self.last_t
        self.last_t = time.time()
        return delta

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

class OurGCEncoder(nn.Module):
    def __init__(self, config, weight_init):
        super(OurGCEncoder, self).__init__()
        in_dim = 64
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.accum = config.accum
        self.drop_prob = config.drop_prob
        self.encoder_user = nn.Linear(3, in_dim)
        self.encoder_item = nn.Linear(3, in_dim)

        # self.rgc_layer = RGCLayer(config, weight_init)
        self.gnn = GNNLayer(config, weight_init, config.model, config.use_uv, self.num_users, self.num_relations)
        self.dense_layer = DenseLayer(config, weight_init)
        self.edge_obj_cache = {}

    def forward(self, x, edge_index, edge_type, edge_norm, data):
        # g = to_networkx(data)
        # if self.accum == 'stack':
        #     # TODO: if stack also refers to input x:
        #     # num_nodes = int(x.shape[0] / self.num_relations)
        #     # assert num_nodes == g.number_of_nodes()
        #     # TODO: else:
        #     assert x.shape[0] == g.number_of_nodes()
        # timer = Timer()
        # print('start', timer.get_time())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cache_key = tuple(list(edge_index.shape))
        if hash(cache_key) not in self.edge_obj_cache:
            edge_index, edge_type = dropout_adj(edge_index, edge_type.view(-1, 1), self.drop_prob)
            edge_index, edge_type = edge_index.to(device), edge_type.view(-1).to(device)
            self.init_g(x.shape[0], edge_index, edge_type)
            relation2edge_dict = self.separate_edges(edge_index, edge_type) # TODO: separate edges correctly
            avg_ratings, num_ratings, ones_vec = self.get_node_metadata()
            self.edge_obj_cache[hash(cache_key)] = \
                edge_index, edge_type, relation2edge_dict, \
                avg_ratings, num_ratings, ones_vec
        else:
            edge_index, edge_type, relation2edge_dict, avg_ratings, num_ratings, ones_vec = \
                self.edge_obj_cache[hash(cache_key)]
        # print('ei', timer.get_time())
        x = self.get_x_init(avg_ratings, num_ratings, ones_vec)
        # print('x_init', timer.get_time())

        features = self.gnn(x, edge_index, relation2edge_dict) # TODO: set up dimensions properly
        # print('gnn', timer.get_time())
        # features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)
        # print('sep+dense', timer.get_time())

        return u_features, i_features

    def init_g(self, num_nodes, edge_index, edge_type):
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        for r in range(self.num_relations):
            edge_index_r_indices = (edge_type == r).nonzero().view(-1)
            edge_index_r = edge_index[:,edge_index_r_indices]
            edge_list_r = edge_index_r.t().tolist()
            g.add_edges_from(edge_list_r, r=r) # TODO: ensure that the nodes added preserve order i.e. g.nodes[nid]['nid'] == nid \forall nid
        self.g = g

    def get_x_init(self, avg_ratings, num_ratings, ones_vec):
        init_features = torch.stack((avg_ratings, num_ratings, ones_vec), dim=-1) # (N+M) x 3
        x_users = self.encoder_user(init_features[:self.num_users])# N x 3 * 3 x D = N x D
        x_items = self.encoder_item(init_features[self.num_users:]) # M x 3 * 3 x D = M x D
        x_init = torch.cat((x_users, x_items), dim=0) # (N+M) x D
        # x_init = self.our_node_dropout(x_init)
        # print(x_init.shape, x_init)
        return x_init

    # def our_node_dropout(self, x):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     drop_mask = torch.rand(x.size(0), device=device) + (1 - self.drop_prob)
    #     drop_mask = torch.floor(drop_mask).type(torch.float)
    #     drop_mask = drop_mask.view(-1,1)
    #     x = x * drop_mask
    #     return x

    def get_node_metadata(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        avg_ratings = torch.zeros(len(self.g.nodes), dtype=torch.float32, device=device)
        num_ratings = torch.zeros(len(self.g.nodes), dtype=torch.float32, device=device)
        ones_vec = torch.ones(len(self.g.nodes), dtype=torch.float32, device=device)
        for (u,v) in self.g.edges():
            rating = self.g.edges[(u,v)]['r']
            avg_ratings[u] += rating
            avg_ratings[v] += rating
            num_ratings[u] += 1
            num_ratings[v] += 1
        avg_ratings[num_ratings > 0.1] = avg_ratings[num_ratings > 0.1] / num_ratings[num_ratings > 0.1]
        return avg_ratings, num_ratings, ones_vec

    def separate_edges(self, edge_index, edge_type):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create map linking users to items and vice versa in bipartite graph
        relation2link_dict = \
            {
                r:{'u2i':defaultdict(list), 'i2u':defaultdict(list)}
                for r in range(self.num_relations)
            }
        for (u,v) in self.g.edges():
            r = self.g.edges[(u,v)]['r']
            if self._is_user_or_item(u) == USER:
                assert self._is_user_or_item(v) == ITEM
                relation2link_dict[r]['u2i'][u].append(v)
                relation2link_dict[r]['i2u'][v].append(u)
            elif self._is_user_or_item(u) == ITEM:
                assert self._is_user_or_item(v) == USER
                relation2link_dict[r]['u2i'][v].append(u)
                relation2link_dict[r]['i2u'][u].append(v)
            else:
                assert False

        # create u2u edges and i2i edges
        relation2edges = \
            {
                r: {'user': None, 'item': None, 'user-item': None}
                for r in range(self.num_relations)
            }
        for r in range(self.num_relations):
            u2i = relation2link_dict[r]['u2i']
            i2u = relation2link_dict[r]['i2u']
            relation2edges[r]['user'] = \
                torch.LongTensor(
                    self._multiply_edges(u2i, i2u)
                ).to(device)
            relation2edges[r]['item'] = \
                torch.LongTensor(
                    self._multiply_edges(i2u, u2i)
                ).to(device)

        # create u2i/i2u edges:
        for r in range(self.num_relations):
            edge_index_r_indices = (edge_type == r).nonzero().view(-1)
            edge_index_r = edge_index[:,edge_index_r_indices]
            relation2edges[r]['user-item'] = edge_index_r
        return relation2edges

    def _multiply_edges(self, a2b, b2a):
        a_edges_u, a_edges_v = [], []
        for a_u, b_li in a2b.items():
            for b in b_li:
                a_v_li = b2a[b]
                a_edges_u.extend([a_u] * len(a_v_li))
                a_edges_v.extend(a_v_li)
        return [a_edges_u, a_edges_v]

    def _is_user_or_item(self, nid):
        if nid < self.num_users:
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
                    u_features = torch.cat((u_features, features[r * num_nodes: r * num_nodes + self.num_users]), dim=0)
                    i_features = torch.cat((i_features, features[r * num_nodes + self.num_users: (r + 1) * num_nodes]), dim=0)

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
                    u_features = torch.cat((u_features, features[r * num_nodes: r * num_nodes + self.num_users]), dim=0)
                    i_features = torch.cat((i_features, features[r * num_nodes + self.num_users: (r + 1) * num_nodes]), dim=0)

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

