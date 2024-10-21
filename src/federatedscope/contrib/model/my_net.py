import torch
import torch.nn.functional as F
# from src.dgld.models.CoLA import CoLA
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from src.federatedscope.register import register_model
# from src.dgld.models.CoLA.model import OneLayerGCNWithGlobalAdg,Discriminator
from pygod.models import CoLA
import numpy as np
import scipy.sparse as sp
import random
import os
from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk
from pygod.models.basic_nn import Vanilla_GCN as GCN

from src.pygod.models.anemone import ANEMONE_Base
from src.pygod.models.cola import CoLA_Base


class MyGCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MyGCN, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x
def gcnbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MyGCN(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model

class MyCola(torch.nn.Module):
    def __init__(self,config,x_dim,device,
                 lr=1e-3,
                 epoch=10,
                 negsamp_ratio=1,
                 readout='avg',
                 weight_decay=0.,
                 batch_size=0,
                 subgraph_size=4,
                 contamination=0.1,
                 gpu=0,
                 verbose=False):
        super(MyCola, self).__init__()
        self.feat_dim = x_dim
        self.lr = lr
        self.num_epoch = epoch
        self.embedding_dim = config.hidden
        self.negsamp_ratio = negsamp_ratio
        self.readout = readout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.verbose = verbose
        self.device = device
        self.model = CoLA_Base(self.feat_dim,
                               self.embedding_dim,
                               'prelu',
                               self.negsamp_ratio,
                               self.readout,
                               self.subgraph_size,device
                               )

    def forward(self, x, adj, all_idx, subgraphs, cur_batch_size):
        output = self.model(x, adj, all_idx, subgraphs, cur_batch_size)
        return output

    #
    #
    #
    # def process_graph(self, G):
    #     """
    #     Description
    #     -----------
    #     Process the raw PyG data object into a tuple of sub data
    #     objects needed for the model.
    #
    #     Parameters
    #     ----------
    #     G : PyTorch Geometric Data instance (torch_geometric.data.Data)
    #         The input data.
    #
    #     Returns
    #     -------
    #     x : torch.Tensor
    #         Attribute (feature) of nodes.
    #     adj : torch.Tensor
    #         Adjacency matrix of the graph.
    #     """
    #     self.num_nodes = G.x.shape[0]
    #     self.feat_dim = G.x.shape[1]
    #     adj = to_dense_adj(G.edge_index)[0]
    #     adj = sp.coo_matrix(adj.cpu().numpy())#
    #     rowsum = np.array(adj.sum(1))
    #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #     adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    #     adj = (adj + sp.eye(adj.shape[0])).todense()
    #
    #     x = G.x[np.newaxis]
    #     adj = torch.FloatTensor(adj[np.newaxis]).to(self.device)
    #     # return data objects needed for the network
    #     return x, adj
    #
    # def generate_rw_subgraph(self,pyg_graph, nb_nodes, subgraph_size):
    #     """Generate subgraph with random walk algorithm."""
    #     row, col = pyg_graph.edge_index
    #     all_idx = torch.tensor(list(range(nb_nodes)))
    #     traces = random_walk(row, col, all_idx, walk_length=3)
    #     subv = traces.tolist()
    #     return subv

def colabuilder(config,input_shape,device):
    num_nodes,x_dim = input_shape
    model = MyCola(config,x_dim,device)
    return model

def anemonebuilder(config,input_shape,device):
    num_nodes, x_dim = input_shape
    model = MyAnemone(config, x_dim, device)
    return model

class MyAnemone(torch.nn.Module):
    def __init__(self,config,x_dim,device,
                 lr=None,
                 verbose=False,
                 epoch=None,
                 embedding_dim=64,
                 negsamp_ratio=1,
                 readout='avg',
                 dataset='cora',
                 weight_decay=0.0,
                 batch_size=300,
                 subgraph_size=4,
                 alpha=1.0,
                 negsamp_ratio_patch=1,
                 negsamp_ratio_context=1,
                 auc_test_rounds=256,
                 contamination=0.1,
                 gpu=0):
        super(MyAnemone, self).__init__()
        self.contamination= contamination
        self.dataset = dataset
        self.feat_dim = x_dim
        self.embedding_dim = config.hidden
        self.negsamp_ratio = negsamp_ratio
        self.readout = readout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.auc_test_rounds = auc_test_rounds
        self.alpha = alpha
        self.negsamp_ratio_patch = config.negsamp_ratio_patch
        self.negsamp_ratio_context = config.negsamp_ratio_context
        self.device = device
        # if gpu >= 0 and torch.cuda.is_available():
        #     self.device = 'cuda:{}'.format(gpu)
        # else:
        #     self.device = 'cpu'

        if lr is None:
            if self.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr']:
                self.lr = 1e-3
            elif self.dataset == 'ACM':
                self.lr = 5e-4
            elif self.dataset == 'BlogCatalog':
                self.lr = 3e-3
            else:
                self.lr = 1e-3

        if epoch is None:
            if self.dataset in ['cora', 'citeseer', 'pubmed']:
                self.num_epoch = 100
            elif self.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
                self.num_epoch = 400
            else:
                self.num_epoch = 100

        self.verbose = verbose
        self.model = ANEMONE_Base(self.feat_dim,
                                  self.embedding_dim,
                                  'prelu',
                                  self.negsamp_ratio_patch,
                                  self.negsamp_ratio_context,
                                  self.readout,device)

    def forward(self, bf,ba):
        logits_1, logits_2 = self.model(bf, ba)
        return logits_1, logits_2
def call_my_net(model_config,  input_shape,device):
    # Please name your gnn model with prefix 'gnn_'
    model = None
    if model_config.type.lower() == "mygcn":
        model = gcnbuilder(model_config,   input_shape)
    elif model_config.type.lower() == "cola":
        model =  colabuilder(model_config,input_shape,device)
        # model = CoLA()
    elif model_config.type.lower() == "anemone":
        model = anemonebuilder(model_config, input_shape, device)
    return model



register_model("cola", call_my_net)
register_model("anemone", call_my_net)