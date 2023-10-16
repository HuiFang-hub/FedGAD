# -*- coding: utf-8 -*-
# @Time    : 23/03/2023 20:04
# @Function:
# federatedscope/contrib/data/my_cora.py
import dgl
import torch
from torch_geometric.data import Data
import copy
import numpy as np
from src.pygod.generator import gen_contextual_outliers, gen_structural_outliers
from src.federatedscope.core.auxiliaries.utils import setup_seed
import src.federatedscope.register as register
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from src.federatedscope.core.splitters.graph import LouvainSplitter
from src.federatedscope.register import register_data
from src.dgld.utils.dataset import GraphNodeAnomalyDectionDataset
from src.dgld.utils.load_data import load_data, load_custom_data, load_truth_data
from src.dgld.utils.load_data import load_data, load_custom_data, load_truth_data
from src.dgld.utils.inject_anomalies import inject_contextual_anomalies, inject_structural_anomalies
from src.dgld.utils.common_params import Q_MAP, K, P
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# def my_cora(config=None):
#     path = config.data.root
#
#     num_split = [232, 542, np.iinfo(np.int64).max]
#     dataset = Planetoid(path,
#                         'cora',
#                         split='random',
#                         num_train_per_class=num_split[0],
#                         num_val=num_split[1],
#                         num_test=num_split[2])
#     global_data = copy.deepcopy(dataset)[0]
#     dataset = LouvainSplitter(config.federate.client_num)(dataset[0])
#
#     data_local_dict = dict()
#     for client_idx in range(len(dataset)):
#         data_local_dict[client_idx + 1] = dataset[client_idx]
#
#     data_local_dict[0] = global_data
#     return data_local_dict, config


# def call_my_data(config):
#     if config.data.type == "mycora":
#         data, modified_config = my_cora(config)
#         return data, modified_config

# def data_split_client(config,args_dict,seed,client_cfgs=None):
#     setup_seed(12345)
#     for func in register.data_dict.values():
#         data_and_config = func(config, client_cfgs)
#         if data_and_config is not None:
#             return data_and_config
#
#
#
#     gnd_dataset = None
#     truth_list = ['weibo', 'tfinance', 'tsocial', 'reddit', 'Amazon', 'Class', 'Disney', 'elliptic', 'Enron']
#     data_name = args_dict['dataset']
#     path = config.data_path
#     # if config.data.type.lower() in [
#     #     'cora',
#     #     'citeseer',
#     #     'pubmed',
#     #     'dblp_conf',
#     #     'dblp_org',
#     # ] or config.data.type.lower().startswith('csbm'):
#     #     gnd_dataset = GraphNodeAnomalyDectionDataset("Cora", p=15, k=50)
#     if config.data.type.lower()  in truth_list:
#         graph = load_truth_data(data_path=path, dataset_name=data_name)
#     elif data_name == 'custom':
#         graph = load_custom_data(data_path=path)
#     else:
#         graph = load_data(data_name)
#         graph = inject_contextual_anomalies(graph=graph, k=K, p=P, q=Q_MAP[data_name], seed=seed)
#         graph = inject_structural_anomalies(graph=graph, p=P, q=Q_MAP[data_name], seed=seed)
#     dataset = [ds for ds in graph]
#     client_num = min(len(dataset), config.federate.client_num
#                      ) if config.federate.client_num > 0 else len(dataset)
#     config.merge_from_list(['federate.client_num', client_num])
#     # label = graph.ndata['label']
#     # data = translator(graph)
#     #
#     # # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
#     # data = convert_data_mode(data, modified_config)
#     #
#     # # Restore the user-specified seed after the data generation
#     # setup_seed(config.seed)
#     #
#     #
#     # dataset = gnd_dataset[0]
#     #
#     #
#     #
#     # global_data = copy.deepcopy(dataset)
#     # dataset = LouvainSplitter(config.federate.client_num)(dataset)
#     # data_local_dict = dict()
#     # for client_idx in range(len(dataset)):
#     #     data_local_dict[client_idx + 1] = dataset[client_idx]
#     # data_local_dict[0] = global_data
#     # setup_seed(config.seed)
#     # return data_local_dict, config

def data_split_client(config):
    setup_seed(12345)
    # for func in register.data_dict.values():
    #     data_and_config = func(config, client_cfgs)
    #     if data_and_config is not None:
    #         return data_and_config
    data = None
    if config.data.type == "cora":
        data = Planetoid('./data/Cora', 'Cora', transform=T.NormalizeFeatures())[0]
    data, ya = gen_contextual_outliers(data, n=100, k=50)
    data, ys = gen_structural_outliers(data, m=10, n=10)
    data.y = torch.logical_or(ys, ya).int()

    global_data = copy.deepcopy(data)
    dataset = LouvainSplitter(config.federate.client_num)(data)

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config

def dgl_to_pyg(g):
    src, dst = g.edges()
    return  Data(
        x=g.ndata['feat'],
        edge_index=torch.stack([src, dst], dim=0),
        ay = g.ndata['label'],
        edge_attr=None,
        y = None,
        pos=None
    )
    # x = g.ndata['feat']
    # # edge_attr = torch.randn(g.number_of_edges(), 5)
    # data = Data(x=x, edge_index=g.edges())
    # data.ay = g.ndata['label']
    # return pyg_data



if __name__ == '__main__':
    print("test")
    my_cora()
    # register_data("mycora", call_my_data)