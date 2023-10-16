# -*- coding: utf-8 -*-
# @Time    : 30/03/2023 21:55
# @Function:
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
import copy
import numpy as np
import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from util.data_process import dgl_to_pyg
from src.pygod.generator import gen_contextual_outliers, gen_structural_outliers
from src.federatedscope.core.auxiliaries.utils import setup_seed
import src.federatedscope.register as register
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from src.federatedscope.core.splitters.graph import LouvainSplitter
from src.federatedscope.register import register_data
import torch
import os
import sys
# from src.dgld.utils.load_data import load_data, load_custom_data, load_truth_data
from src.dgld.utils.inject_anomalies import inject_contextual_anomalies, inject_structural_anomalies
from src.dgld.utils.common_params import Q_MAP, K, P
import random
import community as community_louvain
import networkx as nx
import torch
from collections import Counter
from torch_geometric.datasets.attributed_graph_dataset import AttributedGraphDataset
from torch_geometric.datasets import Reddit2,Flickr,FacebookPagePage,Yelp,PolBlogs,Amazon,Twitch
import logging
from src.dgld.utils.load_data import load_data,load_custom_data, load_truth_data
from util.vision import plot_violinplot

logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def my_cora(config=None):
    path = config.data.root

    num_split = [232, 542, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        'cora',
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = copy.deepcopy(dataset)[0]
    dataset = LouvainSplitter(config.federate.client_num)(dataset[0])

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config

def splite_by_client_pyg(data,config,data_name):
    global_data = copy.deepcopy(data)
    # split setting
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}
    if config.data.splitter == 'louvain':
        from src.federatedscope.core.splitters.graph import LouvainSplitter_gad_pyg
        splitter = LouvainSplitter_gad_pyg(client_num,delta=20)
    dataset = splitter(data, data_name= data_name,seed =config.seed )
    # store in cilent dict
    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])
    data_dict = dict()
    anomy_label= []
    for client_idx in range(1, len(dataset) + 1):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]
        data_dict[client_idx] = {'data': local_data}
        anomy_label.append(local_data.ay)
    anomy_label = torch.cat(anomy_label)
    global_data.ay =  anomy_label
    data_dict[0] = { 'data': global_data}
    return  data_dict,config

def splite_by_client_pyg_before(data,config,data_name):
    global_data = copy.deepcopy(data)
    # split setting
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}
    if config.data.splitter == 'louvain':
        from src.federatedscope.core.splitters.graph import LouvainSplitter
        splitter = LouvainSplitter(client_num+1,delta=config.data.splitter_delta)
        dataset = splitter(data, data_name=data_name, seed=config.seed)
    elif config.data.splitter == 'ScaffoldLdaSplitter': # protein
        from src.federatedscope.core.splitters.graph import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, alpha=config.data.splitter_args[0]['alpha'])
        dataset = splitter(data)
    elif config.data.splitter == 'RelTypeSplitter':
        from src.federatedscope.core.splitters.graph import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, alpha=config.data.splitter_args[0]['alpha'])
        dataset = splitter(data)
    elif config.data.splitter == 'RandomSplitter':
        from src.federatedscope.core.splitters.graph import RandomSplitter
        splitter = RandomSplitter(client_num)
        dataset = splitter(data)
    elif config.data.splitter == 'random_partition':
        from src.federatedscope.core.splitters.graph import random_partition
        dataset,violin_df = random_partition(data,client_num+1)

    # plot
    # plot_violinplot(violin_df)



    # store in cilent dict
    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])
    data_dict = dict()
    num_nodes = []
    num_edges = []
    test_data = dataset[len(dataset)-1]
    test_data.edge_index = add_self_loops(
        to_undirected(remove_self_loops(test_data.edge_index)[0]),
        num_nodes=test_data.x.shape[0])[0]
    for client_idx in range(1, len(dataset)):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]

        data_dict[client_idx] = {'data': local_data,'test_data':test_data}
        num_nodes.append(local_data.num_nodes)
        num_edges.append(local_data.num_edges)

        # anomy_label.append(local_data.ay)
    # anomy_label = torch.cat(anomy_label)
    # global_data.ay =  anomy_label
    global_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(global_data.edge_index)[0]),
            num_nodes=global_data.x.shape[0])[0]
    data_dict[0] = { 'data': global_data,'test_data':test_data}
    return  data_dict,config,num_nodes,num_edges

def call_my_data(config,client_cfgs = None):
    path = config.data.root
    data_name = config.data.type.lower()
    seed = random.randint(1, 100)
    graph = None
    modified_config = None,None
    if  data_name== "mycora":
        data, modified_config = my_cora(config)
    elif data_name == "cora-inj-before": # yes
        data_name = 'Cora'
        graph = Planetoid('./data/cora-inj-before', 'cora', transform=T.NormalizeFeatures())[0]
        # data, ya = gen_contextual_outliers(data, n=100, k=50)
        # data, ys = gen_structural_outliers(data, m=10, n=10)
        # data.y = torch.logical_or(ys, ya).int()
    elif data_name == "citeseer-inj-before": #yes
        if config.dataloader.type == 'pyg' or config.dataloader.type == 'base':
            data_name1 = 'Citeseer'
            graph = Planetoid(f'./data/{data_name}', data_name1, transform=T.NormalizeFeatures())[0]
    elif data_name == "pubmed-inj-before": #?
        data_name1 = 'Pubmed'
        graph = Planetoid(f'./data/{data_name}', data_name1, transform=T.NormalizeFeatures())[0]
        config.data.splitter_delta = 50
    # elif data_name == "reddit2-inj-before": #1G
    #     if config.dataloader.type == 'pyg' or config.dataloader.type == 'base':
    #         data_name1 = 'Reddit2'
    #         graph = Reddit2(f'./data/{data_name}', transform=T.NormalizeFeatures())[0]
    # elif data_name == "flickr-inj-before": # data splite time
    #     if config.dataloader.type == 'pyg' or config.dataloader.type == 'base':
    #         data_name1 = 'flickr'
    #         graph = Flickr(f'./data/{data_name}',transform=T.NormalizeFeatures())[0]
    #         config.data.splitter_delta = 30
    elif data_name == "amazon-c-inj-before":  #? auc
        data_name1 = 'computers'
        graph = Amazon(f'./data/{data_name}',data_name1,transform=T.NormalizeFeatures())[0]

    elif data_name == "amazon-p-inj-before":  # ? auc
        data_name1 = 'photo'
        graph = Amazon(f'./data/{data_name}', data_name1, transform=T.NormalizeFeatures())[0]
        config.data.splitter_delta = 60
    # elif data_name == "yelp-inj-before":  #1.6G
    #     if config.dataloader.type == 'pyg' or config.dataloader.type == 'base':
    #         data_name1 = 'Yelp'
    #         graph =Yelp(f'./data/{data_name}',transform=T.NormalizeFeatures())[0]
    elif data_name == "facebookpagepage-inj-before":  # yes
        if config.dataloader.type == 'pyg' or config.dataloader.type == 'base':
            data_name1 = 'FacebookPagePage'
            graph =FacebookPagePage(f'./data/{data_name}',transform=T.NormalizeFeatures())[0]
    elif data_name == "twitch-pt-inj-before": # yes
        data_name1 ='PT'
        graph = Twitch(f'./data/{data_name}', data_name1,transform=T.NormalizeFeatures())[0]
        config.data.splitter_delta = 50
    elif data_name == "twitch-de-inj-before": # yes
        data_name1 ='DE'
        graph = Twitch(f'./data/{data_name}', data_name1,transform=T.NormalizeFeatures())[0]
        config.data.splitter_delta = 100
    # test = graph.x
    n_outliers = int(graph.num_nodes * config.model.contamination)
    if config.data.anomaly_type == 'c':
        n_outlier_s = 10
        n_outlier_c = n_outliers - n_outlier_s
    elif config.data.anomaly_type == 's':
        n_outlier_c = 10
        n_outlier_s = n_outliers - n_outlier_c
    else:
        n_outlier_c = int(n_outliers*0.5)
        n_outlier_s = n_outliers - n_outlier_c
    graph, ya, outlier_idx_c = gen_contextual_outliers(graph, n=n_outlier_c, k=50)  # contextual outliers is n
    graph, ys, outlier_idx_s = gen_structural_outliers(graph, m=10, n= n_outlier_s//10,#n_outlier_s // 10,
                                                       outlierIdx=outlier_idx_c)  # structural outliers is m×n
    # graph.ay = torch.logical_or(ys, ya).int()
    graph.ay = torch.logical_or(ys, ya).int()
    logger.info(f'anomaly proportion:{Counter(graph.ay.tolist())}')
    logger.info(f'y proportion:{Counter(graph.y.tolist())}')
    config.model.out_channels = len(torch.unique(graph.y))
    sum_nodes = graph.num_nodes
    sum_edges = graph.num_edges
    data, modified_config, num_nodes, num_edges = splite_by_client_pyg_before(graph, config, data_name)
    logger.info(f'num_nodes:{num_nodes} sum_nodes:{sum_nodes} average_num_nodes:{np.mean(num_nodes)}')
    logger.info(f'num_edges:{num_edges} sum_edges:{sum_edges} average_num_edges:{np.mean(num_edges)}')


    return data, modified_config


def call_truth_data(config,client_cfgs = None):
    truth_list = [ 'tfinance', 'tsocial','yelp','amazon']
    data_name = config.data.type.lower()
    if data_name in truth_list:
        if data_name in [ 'tfinance', 'tsocial']:
            graph = load_truth_data(data_path=f'./data/{data_name}', dataset_name=data_name)
            # transform into pyg style
            data = dgl_to_pyg(graph)
        elif data_name== 'yelp':
            dataset = FraudYelpDataset(raw_dir=f'./data/{data_name}')
            # graph = dataset[0]
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph.ndata['feat'] = graph.ndata['feature']
            graph = dgl.add_self_loop(graph)
            data = dgl_to_pyg(graph)
        elif data_name== 'amazon':
            dataset = FraudAmazonDataset()
            graph = dgl.to_homogeneous(dataset[0],
                                       ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph.ndata['feat'] = graph.ndata['feature']
            graph = dgl.add_self_loop(graph)
            data = dgl_to_pyg(graph)

    if data_name == "tfinance":
        config.model.contamination = 0.05
    elif data_name == "tsocial":
        config.model.contamination = 0.03
    elif data_name == "yelp":
        config.model.contamination = 0.15
    elif data_name == "amazon":
        config.model.contamination = 0.07

    # add label
    from src.federatedscope.core.splitters.graph import my_clustering
    client_num = config.federate.client_num
    cluster = my_clustering(data, client_num + 1)
    predict_labels = cluster.run()
    data.y = predict_labels
    config.model.out_channels = len(torch.unique(data.y))
    logger.info(f'anomaly proportion:{Counter(data.ay.tolist())}')
    sum_nodes = data.num_nodes
    sum_edges =  data.num_edges
    data, modified_config, num_nodes, num_edges = splite_by_client_pyg_before(data, config, data_name)
    logger.info(f'num_nodes:{num_nodes} sum_nodes:{sum_nodes} average_num_nodes:{np.mean(num_nodes)}')
    logger.info(f'num_edges:{num_edges} sum_edges:{sum_edges} average_num_edges:{np.mean(num_edges)}')
    return data,config



register_data("cora-inj", call_my_data)
register_data("cora-inj-before", call_my_data)
register_data("citeseer-inj-before", call_my_data)
register_data("pubmed-inj-before", call_my_data)
# register_data("flickr-inj-before", call_my_data)
# register_data("reddit2-inj-before", call_my_data)
# register_data("yelp-inj-before", call_my_data)
# register_data("polblogs-inj-before", call_my_data)
register_data("facebookpagepage-inj-before", call_my_data)
register_data("twitch-pt-inj-before", call_my_data)
register_data("twitch-de-inj-before", call_my_data)
register_data("amazon-c-inj-before", call_my_data)
register_data("amazon-p-inj-before", call_my_data)
register_data("tfinance", call_truth_data)
register_data("tsocial", call_truth_data) # 跑不动
register_data("yelp", call_truth_data)
register_data("amazon", call_truth_data)