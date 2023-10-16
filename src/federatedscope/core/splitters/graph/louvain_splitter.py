from __future__ import absolute_import
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import from_networkx
import dgl
import networkx as nx
import community as community_louvain
import numpy as np
from src.federatedscope.core.splitters import BaseSplitter
from src.dgld.utils.inject_anomalies import inject_contextual_anomalies, inject_structural_anomalies
from src.dgld.utils.common_params import Q_MAP, K, P
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
# from torch_geometric.utils import to_networkx_data
class LouvainSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via louvain algorithm.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
    """
    def __init__(self, client_num, delta=20):
        self.delta = delta
        BaseSplitter.__init__(self, client_num)

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        # node_attrs = data.keys
        data.train_mask, data.val_mask, data.test_mask = split_by_ratio(data.num_nodes)
        has_y_attribute = 'y' in data
        node_attrs = ['x', 'y', 'ay', 'train_mask', 'val_mask', 'test_mask'] if has_y_attribute else ['x', 'ay', 'train_mask', 'val_mask', 'test_mask']
        G = to_networkx(
            data,
            node_attrs=node_attrs,
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)

        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)

        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num

        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            cnt=0
            while len(node_list) + len(client_node_idx[idx]) > max_len_client + self.delta:
                test = len(node_list) + len(client_node_idx[idx])
                t = max_len_client
                cnt+=1
                idx = (idx + 1) % self.client_num
                if cnt%10 ==0:
                    self.delta += cnt
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        return graphs


class LouvainSplitter_gad_pyg(BaseTransform, BaseSplitter):
    """
    Split Data into small data via louvain algorithm.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
    """

    def __init__(self, client_num, delta=20):
        self.delta = delta
        BaseSplitter.__init__(self, client_num)

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        train_mask, val_mask, test_mask = split_by_ratio(data.num_data)

        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)

        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)

        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num

        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(
                    client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num

        graphs = []
        # ay = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            subg = from_networkx(nx.subgraph(G, nodes))
            subg, yc = gen_contextual_outliers(subg, n=int(100/self.client_num), k=int(50/self.client_num))
            subg, ys = gen_structural_outliers(subg, m=int(50/self.client_num), n=int(10/self.client_num))
            subg.ay = torch.logical_or(ys, yc).int()
            graphs.append(subg)
            # ay.append(subg.ay)
        return graphs



# class LouvainSplitter_gad_dgl(BaseTransform, BaseSplitter):
#     """
#     Split Data into small data via louvain algorithm.
#
#     Args:
#         client_num (int): Split data into ``client_num`` of pieces.
#         delta (int): The gap between the number of nodes on each client.
#     """
#     def __init__(self, client_num, delta=20):
#         self.delta = delta
#         BaseSplitter.__init__(self, client_num)
#
#     def __call__(self, data, **kwargs):
#         data.index_orig = torch.arange(data.num_nodes())
#         G = dgl.to_networkx(
#             data,
#             node_attrs=['feat', 'label', 'train_mask', 'val_mask', 'test_mask']).to_undirected()
#
#         # G = data.to_networkx().to_undirected()
#         # # 添加节点特征和标签到 NetworkX 中
#         # for i in range(data.num_nodes()):
#         #     G.nodes[i]['feat'] = data.ndata['feat'][i].numpy()
#         #     G.nodes[i]['label'] = data.ndata['label'][i].numpy()
#         #     G.nodes[i]['train_mask'] = data.ndata['label'][i].numpy()
#         #     G.nodes[i]['val_mask'] = data.ndata['label'][i].numpy()
#         #     G.nodes[i]['test_mask'] = data.ndata['label'][i].numpy()
#
#
#         nx.set_node_attributes(G,dict([(nid, nid) for nid in range(nx.number_of_nodes(G))]),
#                                name="index_orig")
#
#         partition = community_louvain.best_partition(G)
#
#         cluster2node = {}
#         for node in partition:
#             cluster = partition[node]
#             if cluster not in cluster2node:
#                 cluster2node[cluster] = [node]
#             else:
#                 cluster2node[cluster].append(node)
#
#         max_len = len(G) // self.client_num - self.delta
#         max_len_client = len(G) // self.client_num
#
#         tmp_cluster2node = {}
#         for cluster in cluster2node:
#             while len(cluster2node[cluster]) > max_len:
#                 tmp_cluster = cluster2node[cluster][:max_len]
#                 tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
#                                  1] = tmp_cluster
#                 cluster2node[cluster] = cluster2node[cluster][max_len:]
#         cluster2node.update(tmp_cluster2node)
#
#         orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
#         orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)
#
#         client_node_idx = {idx: [] for idx in range(self.client_num)}
#         idx = 0
#         for (cluster, node_list) in orderedc2n:
#             while len(node_list) + len(
#                     client_node_idx[idx]) > max_len_client + self.delta:
#                 idx = (idx + 1) % self.client_num
#             client_node_idx[idx] += node_list
#             idx = (idx + 1) % self.client_num
#
#         graphs = []
#         for owner in client_node_idx:
#             nodes = client_node_idx[owner]
#             nx_sub = nx.subgraph(G, nodes)
#             subg = dgl.from_networkx(nx_sub, node_attrs=['feat', 'label','train_mask','val_mask','test_mask','index_orig'])
#             subg = inject_contextual_anomalies(graph=subg, k=K, p=P, q=Q_MAP[kwargs.get('data_name')], seed=kwargs.get('seed'))
#             subg = inject_structural_anomalies(graph=subg, p=P, q=Q_MAP[kwargs.get('data_name')], seed=kwargs.get('seed'))
#             mask = split_by_ratio(subg.num_nodes(),frac_list=[0.6,0.4,0],shuffle=True, random_state=1)
#             subg.ndata['train_mask'] = mask[0]
#             subg.ndata['val_mask'] = mask[1]
#             subg.ndata['test_mask'] = mask[2]
#             graphs.append(subg)
#
#         return graphs
#
#
# # def splite_by_ratio(graph,frac_list=None, shuffle=False, random_state=None):
#     from itertools import accumulate
#     from sklearn.model_selection import train_test_split
#     # if frac_list is None:
#     #     frac_list = [0.6, 0.4,0]
#     # frac_list = np.asarray(frac_list)
#     # assert np.allclose(np.sum(frac_list), 1.), \
#     #     'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
#     # num_data = graph.num_nodes()
#     # lengths = (num_data * frac_list).astype(int)
#     # lengths[-1] = num_data - np.sum(lengths[:-1])
#     #
#     # if shuffle:
#     #     indices = np.random.RandomState(
#     #         seed=random_state).permutation(num_data)
#     # else:
#     #     indices = np.arange(num_data)
#     # train_indices = indices[:lengths[0]]
#     # val_indices = indices[lengths[0]:lengths[0] +lengths[1]]
#     # test_indices = indices[lengths[0] +lengths[1]:]
#     #
#     # train_mask = torch.zeros(num_data, dtype=torch.bool)
#     # val_mask = torch.zeros(num_data, dtype=torch.bool)
#     # test_mask = torch.zeros(num_data, dtype=torch.bool)
#     #
#     # train_mask[train_indices] = True
#     # val_mask[val_indices] = True
#     # test_mask[test_indices] = True
#     # return [train_mask,val_mask,test_mask]
#     # return [Subset(graph.nodes(), indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]

from sklearn.model_selection import train_test_split
def split_by_ratio(num_data, frac_list=None, shuffle=False, random_state=None):
    if frac_list is None:
        frac_list = [0.8, 0.2, 0]
    frac_list = np.asarray(frac_list)
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])

    if shuffle:
        indices = np.random.RandomState(
            seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    train_indices = indices[:lengths[0]]
    val_indices = indices[lengths[0]:lengths[0] + lengths[1]]
    test_indices = indices[lengths[0] + lengths[1]:]

    train_mask = torch.zeros(num_data, dtype=torch.bool)
    val_mask = torch.zeros(num_data, dtype=torch.bool)
    test_mask = torch.zeros(num_data, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask


class Subset(object):
    """Subset of a dataset at specified indices

    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]


    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)