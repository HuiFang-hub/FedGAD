# FedGAD

This repo contains the sample code for reproducing the results of our paper: Contrastive Federated Learning for Graph Anomaly Detection.

## Introduction

Graph anomaly detection refers to identifying abnormal graph nodes or edges that heavily deviate from normal observations. Previous graph anomaly detection approaches, however, rely on global graph observability, which are inapplicable in federated learning where graph data are distributed in multiple data clients and the global graph observability is unrealistic due to privacy concerns with limited data sharing. To address this issue, we introduce in this paper a new federated learning model for graph anomaly detection (FedGAD for short).

FedGAD enables to collaboratively training of a global graph anomaly detection model among multiple data centres where a portion of cross-site graph links are unobservable.

- To aggregate neighbour information across different clients, FedGAD integrates two mask learners to infer unobservable neighbours, based on which the feature information across graphs can be estimated and aggregated.
- FedGAD employs multi-scale contrastive learning which includes both structure-level and contextual-level learning functions to detect graph anomalies in conditions where graph data are of imbalanced distributions.
- Experimental results on benchmark datasets demonstrate the utility of FedGAD compared to baseline methods.

![Framework](https://github.com/user-attachments/assets/dfead95d-7e84-4467-804a-184e1a965215)
**Figure 1. An overview of FedGAD.**

## Results

## Quick Start

```
bash scitpts/fedGAD.bash `
or
`python main.py --cfg configs/main.yaml device 1
```

## Customize the FedGAD

### Create datasets of anomaly detection.

We provide servel datasets in `src/federatedscope/contrib/data/my_data.py`

- Simulated datasets: cora, citeseer, pubmed, facebookpagepage, twitch-pt, twitch-de, amazon-c, amazon-p
- Real-world datasets: tfinance, tsocial, yelp, amazon

And you can customize yourself dataset, take 'yelp' as an example:

```python
def call_truth_data(config,client_cfgs = None):
    data_name = config.data.type.lower()
    if data_name== 'yelp':
        # get data
        dataset = FraudYelpDataset(raw_dir=f'./data/{data_name}')
        # transform into the stype of pyg data.
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label','train_mask', 'val_mask', 'test_mask'])
        graph.ndata['feat'] = graph.ndata['feature']
        graph = dgl.add_self_loop(graph)
        data = dgl_to_pyg(graph)
    return data

# register
register_data("yelp", call_truth_data)
```

If the data set is simulated, there are no real labels of anomalizes, take 'cora' as an example:

```python
def call_my_data(config,client_cfgs = None):
    data_name = config.data.type.lower()
    if data_name == "cora-inj-before": 
        # get data
        graph = Planetoid('./data/cora-inj-before', 'cora', transform=T.NormalizeFeatures())[0]
        # inject two classes of anomalies
        graph, ya, outlier_idx_c = gen_contextual_outliers(graph, n=n_outlier_c, k=50)  # contextual outliers is n
        graph, ys, outlier_idx_s = gen_structural_outliers(graph, m=10, n= n_outlier_s//10, outlierIdx=outlier_idx_c) 
        graph.ay = torch.logical_or(ys, ya).int()
    return data

# register
register_data("cora", call_my_data)
```
