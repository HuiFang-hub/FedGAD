#!/bin/bash
data_types=("tfinance" "yelp" "amazon") #
client_nums=(3 5 10)
anomaly_types=('cs')
batch_sizes=(100)
seeds=(5 6 7)
for data_type in "${data_types[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for client_num in "${client_nums[@]}"; do
            for anomaly_type in "${anomaly_types[@]}"; do
              for seed in "${seeds[@]}";do
        python train_local.py  --cfg yamlFile/train_local.yaml device 5 data.splitter 'random_partition' federate.method fedavg data.type $data_type federate.client_num $client_num federate.method fedavg seed $seed data.anomaly_type $anomaly_type dataloader.batch_size $batch_size
done
done
done
done
done
