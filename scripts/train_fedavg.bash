#!/bin/bash
# bash fedAnemone.bash

#data_types=("cora-inj-before" "citeseer-inj-before" "pubmed-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "amazon-c-inj-before" "amazon-p-inj-before")  #
#
#data_types=("facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "amazon-c-inj-before" "amazon-p-inj-before")
#conditions=("local" "fedavg")
#client_num=(3)
#anomaly_type=('cs')
#
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_num[@]}"; do
#      for condition in "${conditions[@]}"; do
#          for anomaly_type in "${anomaly_type[@]}"; do
#            for seed in {11..13};do
#      python fedCola.py --cfg configs/fed_cola.yaml device 0 federate.method local data.type $data_type federate.client_num $client_num federate.method $condition seed $seed data.anomaly_type $anomaly_type
#done
#done
#done
#done
#done

#data_types=("amazon-p-inj-before" "amazon-c-inj-before")
#conditions=("local" "fedavg")
#client_num=(3 5)
#anomaly_type=('cs')
#
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_num[@]}"; do
#      for condition in "${conditions[@]}"; do
#          for anomaly_type in "${anomaly_type[@]}"; do
#            for seed in {11..13};do
#      python fedCola.py --cfg configs/fed_cola.yaml device 3 federate.method local data.type $data_type federate.client_num $client_num federate.method $condition seed $seed data.anomaly_type $anomaly_type
#done
#done
#done
#done
#done

#data_types=( "tfinance" "yelp" "amazon" "cora-inj-before" "citeseer-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before")
data_types=("amazon") #
client_nums=(10)
anomaly_types=('cs')
batch_sizes=(50)
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



