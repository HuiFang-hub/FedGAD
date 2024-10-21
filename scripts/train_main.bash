#!/bin/bash
# bash train_fedsage.bash

data_types=( "yelp")  #
client_nums=(10)
anomaly_types=('cs')
seeds=(5)
batch_sizes=(50)
for data_type in "${data_types[@]}"; do
   for batch_size in "${batch_sizes[@]}"; do
  for client_num in "${client_nums[@]}"; do
      for anomaly_type in "${anomaly_types[@]}"; do
          for seed in "${seeds[@]}";do
      python main.py  --cfg yamlFile/train_main.yaml device 1 data.type $data_type federate.client_num $client_num  seed $seed data.anomaly_type $anomaly_type dataloader.batch_size $batch_size
done
done
done
done
done
#
#data_types=(  "citeseer-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "facebookpagepage-inj-before")  #
#client_nums=(10)
#anomaly_types=('cs')
#seeds=(5 6 7)
#batch_sizes=(50)
#for data_type in "${data_types[@]}"; do
#   for batch_size in "${batch_sizes[@]}"; do
#  for client_num in "${client_nums[@]}"; do
#      for anomaly_type in "${anomaly_types[@]}"; do
#          for seed in "${seeds[@]}";do
#      python train_fedsage.py  --cfg configs/train_main.yaml device 5 data.type $data_type federate.client_num $client_num seed $seed data.anomaly_type $anomaly_type dataloader.batch_size $batch_size
#done
#done
#done
#done
#done
#
##"cora-inj-before"
#
#data_types=( "facebookpagepage-inj-before"  )  #
#client_nums=(5 3)
#batch_sizes=(50)
#anomaly_types=('cs')
#seeds=(5 6 7)
#for data_type in "${data_types[@]}"; do
#    for batch_size in "${batch_sizes[@]}"; do
#  for client_num in "${client_nums[@]}"; do
#      for anomaly_type in "${anomaly_types[@]}"; do
#          for seed in "${seeds[@]}";do
#      python train_fedsage.py  --cfg configs/train_main.yaml device 5 data.type $data_type federate.client_num $client_num  seed $seed data.anomaly_type $anomaly_type dataloader.batch_size $batch_size
#done
#done
#done
#done
#done
##
#data_types=("citeseer-inj-before" "amazon")  #
#client_nums=(10)
#anomaly_types=('cs')
#seeds=(5 6 7)
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_nums[@]}"; do
#      for anomaly_type in "${anomaly_types[@]}"; do
#          for seed in "${seeds[@]}";do
#      python train_fedsage.py  --cfg configs/train_main.yaml device 5 federate.method fedsagegod data.type $data_type federate.client_num $client_num federate.method fedsagegod seed $seed data.anomaly_type $anomaly_type dataloader.batch_size 100
#done
#done
#done
#done



