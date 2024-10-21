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

data_types=("tfinance")
conditions=("local" "fedavg")
client_num=(3 5 10)
anomaly_type=('cs')
for data_type in "${data_types[@]}"; do
  for client_num in "${client_num[@]}"; do
      for condition in "${conditions[@]}"; do
          for anomaly_type in "${anomaly_type[@]}"; do
            for seed in {11..13};do
      python fedCola.py --cfg yamlFile/fed_cola.yaml device 3 data.splitter 'random_partition' federate.method local data.type $data_type federate.client_num $client_num federate.method $condition seed $seed data.anomaly_type $anomaly_type
done
done
done
done
done


#
#data_types=("pubmed-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "amazon-c-inj-before" "amazon-p-inj-before")  #
#conditions=("local" "fedavg")
#client_num=(3 5 10)
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