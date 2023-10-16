#!/bin/bash
# bash fedCola.bash

data_types=("cora-inj-before" "citeseer-inj-before") #
conditions=("local")
client_num=(3 5 10)
anomaly_type=('s' 'cs')

for data_type in "${data_types[@]}"; do
  for client_num in "${client_num[@]}"; do
      for condition in "${conditions[@]}"; do
          for anomaly_type in "${anomaly_type[@]}"; do
            for seed in {1..3};do
      python fedCola.py --cfg yamlFile/fed_cola.yaml device 4 federate.method local data.type $data_type federate.client_num $client_num federate.method $condition seed $seed data.anomaly_type $anomaly_type
done
done
done
done
done