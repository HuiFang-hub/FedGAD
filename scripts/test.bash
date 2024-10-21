#!/bin/bash
# bash test.bash


#data_types=("cora-inj-before" "citeseer-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "tfinance" "yelp" "amazon")  #
#client_num=(3)
#batch_size=(50 100 200)
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_num[@]}"; do
#    for batch_size in "${batch_size[@]}"; do
#      for seed in {5..7};do
#        cmd="python train_local.py --cfg configs/train_local.yaml device 5 data.splitter 'random_partition' federate.method local data.type $data_type federate.client_num $client_num federate.method local seed $seed data.anomaly_type cs dataloader.batch_size $batch_size"
#        echo "$cmd"
#done
#done
#done
#done

data_types=("cora-inj-before" "citeseer-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "tfinance" "yelp" "amazon")  #
client_nums=(3 5 10)
anomaly_types=('cs')
seeds=(5 6 7)
for data_type in "${data_types[@]}"; do
      for anomaly_type in "${anomaly_types[@]}"; do
        for client_num in "${client_nums[@]}"; do
          for seed in "${seeds[@]}";do
            cmd="python train_fedsage.py  --cfg yamlFile/train_fedsage.yaml device 2 data.splitter 'random_partition' federate.method local data.type $data_type federate.client_num $client_num federate.method fedsage seed $seed data.anomaly_type $anomaly_type"
            echo "$cmd" >> output.txt
done
done
done
done

