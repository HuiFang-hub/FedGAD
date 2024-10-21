#!/bin/bash
# bash train_fedsage_para.bash

data_types=("cora-inj-before" )  #
fedsageplus_a=(0.1 )
fedsageplus_b=(0.5)
for data_type in "${data_types[@]}"; do
  for a in "${fedsageplus_a[@]}"; do
  for b in "${fedsageplus_b[@]}"; do
      python train_fedsage_para.py  --cfg yamlFile/train_fedsage.yaml device 2 federate.method fedsagegod federate.client_num 5 data.anomaly_type 'cs' dataloader.batch_size 100 fedsageplus.fedgen_epoch 20 data.type $data_type seed 5 fedsageplus.a $a fedsageplus.b $b
done
done
done


data_types=("citeseer-inj-before" )  #
fedsageplus_a=(0.9 )
fedsageplus_b=(0.9)
for data_type in "${data_types[@]}"; do
  for a in "${fedsageplus_a[@]}"; do
  for b in "${fedsageplus_b[@]}"; do
      python train_fedsage_para.py  --cfg yamlFile/train_fedsage.yaml device 2 federate.method fedsagegod federate.client_num 5 data.anomaly_type 'cs' dataloader.batch_size 100 fedsageplus.fedgen_epoch 20 data.type $data_type seed 5 fedsageplus.a $a fedsageplus.b $b
done
done
done



#"cora-inj-before"

#data_types=( "citeseer-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "tfinance" "yelp" "amazon")  #
#client_nums=(3 5 10)
#anomaly_types=('cs')
#seeds=(5 6 7)
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_nums[@]}"; do
#      for anomaly_type in "${anomaly_types[@]}"; do
#          for seed in "${seeds[@]}";do
#      python train_fedsage.py  --cfg configs/train_main.yaml device 0 federate.method local data.type $data_type federate.client_num $client_num federate.method fedsagegod seed $seed data.anomaly_type $anomaly_type dataloader.batch_size 100
#done
#done
#done
#done