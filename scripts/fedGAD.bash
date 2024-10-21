#!/bin/bash
# bash fedGAD.bash

#lr_values=(0.00155 0.003 0.01)
#batch_size_values=(100 300)
#client_num_values=(3 6 9)


## 循环遍历所有参数值
#for lr in "${lr_values[@]}"; do
#    for batch_size in "${batch_size_values[@]}"; do
#        for client_num in "${client_num_values[@]}"; do
#            # 构造参数字符串
#            # 调用程序，并将参数传递给它
#            echo "python fedsage+god_test.py --cfg configs/fedsage+god.yaml federate.client_num $client_num dataloader.batch_size $batch_size train.optimizer.lr $lr"
#            python fedsage+god_test.py --cfg configs/fedsage+god.yaml federate.client_num $client_num dataloader.batch_size $batch_size train.optimizer.lr $lr
#        done
#    done
#done

## 循环遍历所有参数值
#data_types=("cora-inj-before" "citeseer-inj-before")
#for seed in {1..3};
#do
#  for data_type in "${data_types[@]}"; do
#  python fedGAD.py --cfg configs/fedsage+god.yaml data.type "cora-inj-before" seed $seed data.type $data_type
#done
#done


#data_types=("cora-inj-before" "citeseer-inj-before" "pubmed-inj-before" "facebookpagepage-inj-before" "twitch-pt-inj-before" "twitch-de-inj-before" "amazon-c-inj-before" "amazon-p-inj-before") #
#client_num=(5 10)
#anomaly_type=('c')
#
#for data_type in "${data_types[@]}"; do
#  for client_num in "${client_num[@]}"; do
#          for anomaly_type in "${anomaly_type[@]}"; do
#            for seed in {1..3};do
#      python fedGAD.py --cfg configs/fedsage+god.yaml device 4 federate.method local data.type $data_type federate.client_num $client_num federate.method "fedAnemone" seed $seed data.anomaly_type $anomaly_type
#done
#done
#done
#done



data_types=("twitch-pt-inj-before") #
client_num=(3 5)
anomaly_type=('cs')

for data_type in "${data_types[@]}"; do
  for client_num in "${client_num[@]}"; do
    for anomaly_type in "${anomaly_type[@]}"; do
      for seed in {1..3};do
        python main.py --cfg yamlFile/fedsage+god.yaml device 1 federate.method local data.type $data_type federate.client_num $client_num federate.method "fedsage" seed $seed data.anomaly_type $anomaly_type
done
done
done
done

data_types=("amazon-p-inj-before" "amazon-c-inj-before") #
client_num=(3)
anomaly_type=('cs')

for data_type in "${data_types[@]}"; do
  for client_num in "${client_num[@]}"; do
    for anomaly_type in "${anomaly_type[@]}"; do
      for seed in {1..3};do
        python main.py --cfg yamlFile/fedsage+god.yaml device 1 federate.method local data.type $data_type federate.client_num $client_num federate.method "fedsage" seed $seed data.anomaly_type $anomaly_type
done
done
done
done


data_types=("tfinance") #
client_num=(3 5 10)
anomaly_type=('cs')

for data_type in "${data_types[@]}"; do
  for client_num in "${client_num[@]}"; do
    for anomaly_type in "${anomaly_type[@]}"; do
      for seed in {1..3};do
        python main.py --cfg yamlFile/fedsage+god.yaml device 1 data.splitter 'random_partition' federate.method local data.type $data_type federate.client_num $client_num federate.method "fedsage" seed $seed data.anomaly_type $anomaly_type
done
done
done
done