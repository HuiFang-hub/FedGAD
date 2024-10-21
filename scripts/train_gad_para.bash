#!/bin/bash
# bash train_gad_para.bash



#data_types=("twitch-pt-inj-before")  #
#alphas=(0.2 0.4 0.6 0.8)
#for data_type in "${data_types[@]}"; do
#  for alpha in "${alphas[@]}"; do
#      python train_gad_para.py  --cfg configs/train_dag_para.yaml device 5 federate.method fedsagegod federate.client_num 3 data.anomaly_type 'cs' dataloader.batch_size 100 fedsageplus.fedgen_epoch 20 data.type $data_type seed 5 model.alpha $alpha
#done
#done
#
#
#data_types=( "amazon")  #
#alphas=(0 0.2 0.4 0.6 0.8 1)
#for data_type in "${data_types[@]}"; do
#  for alpha in "${alphas[@]}"; do
#      python train_gad_para.py  --cfg configs/train_dag_para.yaml device 5 federate.method fedsagegod federate.client_num 3 data.anomaly_type 'cs' dataloader.batch_size 100 fedsageplus.fedgen_epoch 20 data.type $data_type seed 5 model.alpha $alpha
#done
#done

data_types=("amazon")
alphas=(0.8)

# 打开文件以追加写入
for data_type in "${data_types[@]}"; do
  for alpha in "${alphas[@]}"; do
      # 构建执行语句
      python train_gad_para.py --cfg yamlFile/train_gad_para.yaml device 0 federate.method fedsagegod federate.client_num 3 data.anomaly_type 'cs' dataloader.batch_size 100 fedsageplus.fedgen_epoch 20 data.type $data_type seed 5 model.alpha $alpha

  done
done

# 关闭文件


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