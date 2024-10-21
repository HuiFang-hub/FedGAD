set -e

cudaid=$1
dataset=$2
splitter=$3
gnn=$4
lr=$5
local_update=$6

cd ../../../../..

if [ ! -d "out" ];then
    mkdir out
fi

if [[ $dataset = 'cora' ]]; then
    out_channels=7
    hidden=64
elif [[ $dataset = 'citeseer' ]]; then
    out_channels=6
    hidden=64
elif [[ $dataset = 'pubmed' ]]; then
    out_channels=3
    hidden=64
else
    out_channels=4
    hidden=1024
fi

if [[ $gnn = 'gpr' ]]; then
    layer=10
else
    layer=2
fi

echo "HPO starts..."

mu=(0.1 1.0 5.0)

for (( s=0; s<${#mu[@]}; s++ ))
do
    for k in {1..5}
    do
        python src.federatedscope/main.py --cfg src.federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} train.optimizer.lr ${lr} train.local_update_steps ${local_update} model.type ${gnn} model.out_channels ${out_channels} model.hidden ${hidden} seed $k fedprox.use True fedprox.mu ${mu[$s]} model.layer ${layer} >>out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${mu[$s]}_prox.log 2>&1
    done
done

echo "HPO ends."
