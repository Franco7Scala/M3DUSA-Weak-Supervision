#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/projects/InfluentialNodes

dataset_name="politifact" #"mumin" #politifact
num_layers=2 #3 #2 pe rpolitifact
hidden_channels=64
dropout=0.3
learning_rate=0.005
weight_main_component=1
weight_proxy_component=0.3
weight_consistency=0.2

drop_percentages=(80 70 60 50 40 30 20 10)
training_seeds=(5 4 3 2 1)

base_dir="/home/jovyan/projects/InfluentialNodes/datasets/$dataset_name/"
embeddings_dir="$base_dir/embeddings_sl"
results_dir="$base_dir/results_sl"

for drop_percentage in "${drop_percentages[@]}"; do
  for seed in "${training_seeds[@]}"; do
      current_embs_dir="${embeddings_dir}/drop_${drop_percentage}_seed_${seed}"

      mkdir -p "$current_embs_dir"
      python src/main_es.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$current_embs_dir" --drop-percentage "$drop_percentage"
      rm -rf "$current_embs_dir"

      mkdir -p "$current_embs_dir"
      python src/main_surrogate.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$current_embs_dir" --drop-percentage "$drop_percentage" --weight-main-component "$weight_main_component" --weight-proxy-component "$weight_proxy_component" --weight-consistency "$weight_consistency"
      rm -rf "$current_embs_dir"

  done
done

echo "All experiments completed!"
