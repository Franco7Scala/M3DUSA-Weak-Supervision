#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/projects/InfluentialNodes

dataset_name="politifact" #"mumin" #politifact
num_layers=2 #3 #2 pe rpolitifact
hidden_channels=64
dropout=0.3
learning_rate=0.005

drop_percentages=(80 70 60 50 40 30 20 10)
training_seeds=(42 123 12345 123123 2025)

base_dir="/home/jovyan/projects/InfluentialNodes/datasets/$dataset_name/"
embeddings_dir="$base_dir/embeddings_sl"
results_dir="$base_dir/results_sl"

for dir in "$embeddings_dir" "$results_dir"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

for drop_percentage in "${drop_percentages[@]}"; do
  for seed in "${training_seeds[@]}"; do
      python src/main_es.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$embeddings_dir" --drop-percentage "$drop_percentage"
      python src/main_surrogate.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$embeddings_dir" --drop-percentage "$drop_percentage"
  done
done

echo "All experiments completed!"
