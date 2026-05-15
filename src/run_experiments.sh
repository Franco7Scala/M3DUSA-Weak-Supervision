#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/Projects/M3DUSA_active/

dataset_name="mumin" #politifact

num_layers=3 #2 pe rpolitifact
hidden_channels=64
dropout=0.3
learning_rate=0.005

training_seeds=(42 123 12345 123123 2025)

base_dir="/home/martirano/data/$dataset_name/"
embeddings_dir="$base_dir/embeddings_M3DUSA_active"
results_dir="$base_dir/results_M3DUSA_active"

for dir in "$embeddings_dir" "$results_dir"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done



for seed in "${training_seeds[@]}"
do
    echo "### Running experiment with dataset: $dataset_name, seed: $seed, num-layers "$num_layers", hidden-channels "$hidden_channels", dropout "$dropout", learning-rate "$learning_rate" ###"
    echo "Saving in "$results_dir", "$embeddings_dir""
    python src/main_ES.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$embeddings_dir"
done