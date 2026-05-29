#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/projects/InfluentialNodes

dataset_name="politifact" #"mumin" #politifact
num_layers=2 #3 #2 pe rpolitifact
hidden_channels=64
dropout=0.3
learning_rate=0.005
weight_main_component=1
train_baseline=false

drop_percentages=(60 70 80 90 95 98 99)
weight_proxy_components=(1 0.8 0.6 0.4 0.2 0)
weight_consistencies=(1 0.8 0.6 0.4 0.2 0)
training_seeds=(42 78235 10492 88888 559312)

base_dir="/home/jovyan/projects/InfluentialNodes/datasets/$dataset_name/"
embeddings_dir="$base_dir/embeddings_sl"
results_dir="$base_dir/results_sl"

if $train_baseline; then
  echo "Training baseline model..."
  for drop_percentage in "${drop_percentages[@]}"; do
    for seed in "${training_seeds[@]}"; do
        current_embs_dir="${embeddings_dir}/drop_${drop_percentage}_seed_${seed}"
        mkdir -p "$current_embs_dir"
        python src/main_es.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$current_embs_dir" --drop-percentage "$drop_percentage"
        rm -rf "$current_embs_dir"
    done
  done
fi

for drop_percentage in "${drop_percentages[@]}"; do
  for seed in "${training_seeds[@]}"; do
    for weight_proxy_component in "${weight_proxy_components[@]}"; do
      for weight_consistency in "${weight_consistencies[@]}"; do
        current_embs_dir="${embeddings_dir}/drop_${drop_percentage}_seed_${seed}"
        echo "Running experiment with drop_percentage=$drop_percentage, seed=$seed, weight_proxy_component=$weight_proxy_component, weight_consistency=$weight_consistency"
        mkdir -p "$current_embs_dir"
        python src/main_surrogate.py --dataset-name "$dataset_name" --seed "$seed" --num-layers "$num_layers" --hidden-channels "$hidden_channels" --dropout "$dropout" --learning-rate "$learning_rate" --results-dir "$results_dir" --embs-dir "$current_embs_dir" --drop-percentage "$drop_percentage" --weight-main-component "$weight_main_component" --weight-proxy-component "$weight_proxy_component" --weight-consistency "$weight_consistency"
        rm -rf "$current_embs_dir"
      done
    done
  done
done

message="Experiments%20SurrogateLoss%20completed!"

for chat_id in "${chat_ids[@]}"; do
    url="https://api.telegram.org/bot${token}/sendMessage?chat_id=${chat_id}&text=${message}"
    curl -s -o /dev/null "$url"
done

echo "All experiments completed!"
