import os
import argparse
import torch
import pandas as pd
import time

from torch_geometric.nn import to_hetero
from src.dataset.dataset_loader import get_target_type, build_heterodata
from src.mixed_loss.mixed_loss import MixedLoss
from src.models.GAT import GAT
from src.training.trainer_surrogate import train_node_classifier, eval_node_classifier
from src.support.utils import get_device, set_random_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M3DUSA_active experiment")
    parser.add_argument("--dataset-name", type=str, default="politifact", help="Name of the dataset (e.g., politifact)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GAT layers")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Size of hidden channels")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--results-dir", type=str, default=os.path.join(os.getcwd(), "results"), help="Path (directory) to store results")
    parser.add_argument("--embs-dir", type=str, default=os.path.join(os.getcwd(), "embeddings"), help="Path (directory) to store embeddings")
    parser.add_argument("--weight-main-component", type=float, default=1.0, help="Weight main component of the loss")
    parser.add_argument("--weight-proxy-component", type=float, default=0.5, help="Weight proxy component of the loss")
    parser.add_argument("--weight-consistency", type=float, default=0.5, help="Weight consistency component of the loss")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_layers = args.num_layers
    hidden_channels = args.hidden_channels
    dropout = args.dropout
    lr = args.learning_rate
    results_dir = args.results_dir
    embeddings_dir = args.embs_dir

    set_random_seed(seed)

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')

    # LOAD THE DATASET
    target_type = get_target_type(dataset_name)
    data = build_heterodata(dataset_name, target_type)

    # SET THE MODEL
    model = GAT(hidden_channels=hidden_channels, dropout=dropout, num_layers=num_layers, out_channels=2) #num layers 3 per mumin, 2 per politifact
    model = to_hetero(model, data.metadata(), aggr='sum')

    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    targets = data[target_type].y
    weight_main_component = args.weight_main_component
    weight_proxy_component = args.weight_proxy_component
    weight_consistency = args.weight_consistency
    criterion = MixedLoss(weight_main_component=weight_main_component, weight_proxy_component=weight_proxy_component, weight_consistency=weight_consistency)

    # TRAIN THE MODEL
    start_time = time.time()
    model = train_node_classifier(model, data, optimizer, criterion, seed, target_type, embeddings_dir, n_epochs=1000, patience=100, epsilon=1e-6)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    # EVALUATE THE MODEL
    f1_micro, f1_macro, f1_weigh, auc, prec_0, rec_0, prec_1, rec_1 = eval_node_classifier(model, data, target_type, seed, embeddings_dir)

    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}')
    print(f'precision_0: {prec_0:.3f}, recall_0: {rec_0:.3f},  precision_1: {prec_1:.3f}, recall_1: {rec_1:.3f}')

    # SAVE THE MODEL
    #model_path = os.path.join(models_dir, f"{dataset_name}_seed{seed}_model.pth")
    #torch.save(model.state_dict(), model_path)

    # SAVE THE RESULTS
    df = pd.DataFrame([{
        'Seed': seed, 'F1_micro': f1_micro, 'F1_macro': f1_macro, 'ROC-AUC': auc, 'Prec_0': prec_0, 'Rec_0': rec_0,
        'Prec_1': prec_1, 'Rec_1': rec_1, 'Time': training_time
    }])
    results_path = os.path.join(results_dir, f'{dataset_name}_seed{seed}_results.xlsx')
    print(df)
    df.to_excel(results_path, index=False)
    print(f"Saved at {os.path.abspath(results_path)}")

    #if run == 4:
    #merging_results_mode(results_dir, mode)

