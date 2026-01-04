import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils.mask import mask_to_index
from tqdm import tqdm
from src.utils import compute_auc, merge_masks
from src.utils_graph import k_hop_subgraph


def train(model, all_data, new_nodes, old_nodes, optimizer, criterion, scheduler, run, strategy, directory, reduction_factor, n_epochs=200, kwargs=None):
    n_split = max(1, int(reduction_factor * n_epochs))
    data_splits = strategy.sample(n_split, all_data, new_nodes["train"], old_nodes["train"], kwargs)
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        train_data = data_splits[int(epoch%n_split)][0]
        out = model(train_data.x_dict, train_data.edge_index_dict)
        loss = criterion(out[train_data.target_type], train_data[train_data.target_type].y)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}")

    return model


def evaluate(model, all_data, new_nodes, old_nodes, run, directory):
    test_data = k_hop_subgraph(all_data, merge_masks([new_nodes["test"], old_nodes["test"]])[all_data.target_type], 2)[0]
    model.eval()
    with torch.no_grad():
        out = model(test_data.x_dict, test_data.edge_index_dict)
        pred = out[test_data.target_type].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(test_data.x_dict, test_data.edge_index_dict)[test_data.target_type], -1)
        f1_micro = f1_score(test_data[test_data.target_type].y.cpu(), pred.cpu(), average="micro")
        f1_macro = f1_score(test_data[test_data.target_type].y.cpu(), pred.cpu(), average="macro")
        auc = compute_auc(test_data[test_data.target_type].y.cpu().numpy(), pred_prob.cpu().detach().numpy())
        precision = precision_score(test_data[test_data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        recall = recall_score(test_data[test_data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        return f1_micro, f1_macro, auc, precision, recall
