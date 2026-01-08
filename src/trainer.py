import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils.mask import mask_to_index
from tqdm import tqdm
from src.utils import compute_auc


def train(model, data, optimizer, criterion, scheduler, n_epochs=200):
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out[data.target_type], data[data.target_type].y)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}")

    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out[data.target_type].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(data.x_dict, data.edge_index_dict)[data.target_type], -1)
        f1_micro = f1_score(data[data.target_type].y.cpu(), pred.cpu(), average="micro")
        f1_macro = f1_score(data[data.target_type].y.cpu(), pred.cpu(), average="macro")
        auc = compute_auc(data[data.target_type].y.cpu().numpy(), pred_prob.cpu().detach().numpy())
        precision = precision_score(data[data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        recall = recall_score(data[data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        return f1_micro, f1_macro, auc, precision, recall
