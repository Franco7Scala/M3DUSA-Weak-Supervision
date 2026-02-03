import torch
import numpy

from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from src.support.utils import compute_auc
from src.support.utils_graph import compute_incremental_con_measure


def train(model, data, optimizer, criterion, train_mask, scheduler=None, n_epochs=200):
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion((out[0][data.target_type], out[1][data.target_type], out[2][data.target_type]), data[data.target_type].y, train_mask)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}")

    return model


def evaluate(model, data, mask, connectivity_bound=20):
    model.eval()
    with torch.no_grad():
        out_ic_classification_full, out_ic_regression_full, out_proxy_regression_full = model(data.x_dict, data.edge_index_dict)
        out_ic_classification = out_ic_classification_full[data.target_type][mask.squeeze().bool()]
        out_ic_regression = out_ic_regression_full[data.target_type][mask.squeeze().bool()]
        out_proxy_regression = out_proxy_regression_full[data.target_type][mask.squeeze().bool()]
        report = {}
        # IC Classification Head
        ground_truth = data[data.target_type].y[0][mask.squeeze().bool()].cpu().argmax(dim=-1)
        pred = out_ic_classification.argmax(dim=-1).cpu()
        pred_prob = torch.nn.functional.softmax(out_ic_classification, -1)
        f1_micro = f1_score(ground_truth, pred, average="micro")
        f1_macro = f1_score(ground_truth, pred, average="macro")
        precision = precision_score(ground_truth, pred, average=None, zero_division=0)
        recall = recall_score(ground_truth, pred, average=None, zero_division=0)
        auc = compute_auc(data[data.target_type].y[0][mask.squeeze().bool()].cpu().numpy(), pred_prob.cpu().detach().numpy())
        report["ic_classification_head"] = {"f1_micro": f1_micro, "f1_macro": f1_macro, "auc": auc, "precision": precision, "recall": recall}

        def compute_regression_metrics(true, pred):
            mae = mean_absolute_error(true, pred)
            mse = mean_squared_error(true, pred)
            rmse = numpy.sqrt(mse)
            r2 = r2_score(true, pred)
            mape = numpy.mean(numpy.abs((true - pred) / true)) * 100
            return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

        # IC Regression Head
        ground_truth = data[data.target_type].y[1][mask.squeeze().bool()].cpu().numpy()
        pred = out_ic_regression.cpu().numpy()
        report["ic_regression_head"] = compute_regression_metrics(ground_truth, pred)
        # Proxy Regression Head
        ground_truth = data[data.target_type].y[2][mask.squeeze().bool()].cpu().numpy()
        pred = out_proxy_regression.cpu().numpy()
        report["proxy_regression_head"] = compute_regression_metrics(ground_truth, pred)
        # Connectivity measures
        if connectivity_bound > 0:
            indices = torch.argsort(out_ic_regression_full[data.target_type].squeeze(), descending=False).cpu().squeeze().tolist()[:connectivity_bound]
            report["con_measures"] = compute_incremental_con_measure(indices, data, False)

        return report
