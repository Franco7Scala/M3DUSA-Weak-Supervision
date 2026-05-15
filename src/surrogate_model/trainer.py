import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train(model, optimizer, criterion, epochs, train_dataloader, device):
    pbar = tqdm(range(epochs), desc="Training Epochs")
    for epoch in pbar:
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        pbar.set_postfix(epoch=epoch, train_loss=f"{avg_train_loss:.4f}")

    return model


def evaluate(model, test_dataloader, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            all_targets.extend(batch_y.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision_macro": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "precision_micro": precision_score(all_targets, all_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(all_targets, all_preds, average="micro", zero_division=0),
        "f1_micro": f1_score(all_targets, all_preds, average="micro", zero_division=0),
        "auc": roc_auc_score(all_targets, all_probs)
    }
