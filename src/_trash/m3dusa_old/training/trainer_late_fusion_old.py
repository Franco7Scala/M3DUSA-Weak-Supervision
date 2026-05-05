import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import single
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def train_model(model, train_loader, optimizer, criterion, device, single_embedding, n_epochs=50):
    if single_embedding:
        return train_model_single_embedding(model, train_loader, optimizer, criterion, device, n_epochs)
    else:
        return train_model_multiple_embeddings(model, train_loader, optimizer, criterion, device, n_epochs)


def evaluate_model(model, test_loader, criterion, device, single_embedding):
    if single_embedding:
        return evaluate_model_single_embedding(model, test_loader, criterion, device)
    else:
        return evaluate_model_multiple_embeddings(model, test_loader, criterion, device)


def train_model_single_embedding(model, train_loader, optimizer, criterion, device, n_epochs=50):
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}')

    print('Training completed.')
    return model



def train_model_multiple_embeddings(model, train_loader, optimizer, criterion, device, n_epochs=50):
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        running_loss = 0.0

        for inputs1, inputs2, labels in train_loader:

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs1, inputs2)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs1.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}')

    print('Training completed.')
    return model


def evaluate_model_single_embedding(model, test_loader, criterion, device):
    model.to(device)
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)  # Get probability scores for AUC

            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy()[:, 1])  # Store the probability of the positive class

    epoch_loss = running_loss / len(test_loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weigh = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)
    prec_0 = precision_score(all_labels, all_preds, pos_label=0)
    rec_0 = recall_score(all_labels, all_preds, pos_label=0)
    prec_1 = precision_score(all_labels, all_preds, pos_label=1)
    rec_1 = recall_score(all_labels, all_preds, pos_label=1)

    return f1_micro, f1_macro, f1_weigh, auc, prec_0, rec_0, prec_1, rec_1


def evaluate_model_multiple_embeddings(model, test_loader, criterion, device):
    model.to(device)
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs1, inputs2)
            probs = F.softmax(outputs, dim=1)  # Get probability scores for AUC

            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs1.size(0)

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy()[:, 1])  # Store the probability of the positive class

    epoch_loss = running_loss / len(test_loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weigh = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)
    prec_0 = precision_score(all_labels, all_preds, pos_label=0)
    rec_0 = recall_score(all_labels, all_preds, pos_label=0)
    prec_1 = precision_score(all_labels, all_preds, pos_label=1)
    rec_1 = recall_score(all_labels, all_preds, pos_label=1)

    return f1_micro, f1_macro, f1_weigh, auc, prec_0, rec_0, prec_1, rec_1


#only for the best model --- BilinearModel
def extract_embeddings(model, data_loader, device): #TODO
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            inputs1, inputs2, _ = batch  # Ignore the labels part
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

            embeddings = model.bilinear(inputs1, inputs2)
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all embeddings to get the final embeddings array
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # Ensure this matches the full dataset size

    return all_embeddings