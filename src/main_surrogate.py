import torch
import torch.nn as nn
import os

from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from src.surrogate_model.classification_head import RobertaClassificationHead
from src.surrogate_model.dateset_loader import load_datasets
from src.surrogate_model.process_dataset import extract_embeddings
from src.surrogate_model.text_dataset import TextDataset
from src.surrogate_model.trainer import train, evaluate
from src.utils import print_metrics


if __name__ == "__main__":
    model_name = "FacebookAI/roberta-base"
    dataset_path = "/data/dataset_embeddings_by_roberta.pt"
    model_save_path = "/models/classification_head.pt"
    epochs = 10
    batch_size = 32
    random_state = 42

    #############################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if os.path.exists(dataset_path):
        print("Loading datasets from pre-built...")
        data = torch.load(dataset_path)
        x = data["x"]
        y = data["y"]
        dataset = TextDataset(x, y)

    else:
        print("Loading datasets from sources...")
        texts, labels = load_datasets()
        dataset = TextDataset(texts, labels)

        print(f"Extracting embeddings...")
        x, y = extract_embeddings(texts, labels)

        print(f"Saving dataset...")
        torch.save({"x": x, "y": y}, dataset_path)

    x_numpy = x.cpu().numpy()
    y_numpy = y.cpu().numpy()
    x_train, x_test, y_train, y_test = train_test_split(x_numpy, y_numpy, test_size=0.2, random_state=random_state, stratify=y_numpy)
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing head...")
    head_model = RobertaClassificationHead(hidden_size=768, num_labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head_model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\nTraining head...")
    model = train(model, optimizer, criterion, epochs, train_dataloader, device)

    print("\nEvaluating head...")
    report = evaluate(model, test_dataloader, device)
    print_metrics(report)

    print("\nSaving head...")
    torch.save(head_model.state_dict(), model_save_path)
