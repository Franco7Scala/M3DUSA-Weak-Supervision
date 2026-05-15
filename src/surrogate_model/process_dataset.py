import torch

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_embeddings(dataset, model, tokenizer, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_texts, batch_labels in tqdm(dataloader):
            # tokenization
            inputs = tokenizer(batch_texts).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings)
            all_labels.extend(batch_labels.tolist())

    # concatenating all batches
    x = torch.cat(all_embeddings, dim=0)
    y = torch.tensor(all_labels)

    return x, y
