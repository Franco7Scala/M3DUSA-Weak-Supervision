import torch
import pandas
import transformers
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from src.support.utils import compare_labels, print_metrics
from src.surrogate_model.classification_head import RobertaClassificationHead


if __name__ == "__main__":
    embedder_model_name = "FacebookAI/roberta-base"
    classifier_model_path = "/home/jovyan/projects/InfluentialNodes/models/classification_head.pt"
    #csv_input_path = "/home/jovyan/projects/InfluentialNodes/datasets/politifact/news_embeddings.csv"
    #csv_output_path = "/home/jovyan/projects/InfluentialNodes/datasets/politifact/news_texts_with_label.csv"
    csv_input_path = "/home/jovyan/projects/InfluentialNodes/datasets/mumin/claims.csv"
    csv_output_path = "/home/jovyan/projects/InfluentialNodes/datasets/mumin/claims_with_label.csv"


    #############################################################################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print(f"Loading models...")
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)
    model = AutoModel.from_pretrained(embedder_model_name).to(device)
    model.eval()
    classifier = RobertaClassificationHead().to(device)
    classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier.eval()

    print(f"Loading texts...")
    df = pandas.read_csv(csv_input_path)

    if "politifact" in csv_input_path:
        print(f"Computing embeddings and surrogate-label for politifact...")
        embeddings = []
        surrogate_labels = []
        confidences = []
        for text in tqdm(df["text"].tolist(), desc="Computing:"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            embedding = outputs.last_hidden_state[0, 0, :]
            surrogate_label = classifier(embedding.unsqueeze(0)).argmax(dim=1).item()
            embeddings.append(embedding.cpu().tolist())
            surrogate_labels.append(0 if surrogate_label == 1 else 1)
            confidences.append(1 - F.softmax(classifier(embedding.unsqueeze(0)),dim=1).min().item())

        df["embedding_roberta"] = embeddings

    elif "mumin" in csv_input_path:
        print(f"Computing surrogate-labels for mumin...")
        surrogate_labels = []
        confidences = []
        for embedding in tqdm(df["embedding"].tolist(), desc="Computing:"):
            embedding = embedding.strip().replace("[", "").replace("]", "").replace("\n", "").replace("  ", " ").split(" ")
            embedding = [float(s) for s in embedding if s != ""]
            embedding = torch.Tensor(embedding).to(device)
            surrogate_label = classifier(embedding.unsqueeze(0)).argmax(dim=1).item()
            surrogate_labels.append(0 if surrogate_label == 1 else 1)
            confidences.append(1 - F.softmax(classifier(embedding.unsqueeze(0)), dim=1).min().item())

        df["label"] = df["label"].map({"misinformation": 1, "factual": 0})

    else:
        raise Exception("Unknown dataset!")

    df["surrogate_labels"] = surrogate_labels
    df["confidences"] = confidences
    print_metrics({"RESULTS": compare_labels(df["label"].tolist(), df["surrogate_labels"].tolist())})
    print(f"Saving to csv...")
    df.to_csv(csv_output_path, index=False)
    print(f"Completed!")
