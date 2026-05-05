import os
import torch
import torch.nn.functional as F

from src._trash._competitors.HHGNN.HHGNN import heterodata_to_global_adj, get_mo_cam, HHGNN
from src.dataset_loader import load_dataset
from src._trash.influence.simulation.influence_score_ic import compute_influence_scores
from src._trash.support.utils import get_device, get_base_dir
from src._trash.support.utils_graph import build_metapath_graphs, compute_layer_probabilities

if __name__ == "__main__":
    # --- Configurazione ---
    dataset_name = "imdb"  # 'imdb', 'dblp', etc.
    beta = 0.55
    num_steps = 3
    n_sim = 1  # Ridotto per test veloce, aumentalo a 200+ per risultati reali
    reduction_factor = 4  # Usa un fattore più alto se il grafo è grande per evitare OOM con HHGNN denso
    num_hops = 2

    # Parametri HHGNN
    hhgnn_K = 2  # Ordine dei path (K-hop)
    hidden_dim = 64
    epochs = 1
    lr = 0.005

    results_dir = os.path.join(get_base_dir(), dataset_name, "hhgnn_results")
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()

    print(f"--- Caricamento Dataset: {dataset_name} ---")
    data = load_dataset(dataset_name, reduction_factor=reduction_factor, k=num_hops, device=device)
    target_type = data.target_type
    print(f"Target Type: {target_type} | Num Nodes: {data[target_type].num_nodes}")

    # --- Calcolo Ground Truth (IC Scores) ---
    print("--- Calcolo IC Scores (Ground Truth) ---")
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, beta)

    ic_scores_dict = compute_influence_scores(
        layer_graphs=layer_graphs,
        layer_probs=layer_probs,
        num_steps=num_steps,
        n_sim=n_sim,
        seed=42,
        out_dir=results_dir
    )

    # Converti ic_scores_dict in tensore Y allineato con data[target_type].x
    # Assumiamo che gli indici in ic_scores_dict corrispondano agli indici locali 0..N del target type
    num_targets = data[target_type].num_nodes
    y_true = torch.zeros((num_targets, 1), device=device)

    # ic_scores_dict potrebbe avere chiavi stringa o int, normalizziamo
    max_score = 0
    for node_idx, score in ic_scores_dict.items():
        idx = int(node_idx)
        if idx < num_targets:
            y_true[idx] = score
            if score > max_score: max_score = score

    # Normalizza labels [0, 1] per stabilità
    if max_score > 0:
        y_true = y_true / max_score

    # --- Preparazione Dati HHGNN (MO-CAM) ---
    print("--- Costruzione MO-CAM per HHGNN ---")
    # Nota: HHGNN richiede la matrice di adiacenza globale per calcolare i path
    global_adj, target_global_indices = heterodata_to_global_adj(data.cpu(), target_type)

    # Sposta su device e calcola MO-CAM
    #TODO manca il metodo get_mo_cam
    mo_cam_layers = get_mo_cam(global_adj, target_global_indices, K=hhgnn_K, device=device)

    # Feature Input
    x_input = data[target_type].x.to(device)

    # Split Train/Test (Esempio: 60% train, 20% val, 20% test)
    indices = torch.randperm(num_targets)
    split1 = int(0.6 * num_targets)
    split2 = int(0.8 * num_targets)
    train_idx = indices[:split1]
    val_idx = indices[split1:split2]
    test_idx = indices[split2:]

    # --- Inizializzazione Modello ---
    model = HHGNN(in_dim=x_input.shape[1], hidden_dim=hidden_dim, K=hhgnn_K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --- Training Loop ---
    print("--- Avvio Training HHGNN ---")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred_scores = model(x_input, mo_cam_layers)  # Output [N, 1]

        # Loss calcolata solo su nodi di training
        train_pred = pred_scores[train_idx]
        train_true = y_true[train_idx]

        mse_loss = F.mse_loss(train_pred, train_true)

        # Pairwise Ranking Loss
        # diff_pred: [M, M], diff_true: [M, M]
        diff_pred = train_pred - train_pred.t()
        diff_true = train_true - train_true.t()
        # Loss = ReLU(-sign(diff_true) * diff_pred)
        rank_loss = torch.relu(-torch.sign(diff_true) * diff_pred).mean()

        total_loss = 0.5 * mse_loss + 0.5 * rank_loss

        total_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = pred_scores[val_idx]
            val_true = y_true[val_idx]
            val_mse = F.mse_loss(val_pred, val_true)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                # Salva checkpoint se necessario

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss.item():.4f} (MSE: {mse_loss:.4f}, Rank: {rank_loss:.4f}) | Val MSE: {val_mse:.4f}")

    # --- Evaluation Finale ---
    print("--- Evaluation sul Test Set ---")
    model.eval()
    with torch.no_grad():
        final_scores = model(x_input, mo_cam_layers)
        test_pred = final_scores[test_idx].flatten()
        test_true = y_true[test_idx].flatten()

        # Metriche
        from scipy.stats import kendalltau

        # Kendall's Tau
        tau, _ = kendalltau(test_pred.cpu().numpy(), test_true.cpu().numpy())

        # Top-K Precision (Intersezione dei top-K nodi)
        k = 10
        _, topk_pred_indices = torch.topk(test_pred, k)
        _, topk_true_indices = torch.topk(test_true, k)

        intersection = len(set(topk_pred_indices.cpu().numpy()) & set(topk_true_indices.cpu().numpy()))
        precision_at_k = intersection / k

        print(f"Risultati Test:")
        print(f"Kendall's Tau: {tau:.4f}")
        print(f"Precision@{k}: {precision_at_k:.4f}")

        # Salva predizioni
        out_data = {
            "node_indices": test_idx.cpu().numpy().tolist(),
            "true_scores": test_true.cpu().numpy().tolist(),
            "pred_scores": test_pred.cpu().numpy().tolist()
        }
        import json
        with open(os.path.join(results_dir, "hhgnn_evaluation.json"), 'w') as f:
            json.dump(out_data, f)
