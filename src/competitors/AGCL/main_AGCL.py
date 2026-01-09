import os
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# Import locali
from src.dataset_loader import load_dataset
# Assicurati che il file generato precedentemente sia salvato come agcl_model.py nella cartella corretta
from src.competitors.AGCL.AGCL import AGCL, calculate_h_index, sir_simulation


def get_degrees(edge_index, num_nodes):
    """Calcola il grado dei nodi per il sampling strategy."""
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=edge_index.device))
    return deg


def sampling_strategy(edge_index, num_nodes, target_indices, sample_size=50):
    """
    Strategia di campionamento basata sulla distribuzione Power-law (semplificata).
    Il paper suggerisce di dividere i nodi in gruppi basati sul grado e campionare da essi.
    """
    degrees = get_degrees(edge_index, num_nodes)
    target_degrees = degrees[target_indices]

    # Probabilità di campionamento proporzionale al grado (per catturare hub importanti)
    # oppure inversa se si vuole coprire la coda lunga. Il paper suggerisce di preservare la distribuzione.
    # Qui usiamo un campionamento pesato sul grado per trovare nodi influenti per il training.
    weights = target_degrees.float() + 1e-5  # Evita divisione per zero
    sampling_probs = weights / weights.sum()

    sampled_idx_local = torch.multinomial(sampling_probs, sample_size, replacement=False)
    sampled_global_indices = target_indices[sampled_idx_local]

    return sampled_global_indices


def main():
    # --- 1. Configurazione ---
    dataset_name = "imdb"  # Opzioni: "imdb", "dblp", "aminer", "politifact"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iperparametri AGCL
    hidden_channels = 64
    learning_rate = 0.005
    epochs = 50
    batch_size_sampling = 100  # Numero di nodi campionati per calcolare la SIR Loss per epoca
    phi = 0.5  # Peso tra importanza locale e globale

    # Parametri Dataset
    reduction_factor = 1  # Usa 1 per il dataset completo
    num_hops = 2

    print(f"--- Caricamento dataset: {dataset_name} su {device} ---")

    # --- 2. Caricamento Dati ---
    # Nota: load_dataset restituisce un oggetto HeteroData con metapath già aggiunti
    data = load_dataset(dataset_name, reduction_factor=reduction_factor, k=num_hops, device=device)
    target_type = data.target_type
    print(f"Target node type: {target_type}")

    # Dizionario dimensioni input per il PreTransformer
    in_channels_dict = {
        node_type: data[node_type].x.size(1)
        for node_type in data.node_types
    }

    # --- 3. Pre-elaborazione per AGCL ---
    # Il modello AGCL (versione GCN implementata) lavora bene su una struttura unificata.
    # Convertiamo l'HeteroData in omogeneo per estrarre l'edge_index globale che include i metapath.
    print("Conversione in grafo omogeneo per elaborazione topologica...")
    homo_data = data.to_homogeneous()
    edge_index = homo_data.edge_index.to(device)

    # Mappatura indici: Dobbiamo sapere quali indici nel grafo omogeneo corrispondono al target_type
    # data[target_type].batch o un meccanismo simile servirebbe se usassimo batch loader,
    # ma qui to_homogeneous concatena i nodi. PyG mantiene un vettore 'node_type'.

    # Troviamo l'ID numerico del tipo target nel grafo omogeneo
    target_type_id = data.node_types.index(target_type)
    # Otteniamo gli indici globali dei nodi target
    target_global_indices = torch.nonzero(homo_data.node_type == target_type_id).flatten()

    print("Calcolo H-index (Statico)...")
    # Calcoliamo l'H-index sulla matrice di adiacenza densa (attenzione alla memoria per grafi grandi)
    # Per grafi molto grandi, usare implementazione sparsa.
    adj_dense = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)).to(device),
        (homo_data.num_nodes, homo_data.num_nodes)
    ).to_dense()
    h_index = calculate_h_index(adj_dense)

    # --- 4. Inizializzazione Modello ---
    model = AGCL(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        phi=phi
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. Training Loop ---
    print("\n--- Inizio Training AGCL ---")
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # A. Forward Pass
        # Passiamo x_dict originale (eterogeneo) e edge_index globale (che include metapath)
        ci, gi, cl_loss = model(data.x_dict, edge_index, h_index)

        # B. Sampling Strategy (Sezione 4.2.1 Paper)
        # Selezioniamo un sottoinsieme di nodi target per calcolare la Ground Truth SIR
        sampled_nodes = sampling_strategy(edge_index, homo_data.num_nodes, target_global_indices, sample_size=batch_size_sampling)

        # C. Calcolo Ground Truth (SIR Simulation)
        # Nota: Eseguito solo sui nodi campionati per efficienza
        with torch.no_grad():
            ground_truth = sir_simulation(edge_index, homo_data.num_nodes, sampled_nodes, beta=0.1, gamma=0.1, steps=20)

        # D. Calcolo Loss
        # Loss 1: Regressione sull'importanza globale (predetto vs SIR)
        pred_importance = gi[sampled_nodes]
        reg_loss = F.mse_loss(pred_importance, ground_truth)

        # Loss Totale: Combinazione loss contrastiva (struttura) e regressione (influenza)
        # Il paper usa pesi di regolarizzazione, qui ipotizziamo 0.1 per il contrastive
        total_loss = reg_loss + 0.1 * cl_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | Reg Loss: {reg_loss.item():.4f} | CL Loss: {cl_loss.item():.4f}")

    # --- 6. Valutazione / Ranking Finale ---
    print("\n--- Valutazione Finale ---")
    model.eval()
    with torch.no_grad():
        final_ci, final_gi, _ = model(data.x_dict, edge_index, h_index)

        # Estraiamo solo i punteggi dei nodi target (es. attori o autori)
        target_scores = final_ci[target_global_indices]

        # Top-K Ranking
        k = 20
        topk_values, topk_indices_local = torch.topk(target_scores, k)

        # Convertiamo gli indici locali in indici globali se necessario, o mostriamo ID relativo
        print(f"\nTop-{k} Nodi Influenti ({target_type}):")
        for rank, (score, idx) in enumerate(zip(topk_values, topk_indices_local)):
            print(f"Rank {rank + 1}: Node ID {idx.item()} (Score: {score:.4f})")

    # Salvataggio risultati (opzionale)
    # output_path = os.path.join(get_base_dir(), dataset_name, "agcl_results.pt")
    # torch.save(target_scores, output_path)


if __name__ == "__main__":
    main()