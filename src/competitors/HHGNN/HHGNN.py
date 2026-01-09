import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Impostiamo il device (GPU se disponibile, altrimenti CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. KAN Linear Layer (Basato su B-Spline)
# Riferimento: Sezione 4.3, Eq. 12
# =============================================================================

class KANLinear(nn.Module):
    """
    Implementazione del Linear Layer potenziato.
    Formula: linear(x) = sigma(W_m * x + b) + W_s * bspline(x)
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Componente 1: MLP Standard (W_m * x + b)
        self.base_linear = nn.Linear(in_features, out_features)

        # Componente 2: B-Spline parametrizzata
        # W_s: peso per la componente spline
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        nn.init.xavier_uniform_(self.spline_weight)

        # Griglia fissa per la base delle funzioni
        self.grid_size = grid_size
        # Usiamo linspace per creare i centri delle funzioni di base
        self.register_buffer("grid", torch.linspace(-1, 1, grid_size + 1))

    def b_spline_basis(self, x):
        """
        Calcola la base delle funzioni per l'input x.
        Nota: Per mantenere il codice compatto in un singolo file, usiamo
        una base di Funzioni a Base Radiale (RBF) che approssima
        il comportamento locale delle B-Spline.
        """
        x_expanded = x.unsqueeze(-1)  # [Batch, In, 1]
        grid = self.grid.view(1, 1, -1)  # [1, 1, Grid]

        # Approssimazione tramite Gaussiane (comportamento simile a spline locali)
        sigma = 2.0 / self.grid_size
        basis = torch.exp(-((x_expanded - grid) ** 2) / (2 * sigma ** 2))

        # Padding se necessario per matchare la dimensione dei pesi (spline_order)
        target_dim = self.spline_weight.shape[-1]
        if basis.shape[-1] < target_dim:
            padding = target_dim - basis.shape[-1]
            basis = F.pad(basis, (0, padding))
        elif basis.shape[-1] > target_dim:
            basis = basis[..., :target_dim]

        return basis

    def forward(self, x):
        # Parte 1: MLP Standard con attivazione (Eq. 12 usa sigma, qui SiLU)
        base_out = F.silu(self.base_linear(x))

        # Parte 2: B-Spline component
        spline_basis = self.b_spline_basis(x)  # [Batch, In, Grid]

        # Calcolo tensoriale: W_s * bspline(x)
        # Einstein Summation: b=batch, i=input_dim, o=out_dim, g=grid_dim
        spline_out = torch.einsum('big,oig->bo', spline_basis, self.spline_weight)

        return base_out + spline_out


# =============================================================================
# 2. Generazione MO-CAM (Homogeneous Subgraph Extraction)
# Riferimento: Sezione 4.2, Algoritmo 1, Eq. 1-5
# =============================================================================

def get_mo_cam(adj_matrix, target_indices, K, r_threshold=0.01):
    """
    Costruisce la Multi-Order Contextual Adjacency Matrix.
    Simula l'estrazione dei sottografi omogenei dal grafo eterogeneo.
    """
    # Assicuriamoci che adj sia su device
    adj_matrix = adj_matrix.to(device)
    num_nodes = adj_matrix.shape[0]

    # --- Normalizzazione Globale (Laplacian) Eq. 3, 4 ---
    # L_sys = D^(-1/2) * A * D^(-1/2)
    deg = adj_matrix.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    L_sys = torch.mm(torch.mm(D_inv_sqrt, adj_matrix), D_inv_sqrt)

    mo_cam_layers = []
    current_adj = L_sys.clone()

    print(f"Generazione MO-CAM per K={K} ordini...")

    for k in range(1, K + 1):
        # Potenza della matrice per catturare hop k-esimi
        if k > 1:
            current_adj = torch.mm(current_adj, L_sys)

        # --- Estrazione Sottografo Omogeneo Eq. 1 ---
        # Selezioniamo solo le interazioni tra nodi target
        # Slicing: righe target, colonne target
        sub_adj = current_adj[target_indices][:, target_indices]

        # --- Normalizzazione Locale (Min-Max) Eq. 5 ---
        min_val = sub_adj.min()
        max_val = sub_adj.max()
        if max_val - min_val > 1e-9:
            sub_adj = (sub_adj - min_val) / (max_val - min_val)

        # --- Sparsification Eq. 2 ---
        # Rimuoviamo rumore (valori bassi)
        sub_adj = torch.where(sub_adj < r_threshold, torch.tensor(0.0).to(device), sub_adj