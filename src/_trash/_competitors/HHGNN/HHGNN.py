import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import dense_to_sparse


# Impostiamo il device (GPU se disponibile, altrimenti CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch


def heterodata_to_global_adj(data, target_type):
    """
    Converte un oggetto HeteroData (PyG) in una matrice di adiacenza globale densa
    e restituisce gli indici globali dei nodi target.
    Necessario per calcolare le potenze della matrice A^K in HHGNN.
    """
    node_types = data.metadata()[0]
    edge_types = data.metadata()[1]

    # 1. Calcola gli offset per mappare ogni tipo di nodo a indici globali unici (0...N)
    node_offsets = {}
    current_offset = 0
    total_nodes = 0

    # Ordine deterministico basato sui metadati
    for nt in node_types:
        num = data[nt].num_nodes
        node_offsets[nt] = current_offset
        current_offset += num
        total_nodes += num

    # 2. Costruisci le liste di adiacenza globali
    row_list, col_list = [], []

    for src_type, rel, dst_type in edge_types:
        # Prendi gli archi per questo tipo di relazione
        edge_index = data[(src_type, rel, dst_type)].edge_index

        # Aggiungi gli offset per trasformare indici locali in globali
        src_global_idx = edge_index[0] + node_offsets[src_type]
        dst_global_idx = edge_index[1] + node_offsets[dst_type]

        row_list.append(src_global_idx)
        col_list.append(dst_global_idx)

        # Rendiamo la matrice simmetrica (grafo non orientato) se src != dst
        # Questo è importante per la stabilità spettrale citata nel paper
        if src_type != dst_type:
            row_list.append(dst_global_idx)
            col_list.append(src_global_idx)

    # Concatena tutti gli archi
    if len(row_list) > 0:
        all_rows = torch.cat(row_list)
        all_cols = torch.cat(col_list)
    else:
        all_rows = torch.tensor([], dtype=torch.long)
        all_cols = torch.tensor([], dtype=torch.long)

    # 3. Crea la matrice densa
    # Nota: Se il grafo è enorme (>20k nodi), questa operazione potrebbe usare molta RAM.
    # In quel caso bisognerebbe usare sparse_coo_tensor, ma HHGNN richiede mm() denso per A^K.
    adj = torch.zeros((total_nodes, total_nodes))
    adj[all_rows, all_cols] = 1.0

    # 4. Recupera gli indici globali dei nodi target (es. 'user' o 'author')
    target_start = node_offsets[target_type]
    target_count = data[target_type].num_nodes
    target_global_indices = torch.arange(target_start, target_start + target_count)

    return adj, target_global_indices


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

def get_mo_cam(global_adj, target_indices, K, r_threshold=0.01, device='cpu'):
    # ... (codice iniziale di normalizzazione uguale a prima) ...
    global_adj = global_adj.to(device)
    deg = global_adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L_sys = torch.mm(torch.mm(D_inv_sqrt, global_adj), D_inv_sqrt)

    mo_cam_layers = []
    current_adj = L_sys.clone()

    for k in range(1, K + 1):
        if k > 1:
            current_adj = torch.mm(current_adj, L_sys)

        sub_adj = current_adj[target_indices][:, target_indices]

        # Normalizzazione locale
        min_val = sub_adj.min()
        max_val = sub_adj.max()
        if max_val - min_val > 1e-9:
            sub_adj = (sub_adj - min_val) / (max_val - min_val)

        # Sparsification
        sub_adj = torch.where(sub_adj < r_threshold, torch.tensor(0.0).to(device), sub_adj)

        # --- PARTE CRITICA MODIFICATA ---
        # Creiamo manualmente gli indici invece di usare dense_to_sparse
        indices = sub_adj.nonzero(as_tuple=False).t()
        values = sub_adj[indices[0], indices[1]]

        # Restituiamo 3 valori: indici, pesi, e numero di nodi (shape)
        num_nodes_in_layer = sub_adj.shape[0]
        mo_cam_layers.append((indices, values, num_nodes_in_layer))
        # --------------------------------

    return mo_cam_layers


class HHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, K, T_scale=1.0):
        """
        Args:
            in_dim: Dimensione feature input
            hidden_dim: Dimensione embedding nascosto
            K: Numero massimo di ordini (lunghezza path contestuale)
            T_scale: Fattore di temperatura per l'attenzione (Eq. 7) [cite: 286]
        """
        super(HHGNN, self).__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.T = T_scale

        # Encoder iniziale delle feature [cite: 290]
        self.feat_encoder = nn.Linear(in_dim, hidden_dim)

        # Lista di Layer KAN, uno per ogni ordine k [cite: 284]
        # "We maintain a learnable weight matrix for each order"
        self.kan_layers = nn.ModuleList([
            KANLinear(hidden_dim, hidden_dim) for _ in range(K)
        ])

        # Matrici per l'Attenzione (Query, Key, Value) [cite: 301]
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Normalizzazione e Attivazione per f(Message) (Eq. 9) [cite: 304, 310]
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU()

        # Layer finale per il ranking (proiezione a scalare)
        # Necessario per ottenere il "Final rank" mostrato in Fig 1 [cite: 181]
        self.score_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x, mo_cam_layers):
        """
        Args:
            x: Tensor [Num_Nodes, In_Dim]
            mo_cam_layers: Lista di (indices, values, num_nodes) restituita da get_mo_cam
        """
        # 1. Encoding Iniziale
        h = self.feat_encoder(x)
        num_nodes = h.size(0)

        # Liste per accumulare i vettori Key e Value per ogni ordine k
        keys_list = []
        values_list = []

        # Query vector: dipende dal nodo target corrente (Eq. 8) [cite: 295]
        query = self.W_q(h)

        # 2. Cross-Order Message Passing [cite: 318-321]
        for k in range(self.K):
            # Estrai dati sparsi per l'ordine k
            edge_index, edge_weight, _ = mo_cam_layers[k]

            # --- Propagazione (Eq. 11: propagate) ---
            if edge_index.size(1) > 0:
                # Costruisci tensore sparso per moltiplicazione efficiente
                adj_sparse = torch.sparse_coo_tensor(
                    edge_index, edge_weight, (num_nodes, num_nodes)
                ).to(x.device)

                # Aggregazione delle feature dei vicini (weighted sum)
                propagated = torch.sparse.mm(adj_sparse, h)
            else:
                propagated = torch.zeros_like(h)

            # --- Trasformazione KAN (Eq. 11 e 12) ---
            # Message(h_s) = Linear(propagate) + Linear(self)
            # Applicazione del layer KAN specifico per l'ordine k
            msg = self.kan_layers[k](propagated) + self.kan_layers[k](h)

            # --- Attivazione e Normalizzazione (Eq. 9) ---
            # f(x) = g(g(x)) con LayerNorm e LeakyReLU
            msg_transformed = self.activation(self.layer_norm(msg))

            # --- Calcolo Key e Value (Eq. 8 e 295) ---
            k_vect = self.W_k(msg_transformed)
            v_vect = self.W_v(msg_transformed)

            keys_list.append(k_vect.unsqueeze(0))  # [1, N, Hidden]
            values_list.append(v_vect.unsqueeze(0))  # [1, N, Hidden]

        # Concatenazione lungo la dimensione degli ordini (dim 0)
        K_stack = torch.cat(keys_list, dim=0)  # [K, N, Hidden]
        V_stack = torch.cat(values_list, dim=0)  # [K, N, Hidden]

        # 3. Edge-level (Order-level) Attention Mechanism [cite: 294]
        # Attention(t) = Softmax( (Query * Key) / T )

        # Espandiamo la query per confrontarla con ogni ordine K
        query_expanded = query.unsqueeze(0).expand(self.K, -1, -1)  # [K, N, Hidden]

        # Dot product
        attn_logits = torch.sum(query_expanded * K_stack, dim=-1)  # [K, N]

        # Scaling con temperatura T [cite: 286, 299]
        attn_logits = attn_logits / self.T

        # Softmax lungo la dimensione degli ordini (dim 0)
        # Determina quanto ogni ordine k è importante per il nodo i
        attn_weights = F.softmax(attn_logits, dim=0).unsqueeze(-1)  # [K, N, 1]

        # 4. Aggregazione Finale (Eq. 10) [cite: 315]
        # Weighted sum dei Values
        weighted_values = attn_weights * V_stack  # [K, N, Hidden]
        final_embedding = torch.sum(weighted_values, dim=0)  # [N, Hidden]

        # 5. Predizione Score (Ranking)
        scores = self.score_predictor(final_embedding)  # [N, 1]

        return scores