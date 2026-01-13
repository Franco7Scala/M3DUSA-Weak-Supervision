import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import numpy as np


class PreTransformer(nn.Module):
    """
    Modulo di pre-elaborazione per reti eterogenee.
    Proietta le feature di nodi di tipo diverso in uno spazio vettoriale comune.
    Rif: Eq. (1) del paper.
    """

    def __init__(self, in_channels_dict, out_channels):
        super(PreTransformer, self).__init__()
        self.projections = nn.ModuleDict()
        # Crea un layer lineare per ogni tipo di nodo (es. 'author', 'paper')
        for node_type, in_dim in in_channels_dict.items():
            self.projections[node_type] = nn.Linear(in_dim, out_channels)

    def forward(self, x_dict):
        out_dict = {}
        for node_type, x in x_dict.items():
            # Applicazione della proiezione e funzione di attivazione
            out_dict[node_type] = torch.relu(self.projections[node_type](x))
        return out_dict


class HGCL_Encoder(nn.Module):
    """
    Encoder GCN base utilizzato all'interno di HGCL.
    Rif: Eq. (3) e Eq. (7).
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(HGCL_Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Layer fully connected finale per ottenere l'embedding
        self.fc = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        # Eq. (4) e Eq. (8)
        return torch.relu(self.fc(x))


class HGCL(nn.Module):
    """
    Modulo Heterogeneous Graph Contrastive Learning.
    Gestisce la creazione delle due viste (Neighbor e Cross-domain) e l'apprendimento contrastivo.
    Rif: Sezione 4.1.
    """

    def __init__(self, hidden_channels, masking_rate=0.1):
        super(HGCL, self).__init__()
        self.encoder = HGCL_Encoder(hidden_channels, hidden_channels)
        self.masking_rate = masking_rate

    def random_masking(self, x):
        """
        Neighbor View: Mascheramento casuale delle feature.
        Rif: Eq. (2).
        """
        if self.training:
            mask = torch.rand(x.size(), device=x.device) > self.masking_rate
            return x * mask.float()
        return x

    def graph_diffusion(self, edge_index, num_nodes):
        """
        Cross-domain View: Simulazione della diffusione del grafo (Heat Kernel/PPR).
        Rif: Eq. (5) e (6).
        Nota: Implementazione semplificata che aggiunge connessioni a k-hop per simulare la diffusione.
        """
        # Convertiamo in denso per calcolare A^2 + A + I (diffusione approssimata)
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        # Diffusione semplice: connessioni dirette + connessioni a 2 salti + self-loops
        adj_diff = torch.matmul(adj, adj) + adj + torch.eye(num_nodes, device=adj.device)
        # Binarizzazione per mantenere la struttura del grafo
        adj_diff[adj_diff > 0] = 1

        edge_index_diff, _ = dense_to_sparse(adj_diff)
        return edge_index_diff

    def contrastive_loss(self, z1, z2):
        """
        Calcola la loss contrastiva (InfoNCE o simile) tra le due viste.
        Rif: Eq. (11).
        """
        # Calcoliamo la similarità del coseno tra le rappresentazioni dello stesso nodo
        sim = F.cosine_similarity(z1, z2, dim=1)
        # Loss log-sigmoid negativa (versione semplificata della loss nel paper)
        loss = -torch.log(torch.sigmoid(sim)).mean()
        return loss

    def forward(self, x, edge_index):
        # 1. Neighbor View (Feature Masking)
        x_masked = self.random_masking(x)
        z_neighbor = self.encoder(x_masked, edge_index)

        # 2. Cross-domain View (Graph Diffusion)
        # Nota: La diffusione può essere costosa su grafi enormi, qui viene calcolata on-the-fly
        edge_index_diff = self.graph_diffusion(edge_index, x.size(0))
        z_cross = self.encoder(x, edge_index_diff)

        return z_neighbor, z_cross


class AIE(nn.Module):
    """
    Attention-based Node Importance Evaluation.
    Calcola l'importanza Globale e Locale usando un meccanismo di attenzione condiviso.
    Rif: Sezione 4.2.
    """

    def __init__(self, hidden_channels, phi=0.5):
        super(AIE, self).__init__()
        self.phi = phi  # Parametro di combinazione (phi), Eq. (20)

        # Matrici per l'Attention Mechanism Eq. (13)
        self.W_Q = nn.Linear(hidden_channels, hidden_channels)
        self.W_K = nn.Linear(hidden_channels, hidden_channels)
        self.W_U = nn.Linear(hidden_channels, hidden_channels)

        # Layer per il calcolo finale dell'importanza globale Eq. (16)
        self.fc_global = nn.Linear(hidden_channels, 1)

    def forward(self, z, h_index):
        """
        Input:
            z: Embedding dei nodi (output di HGCL)
            h_index: Tensore con i valori H-index per ogni nodo
        Output:
            CI: Combined Importance (Importanza finale del nodo)
            GI: Global Importance (usato per il training con SIR)
        """
        # --- Calcolo Attenzione (Eq. 13 & 14) ---
        Q = self.W_Q(z)
        K = self.W_K(z)
        U = self.W_U(z)

        # Attenzione scalata (dot product)
        # Nota: Per dataset grandi, questa matrice N*N potrebbe richiedere batching.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(z.size(1))
        alpha = F.softmax(scores, dim=-1)  # Coefficienti a_ij

        # --- Modulo Importanza Globale (Eq. 15 & 16) ---
        z_prime = torch.matmul(alpha, U)  # Aggregazione pesata embedding
        GI = torch.sigmoid(self.fc_global(z_prime)).squeeze()

        # --- Modulo Importanza Locale (Eq. 19) ---
        # Usa gli stessi coefficienti alpha per pesare l'H-index dei vicini
        weighted_h = torch.matmul(alpha, h_index.float().unsqueeze(-1)).squeeze()
        LI = h_index + weighted_h

        # Normalizzazione Min-Max di LI per renderlo comparabile a GI (0-1)
        if LI.max() > LI.min():
            LI = (LI - LI.min()) / (LI.max() - LI.min())
        else:
            LI = torch.zeros_like(LI)

        # --- Importanza Finale (Eq. 20) ---
        CI = self.phi * LI + (1 - self.phi) * GI

        return CI, GI


class AGCL(nn.Module):
    """
    Modello completo AGCL che unisce HGCL e AIE.
    """

    def __init__(self, in_channels_dict, hidden_channels, phi=0.5, masking_rate=0.1):
        super(AGCL, self).__init__()
        # 1. Pre-transformer per gestire feature eterogenee
        self.pre_transform = PreTransformer(in_channels_dict, hidden_channels)

        # 2. Modulo Contrastivo
        self.hgcl = HGCL(hidden_channels, masking_rate)

        # 3. Modulo di Valutazione Importanza
        self.aie = AIE(hidden_channels, phi)

    def forward(self, x_dict, edge_index, h_index, node_type_order=None):
        """
        Forward pass completo.
        Args:
            x_dict: Dizionario {node_type: features}
            edge_index: Edge index globale (o omogeneizzato)
            h_index: H-index pre-calcolato per i nodi
            node_type_order: Lista ordinata dei tipi di nodo per concatenare le feature (opzionale)
        """
        # A. Proiezione feature nello stesso spazio
        x_proj_dict = self.pre_transform(x_dict)

        # Concatenazione feature per elaborazione GCN (ordine deterministico)
        if node_type_order is None:
            node_type_order = sorted(x_proj_dict.keys())
        x_all = torch.cat([x_proj_dict[nt] for nt in node_type_order], dim=0)

        # B. HGCL: Ottieni embeddings e calcola contrastive loss
        z_neighbor, z_cross = self.hgcl(x_all, edge_index)
        cl_loss = self.hgcl.contrastive_loss(z_neighbor, z_cross)

        # Fusione delle viste (Eq. 10 nel paper usa mean pooling)
        z_final = (z_neighbor + z_cross) / 2

        # C. AIE: Calcola importanza
        ci_score, gi_score = self.aie(z_final, h_index)

        return ci_score, gi_score, cl_loss


# --- Funzioni di Utilità (H-index e SIR) ---

def calculate_h_index(adj_matrix):
    """
    Calcola l'H-index per ogni nodo basandosi sulla matrice di adiacenza densa.
    Rif: Eq. (18) HD_i = max(min(c(i), i)).
    """
    device = adj_matrix.device
    num_nodes = adj_matrix.shape[0]
    h_indices = torch.zeros(num_nodes, device=device)

    # Gradi dei nodi
    degrees = torch.sum(adj_matrix, dim=1)

    for i in range(num_nodes):
        # Trova vicini
        neighbors_idx = torch.nonzero(adj_matrix[i]).flatten()
        if len(neighbors_idx) == 0:
            continue

        # Ottieni i gradi dei vicini
        neighbor_degrees = degrees[neighbors_idx]

        # Ordina in modo decrescente
        sorted_degrees, _ = torch.sort(neighbor_degrees, descending=True)

        # Calcolo H-index
        h = 0
        for idx, deg in enumerate(sorted_degrees):
            if deg >= idx + 1:
                h = idx + 1
            else:
                break
        h_indices[i] = h

    return h_indices


def sir_simulation(edge_index, num_nodes, seeds, beta=0.1, gamma=0.1, steps=50):
    """
    Simulazione modello SIR per generare le etichette di training (Ground Truth).
    Rif: Eq. (12) e Sezione 4.2.1.
    Args:
        edge_index: Connessioni del grafo
        num_nodes: Numero totale nodi
        seeds: Indici dei nodi target (sampled nodes) per cui calcolare l'importanza
    Returns:
        labels: Tensore con il range di propagazione massimo normalizzato per ogni seed.
    """
    device = edge_index.device
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    labels = torch.zeros(len(seeds), device=device)

    for idx, seed_node in enumerate(seeds):
        # Stato: 0 = Susceptible, 1 = Infected, 2 = Recovered
        status = torch.zeros(num_nodes, device=device)
        status[seed_node] = 1  # Inizia infetto

        for _ in range(steps):
            infected_mask = (status == 1)
            if not infected_mask.any():
                break

            # Calcolo probabilità infezione
            potential_hosts = torch.matmul(adj, infected_mask.float())
            infection_prob = 1 - (1 - beta) ** potential_hosts

            # Nuovi infetti (da S a I)
            new_infected = (torch.rand(num_nodes, device=device) < infection_prob) & (status == 0)

            # Nuovi guariti (da I a R)
            new_recovered = (torch.rand(num_nodes, device=device) < gamma) & (status == 1)

            status[new_infected] = 1
            status[new_recovered] = 2

        # L'importanza è il numero totale di nodi che sono stati infettati o guariti
        total_affected = torch.sum(status != 0)
        labels[idx] = total_affected

    # Normalizzazione etichette tra 0 e 1
    if labels.max() > 0:
        labels = labels / labels.max()

    return labels
