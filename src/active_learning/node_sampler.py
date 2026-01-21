import torch

from src.support.utils_graph import k_hop_subgraph


class ActiveLearningSampler:

    def __init__(self, al_technique, k):
        self.al_technique = al_technique
        self.k = k

    def sample(self, data, train_mask_labeled):
        train_mask_unlabeled = 1 - train_mask_labeled
        train_mask_unlabeled = train_mask_unlabeled.squeeze()

        node_scores = []
        node_subgraph = k_hop_subgraph(data, torch.nonzero(train_mask_unlabeled).flatten(), 2)[0].to(data.device)
        scores = self.al_technique.get_score(node_subgraph, data.target_type)
        for idx, score in enumerate(scores):
            node_scores.append((train_mask_unlabeled[idx].item(), score))

        # sorting nodes keeping index and related score
        excluded_indices = torch.nonzero(train_mask_labeled).flatten().tolist()
        all_sorted_nodes = sorted(range(len(node_scores)), key=lambda j: node_scores[j][1], reverse=True)
        selected_nodes = [idx for idx in all_sorted_nodes if idx not in excluded_indices][:self.k]
        train_mask_labeled[selected_nodes] = 1

        return train_mask_labeled
