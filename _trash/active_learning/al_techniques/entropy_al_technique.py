import torch
import numpy

from _trash.active_learning.al_techniques.abstract_al_technique import AbstractALTechnique


class EntropyALTechnique(AbstractALTechnique):

    def get_score(self, samples, target_type):
        preds = torch.nn.functional.softmax(self.model(samples.x_dict, samples.edge_index_dict)[0][target_type], dim=1).detach().cpu().numpy()
        return (numpy.log(preds + 1e-6) * preds).sum(axis=1)
