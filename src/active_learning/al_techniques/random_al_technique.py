import random

from src.active_learning.al_techniques.abstract_al_technique import AbstractALTechnique


class RandomALTechnique(AbstractALTechnique):

    def get_score(self, samples, target_type):
        return [random.uniform(0, 1) for _ in range(samples[target_type].x.shape[0])]
