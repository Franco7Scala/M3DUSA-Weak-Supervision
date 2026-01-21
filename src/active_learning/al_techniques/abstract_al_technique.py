

class AbstractALTechnique:

    def __init__(self, model):
        self.model = model

    def get_score(self, samples, target_type):
        pass
