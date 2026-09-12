from model import create_model

class DiabetesClient:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self, weights):
        model = create_model()
        model.set_weights(weights)

        class_weight = {0: 1, 1: 2}

        model.fit(self.X, self.y, epochs=5, batch_size=32, class_weight=class_weight)

        return model.get_weights(), len(self.X)