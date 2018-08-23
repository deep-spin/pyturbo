import torch
import torch.optim as optim

class NeuralScorer(object):
    def __init__(self, model=None):
        self.scores = None
        if model:
            self.initialize(model)

    def initialize(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def compute_scores(self, instance, parts):
        # Run the forward pass.
        self.scores = self.model(instance, parts)
        return self.scores.detach().numpy()

    def compute_gradients(self, gold_output, predicted_output):
        # Compute error.
        error = (self.scores * torch.tensor(
            predicted_output - gold_output, dtype=self.scores.dtype)).sum()
        # Backpropagate to accumulate gradients.
        error.backward()

    def make_gradient_step(self):
        self.optimizer.step()
        # Clear out the gradients before the next batch.
        self.model.zero_grad()
