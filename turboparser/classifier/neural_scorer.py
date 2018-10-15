import torch
import torch.optim as optim

class NeuralScorer(object):
    def __init__(self, model=None, learning_rate=0.001):
        self.scores = None
        if model:
            self.initialize(model, learning_rate)

    def initialize(self, model, learning_rate=0.001):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def compute_scores(self, instances, parts):
        # Run the forward pass.
        if not isinstance(instances, list):
            instances = [instances]
            parts = [parts]

        self.scores = self.model(instances, parts)
        return self.scores.detach().numpy()

    def compute_gradients(self, gold_output, predicted_output):
        """
        Compute the error gradient.

        :param gold_output: either a numpy 1d array for a single item or a list
            of 1d arrays for a batch.
        :param predicted_output: same as gold_output
        """
        if isinstance(gold_output, list):
            batch_size = len(gold_output)
            max_length = max(len(g) for g in gold_output)
            shape = [batch_size, max_length]
            diff = torch.zeros(shape, dtype=self.scores.dtype)
            for i in range(batch_size):
                gold_item = gold_output[i]
                pred_item = predicted_output[i]
                diff[i, :len(gold_item)] = torch.tensor(pred_item - gold_item)
        else:
            diff = torch.tensor(predicted_output - gold_output,
                                dtype=self.scores.dtype)

        error = (self.scores * diff).sum()
        # Backpropagate to accumulate gradients.
        error.backward()

    def make_gradient_step(self):
        self.optimizer.step()
        # Clear out the gradients before the next batch.
        self.model.zero_grad()