import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


class NeuralScorer(object):
    def __init__(self):
        self.part_scores = None
        self.model = None

    def initialize(self, model, learning_rate=0.001, decay=1,
                   beta1=0.9, beta2=0.999):
        self.set_model(model)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params, lr=learning_rate, betas=(beta1, beta2))
        self.scheduler = scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=decay, patience=0, verbose=True)

    def set_model(self, model):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

    def compute_scores(self, instances, parts):
        # Run the forward pass.
        if not isinstance(instances, list):
            instances = [instances]
            parts = [parts]

        self.part_scores = self.model(instances, parts)
        return self.part_scores.detach().numpy()

    def train_mode(self):
        """
        Set the neural model to training mode
        """
        self.model.train()

    def eval_mode(self):
        """
        Set the neural model to eval mode
        """
        self.model.eval()

    def lr_scheduler_step(self, accuracy):
        """
        Perform a step of the learning rate scheduler, based on accuracy.
        """
        self.scheduler.step(accuracy)

    def compute_gradients(self, gold_parts, predicted_parts, gold_labels):
        """
        Compute the error gradient.

        :param gold_parts: either a numpy 1d array for a single item or a list
            of 1d arrays for a batch.
        :param predicted_parts: same as gold_output
        :param gold_labels: labels for additional targets, if used
        """
        if isinstance(gold_parts, list):
            batch_size = len(gold_parts)
            max_length = max(len(g) for g in gold_parts)
            shape = [batch_size, max_length]
            diff = torch.zeros(shape, dtype=torch.float)
            for i in range(batch_size):
                gold_item = gold_parts[i]
                pred_item = predicted_parts[i]
                diff[i, :len(gold_item)] = torch.tensor(pred_item - gold_item)
        else:
            diff = torch.tensor(predicted_parts - gold_parts,
                                dtype=torch.float)

        error = (self.part_scores * diff).sum()
        # Backpropagate to accumulate gradients.
        error.backward()

    def make_gradient_step(self):
        self.optimizer.step()
        # Clear out the gradients before the next batch.
        self.model.zero_grad()
