import numpy as np
from optimizer import *


class FedAvg(FederatedOptimizer):
    def __init__(self, lr, bs, le, seltype, powd, train_data_dir, test_data_dir, sample_ratio):
        super(FedAvg, self).__init__(lr, bs, le, seltype, powd, train_data_dir, test_data_dir, sample_ratio)

    def compute_gradient(self, x, i):
        return self.compute_gradient_template(x, i)

    def local_update(self, local_losses):
        worker_set = self.select_client(local_losses)

        Delta = list()
        weight = 1 / self.sample_ratio

        for i in worker_set:
            lr = self.lr
            local_parameters = self.central_parameter + 0
            for t in range(self.le):
                scale = 1
                local_parameters -= lr * self.compute_gradient(local_parameters, i) * scale
            Delta.append((local_parameters - self.central_parameter) * weight)

        Delta = np.array(Delta)

        return Delta, worker_set

    def aggregate(self, Delta):
        self.central_parameter += np.sum(Delta, axis=0)
