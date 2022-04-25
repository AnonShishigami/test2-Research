import numpy as np


class Logistic:

    def __init__(self, lambda_, a, b):
        self.lambda_ = lambda_
        self.a = a
        self.b = b

    def Lambda(self, delta):
        return self.lambda_ / (1. + np.exp(self.a + self.b * delta))
