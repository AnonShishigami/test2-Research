import numpy as np


class Logistic:

    def __init__(self, lambda_, a, b):
        self.lambda_ = lambda_
        self.a = a
        self.b = b

    def Lambda(self, delta):
        return self.lambda_ / (1. + np.exp(self.a + self.b * delta))


class MixedLogistics:

    def __init__(self, logistic_1, logistic_2):
        self.logistic_1 = logistic_1
        self.logistic_2 = logistic_2

    def Lambda(self, delta):
        return self.logistic_1.Lambda(delta) + self.logistic_2.Lambda(delta)