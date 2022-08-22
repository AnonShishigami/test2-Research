import numpy as np
from .demand_curve import Logistic


class LogisticTools(Logistic):

    def __init__(self, lambda_, a, b):
        super().__init__(lambda_, a, b)

    def __Lambda_inverse(self, intensity):
        return (np.log(
            self.lambda_ / np.minimum(np.maximum(intensity, 1e-10), self.lambda_ - 1e-10) - 1.) - self.a) / self.b

    def H(self, p):
        p = np.minimum(np.maximum(p, -50. / self.b), 50. / self.b)
        delta = p
        exp_term = np.exp(-self.a - self.b * delta)
        obj = self.b * delta - exp_term - (1. + self.b * p)
        while np.any(np.abs(obj) > self.b * 1e-6):
            delta = delta - obj / self.b / (1. + exp_term)
            exp_term = np.exp(-self.a - self.b * delta)
            obj = self.b * delta - exp_term - (1. + self.b * p)
        return self.lambda_ * (delta - p - 1. / self.b)

    def H_prime(self, p):
        p = np.minimum(np.maximum(p, -50. / self.b), 50. / self.b)
        delta = p
        exp_term = np.exp(-self.a - self.b * delta)
        obj = self.b * delta - exp_term - (1. + self.b * p)
        while np.any(np.abs(obj) > self.b * 1e-6):
            delta = delta - obj / self.b / (1. + exp_term)
            exp_term = np.exp(-self.a - self.b * delta)
            obj = self.b * delta - exp_term - (1. + self.b * p)
        return - self.lambda_ * (1. - 1. / (1. + exp_term))

    def H_second(self, p):
        p = np.minimum(np.maximum(p, -50. / self.b), 50. / self.b)
        delta = p
        exp_term = np.exp(-self.a - self.b * delta)
        obj = self.b * delta - exp_term - (1 + self.b * p)
        while np.any(np.abs(obj) > self.b * 1e-6):
            delta = delta - obj / self.b / (1 + exp_term)
            exp_term = np.exp(-self.a - self.b * delta)
            obj = self.b * delta - exp_term - (1 + self.b * p)
        return self.lambda_ * (1. - 1. / (1. + exp_term)) / (self.b * (delta - p) ** 2)

    def delta(self, p):
        p = np.minimum(np.maximum(p, -50. / self.b), 50. / self.b)
        return self.__Lambda_inverse(-self.H_prime(p))