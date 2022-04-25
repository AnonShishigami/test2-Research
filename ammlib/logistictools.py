import numpy as np
from scipy.special import lambertw
from .logistic import Logistic


class LogisticTools(Logistic):

    def __init__(self, lambda_, a, b):
        super().__init__(lambda_, a, b)
        self.myopic_quote = self.__myopic_quote()

    def __myopic_quote(self):
        delta = 0.
        exp_term = np.exp(-self.a - self.b * delta)
        obj = self.b * delta - exp_term - 1.
        while np.abs(obj) > self.b * 1e-8:
            delta = delta - obj / self.b / (1. + exp_term)
            exp_term = np.exp(-self.a - self.b * delta)
            obj = self.b * delta - exp_term - 1.
        return delta