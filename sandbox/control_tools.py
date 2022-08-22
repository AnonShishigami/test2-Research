import numpy as np
from .demand_curve import Logistic, MixedLogistics


class LogisticExtended(Logistic):

    def __init__(self, lambda_, a, b):
        super().__init__(lambda_, a, b)
        self.delta0 = self.delta(0.)
        self.H_prime0 = -self.Lambda(self.delta0)
        self.H_second0 = self.lambda_ * self.b * np.exp(self.a + self.b * self.delta0)**2 / (1.+np.exp(self.a + self.b * self.delta0))**3

    def delta(self, p):
        p = np.minimum(np.maximum(p, -50. / self.b), 50. /self.b)
        delta = p
        exp_term = np.exp(-self.a - self.b * delta)
        obj = self.b * delta - exp_term - (1. + self.b * p)
        while np.any(np.abs(obj) > self.b * 1e-6):
            delta = delta - obj / self.b / (1. + exp_term)
            exp_term = np.exp(-self.a - self.b * delta)
            obj = self.b * delta - exp_term - (1. + self.b * p)
        return delta


class MixedLogisticsExtended(MixedLogistics):

    def __init__(self, logistic_1, logistic_2):
        super().__init__(logistic_1, logistic_2)
        self.ext_logistic_1 = LogisticExtended(logistic_1.lambda_, logistic_1.a, logistic_1.b)
        self.ext_logistic_2 = LogisticExtended(logistic_2.lambda_, logistic_2.a, logistic_2.b)
        self.delta0 = self.delta(0.)
        self.H_prime0 = -self.Lambda(self.delta0)
        lprime = self.__Lambda_prime(self.delta0)
        self.H_second0 = - lprime**3 / (2.* lprime**2 - self.Lambda(self.delta0) * self.__Lambda_second(self.delta0))

    def __Lambda_prime(self, delta):
        sigma_term_1 = 1. / (1. + np.exp(self.logistic_1.a + self.logistic_1.b * delta))
        sigma_term_2 = 1. / (1. + np.exp(self.logistic_2.a + self.logistic_2.b * delta))
        return - self.logistic_1.lambda_ * self.logistic_1.b * sigma_term_1 * (1. - sigma_term_1)\
               - self.logistic_2.lambda_ * self.logistic_2.b * sigma_term_2 * (1. - sigma_term_2)

    def __Lambda_second(self, delta):
        sigma_term_1 = 1. / (1. + np.exp(self.logistic_1.a + self.logistic_1.b * delta))
        sigma_term_2 = 1. / (1. + np.exp(self.logistic_2.a + self.logistic_2.b * delta))
        return self.logistic_1.lambda_ * self.logistic_1.b**2 * sigma_term_1 * (1. - sigma_term_1) * (1. - 2. * sigma_term_1)\
               + self.logistic_2.lambda_ * self.logistic_2.b**2 * sigma_term_2 * (1. - sigma_term_2) * (1. - 2. * sigma_term_2)

    def delta(self, p):
        delta_1 = self.ext_logistic_1.delta(p)
        delta_2 = self.ext_logistic_2.delta(p)
        deltas = np.linspace(np.minimum(delta_1, delta_2), np.maximum(delta_1, delta_2), 1000)
        return deltas[np.argmax(deltas * self.Lambda(deltas))]