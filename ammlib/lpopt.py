import numpy as np
from scipy.interpolate import interp2d
from .liquidityprovider import BaseLiquidityProvider
from .logistic import Logistic
from .logistictools import LogisticTools


class NumericalParams:

    def __init__(self, T, nb_t, Vs_min, Vs_max):
        self.T= T
        self.nb_t = nb_t
        self.Vs_min = Vs_min
        self.Vs_max = Vs_max
        if Vs_min[0] < 0 or Vs_min[1] < 0:
            raise ValueError('Stay positive please')


class LiquidityProviderHJB(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, gamma, num_params):

        super().__init__(name, initial_inventories, initial_cash, market)
