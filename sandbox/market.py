import numpy as np


class SimulationResults:

    def __init__(self, times, market_swap_prices, inventories, cash, pnl, volumes, arb_volumes, proposed_swap_price_diffs, market):

        self.times = times
        self.market_swap_prices = market_swap_prices
        self.inventories = inventories
        self.cash = cash
        self.pnl = pnl
        self.market = market
        self.volumes = volumes
        self.arb_volumes = arb_volumes
        self.proposed_swap_price_diffs = proposed_swap_price_diffs


class Market:

    def __init__(self, currencies, init_swap_price_01, mu, sigma, sizes,
                 intensity_functions_01_object, intensity_functions_10_object):

        self.currencies = currencies
        self.init_swap_price_01 = init_swap_price_01
        self.mu = mu
        self.sigma = sigma
        self.sizes = sizes  # in reference currency
        self.nb_sizes = sizes.shape[0]
        self.intensities_functions_01_object = intensity_functions_01_object
        self.intensities_functions_10_object = intensity_functions_10_object
        self.intensities_functions_01 = [obj.Lambda for obj in intensity_functions_01_object]
        self.intensities_functions_10 = [obj.Lambda for obj in intensity_functions_10_object]

    def merton_jump_paths(self, S, dt, r, sigma,  lam, m, v, steps):
        size=steps
        poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                            np.random.normal(m,v, size=size)).cumsum(axis=0)
        geo = np.cumsum(((r -  (sigma**2)/2 -lam*(m  + v**2*0.5))*dt +\
                                sigma*np.sqrt(dt) * \
                                np.random.normal(size=size)), axis=0)
        
        return np.exp(geo+poi_rv)*S
        
    def simulate(self, dt, T, lp, verbose=False, process_type="merton_jump", process_parameters=()):

        nb_t = int(T / dt) + 1
        times = np.linspace(0., T, nb_t)

        market_swap_prices_01 = self._generate_price_process(process_type, dt, nb_t, times, *process_parameters)
        
        current_swap_price_01 = market_swap_prices_01[0]
        current_swap_price_10 = 1. / current_swap_price_01

        inventories = np.zeros([nb_t, 2])
        cash = np.zeros(nb_t)
        mtm_value = np.zeros(nb_t)
        mtm_value_hodl = np.zeros(nb_t)
        volumes = np.zeros(nb_t)
        arb_volumes = np.zeros(nb_t)

        inventories[0] = lp.inventories
        cash[0] = lp.cash
        mtm_value[0] = lp.mtm_value(self.init_swap_price_01)
        mtm_value_hodl[0] = lp.initial_cash + lp.initial_inventories[0] + current_swap_price_01 * \
                            lp.initial_inventories[1]

        proposed_swap_price_diffs = np.zeros(2 * nb_t)

        if verbose:
            print("C'est parti pour %d périodes\n"%nb_t)
        
        for t in range(nb_t-1):
            
            # every lp can decide to implement an off-chain bot able to trigger on-chain txs -- such as to stop operating etc. -- based on off-chain data.
            lp.bot(current_swap_price_01)

            if verbose and t % 100 == 0:
                print(t)

            current_time = times[t]
            lp.oracle.update(current_time, current_swap_price_01)
            
            # retail trading
            for index in np.random.permutation(2 * self.nb_sizes):
                side = index % 2
                size_index = index // 2
                size = self.sizes[size_index]
                if side == 0:
                    proposed_swap_price_01 = lp.proposed_swap_prices_01(size * current_swap_price_10)
                    proposed_swap_delta_01 = (proposed_swap_price_01 - current_swap_price_01) / current_swap_price_01
                    proposed_swap_price_diffs[2 * t] = proposed_swap_delta_01
                    proba_01 = 1. - np.exp(- self.intensities_functions_01[size_index](proposed_swap_delta_01) * dt)
                    if proba_01 > 0:
                        trade_01 = np.random.binomial(1, proba_01)
                        lp.update_01(trade_01)
                        if trade_01:
                            volumes[t+1] += size         
                else:
                    proposed_swap_price_10 = lp.proposed_swap_prices_10(size)
                    proposed_swap_delta_10 = (proposed_swap_price_10 - current_swap_price_10) / current_swap_price_10
                    proposed_swap_price_diffs[2 * t + 1] = proposed_swap_delta_10
                    proba_10 = 1. - np.exp(- self.intensities_functions_10[size_index](proposed_swap_delta_10) * dt)
                    if proba_10 > 0:
                        trade_10 = np.random.binomial(1, proba_10)
                        lp.update_10(trade_10)
                        if trade_10:
                            volumes[t+1] += size

            # spatial arbitrage trading
            for side in np.random.permutation([0, 1]):
                if side == 0:
                    arb_size = lp.arb_01(
                        swap_price_01=current_swap_price_01,
                        fixed_cost=0.01,
                        relative_cost=0.075 / 100
                    ) * current_swap_price_01
                else:
                    arb_size = lp.arb_10(
                        swap_price_10=current_swap_price_10,
                        fixed_cost=0.01,
                        relative_cost=0.075 / 100
                    )
                if arb_size > 0:
                    arb_volumes[t+1] += arb_size

            current_swap_price_01 = market_swap_prices_01[t + 1]
            current_swap_price_10 = 1. / current_swap_price_01

            # temporal arbitrage trading
            for side in np.random.permutation([0, 1]):
                if side == 0:
                    arb_size = lp.arb_01(
                        swap_price_01=current_swap_price_01,
                        fixed_cost=0.01,
                        relative_cost=0.075 / 100
                    ) * current_swap_price_01
                    if arb_size > 0:
                        arb_volumes[t+1] += arb_size         
                else:
                    arb_size = lp.arb_10(
                        swap_price_10=current_swap_price_10,
                        fixed_cost=0.01,
                        relative_cost=0.075 / 100
                    )
                    if arb_size > 0:
                        arb_volumes[t+1] += arb_size

            inventories[t + 1, :] = [v for v in lp.inventories]
            cash[t + 1] = lp.cash
            mtm_value[t + 1] = lp.mtm_value(current_swap_price_01)
            mtm_value_hodl[t + 1] = lp.initial_cash + lp.initial_inventories[0] + current_swap_price_01 * lp.initial_inventories[1]
    
        return SimulationResults(
            times, market_swap_prices_01, inventories, cash, (mtm_value - mtm_value_hodl) / mtm_value_hodl, volumes, arb_volumes, proposed_swap_price_diffs, self
        )

    def _generate_price_process(self, process_type, dt, nb_t, times, *args):
        if process_type == "merton_jump":
            S = self.init_swap_price_01 # current stock price
            dt = dt # time step
            r = 0.0 # risk free rate
            m = self.mu # meean of jump size
            v = 0.001 # standard deviation of jump
            lam = 1000 # intensity of jump i.e. number of jumps per annum
            steps = nb_t # time steps
            sigma = self.sigma # annaul standard deviation , for weiner process
            return self.merton_jump_paths(S, dt, r, sigma, lam, m, v, steps)
        elif process_type == "gbm":
            market_swap_prices_01 = np.zeros([nb_t])
            market_swap_prices_01[1:] = np.sqrt(dt) * np.cumsum(np.random.normal(0., self.sigma, nb_t-1))
            return self.init_swap_price_01 * np.exp((self.mu - (self.sigma ** 2) / 2.) * times + market_swap_prices_01)
        else:
            raise ValueError("unrecognized process type:", process_type)
        