import numpy as np


class StrategyTestTemplate:
    
    def setUp(self):
        self.test_exchange_rtol = 1e-10  # equality between expected and actual values is required when set to None

    # should be overloaded
    @staticmethod
    def _get_strategy(self, event):
        return {}

    @staticmethod
    def _get_scenarios():
        return []

    def _get_exchange_args(self, event):
        pass
    
    def _generate_state_getter(self):
        pass
        
    def _check_state(self, groundtruth, actual):
        for key in groundtruth:
            if self.test_exchange_rtol:
                np.testing.assert_allclose(np.array(groundtruth[key], dtype=float), np.array(actual[key], dtype=float), rtol=self.test_exchange_rtol)
            else:
                np.testing.assert_array_equal(groundtruth[key], actual[key])

    def  _test_exchange(self, expected_price, buy_idx, buy_amount, time):
        
        if buy_idx == 1:
            pricing_function = self.strategy.proposed_swap_prices_01
            update_function = self.strategy.update_01
            price_getter = lambda: self.strategy.last_answer_01
        else:
            pricing_function = self.strategy.proposed_swap_prices_10
            update_function = self.strategy.update_10
            price_getter = lambda: self.strategy.last_answer_10
        
        # save state before executing tx
        state = self.strategy.get_state()

        # execute trade
        pricing_function(time, buy_amount)
        update_function(1)
        actual_price = price_getter()
        if self.test_exchange_rtol:
            np.testing.assert_allclose(float(expected_price), float(actual_price), rtol=self.test_exchange_rtol)
        else:
            np.testing.assert_array_equal(expected_price, actual_price)
        
        # restore state then check that result is unchanged
        self.strategy.restore_state(state)
        pricing_function(time, buy_amount)
        update_function(1)
        actual_price = price_getter()
        if self.test_exchange_rtol:
            np.testing.assert_allclose(float(expected_price), float(actual_price), rtol=self.test_exchange_rtol)
        else:
            np.testing.assert_array_equal(expected_price, actual_price)
        

    def test_scenarios(self):
        for scenario in self._get_scenarios():
            self.strategy = self._get_strategy(scenario["init_state"])
            for event in scenario.get("events", []):
                
                exchange_args = self._get_exchange_args(event)
                if exchange_args is None:
                    continue
                
                actual_state_getter = self._generate_state_getter()
                self._test_exchange(*exchange_args)
                self._check_state(event["state"], actual_state_getter())