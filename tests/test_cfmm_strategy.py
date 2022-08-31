import unittest

import numpy as np

from tests.strategy_test_template import StrategyTestTemplate
from tests.strategy_test_oracle import StrategyTestOracle
from sandbox.strategies.cfmm_sqrt.strategy import CFMMSqrt


class TestCFMMSqrt(StrategyTestTemplate, unittest.TestCase):
    
    @staticmethod
    def _get_strategy(state):
        # corresponds to the state right before the first event was validated on chain
        reserve0 = state["reserve0"]
        reserve1 = state["reserve1"]
        # hyper-parameters
        fee_tier = 0.3 / 100
        strategy = CFMMSqrt(
            name="uni_v2", 
            initial_inventories=[reserve0, reserve1],
            initial_cash=0, market=None, oracle=StrategyTestOracle(), support_arb=False, 
            delta=fee_tier
        )
        return strategy
    
    @staticmethod
    def _get_scenarios():
        # each scenario's events should sorted in chronological order
        return [
            {
                "init_state": {
                    "reserve0": 58319143343313,
                    "reserve1": 34218188785634764296618,
                },
                "events": [
                    {
                        "block": 15408828,
                        "type": "Swap",
                        "amount0In": 250000000,
                        "amount1In": 0,
                        "amount0Out": 0,
                        "amount1Out": 146244382452594356,
                        "state": {
                            "reserve0": 58319393343313,
                            "reserve1": 34218042541252311702262,
                        }
                    },
                ]
            },
            {
                "init_state": {
                    "reserve0": 58319325968794,
                    "reserve1": 34218082191252311702262,
                },
                "events": [
                    {
                        "block": 15408840,
                        "type": "Swap",
                        "amount0In": 400000000,
                        "amount1In": 0,
                        "amount0Out": 0,
                        "amount1Out": 233988950256042379,
                        "state": {
                            "reserve0": 58319725968794,
                            "reserve1": 34217848202302055659883,
                        }
                    },
                ]
            }
        ]

    def _get_exchange_args(self, event):
        if event["type"] == "Swap":
            if event["amount0Out"]:
                buy_token = 0
                buy_amount = event["amount0Out"]
                sell_amount = event["amount1In"]
            else:
                buy_token = 1
                buy_amount = event["amount1Out"]
                sell_amount = event["amount0In"]
            return (
                sell_amount / buy_amount,
                buy_token,
                buy_amount,
                0,  # unused
            )
        return None

    def _generate_state_getter(self):
        return lambda: dict((
            ("reserve0", self.strategy.inventories[0]),
            ("reserve1", self.strategy.inventories[1]),
        ))


if __name__ == "__main__":
    unittest.main()
