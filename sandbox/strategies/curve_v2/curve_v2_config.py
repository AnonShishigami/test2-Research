
N_COINS: int = 2  # <- change
PRECISION: int = 10 ** 18  # The precision to convert to
A_MULTIPLIER: int = 10000

# These addresses are replaced by the deployer

KILL_DEADLINE_DT: int = 2 * 30 * 86400

MIN_GAMMA: int = 10**10
MAX_GAMMA: int = 2 * 10**16
MIN_A: int = N_COINS**N_COINS * A_MULTIPLIER / 10
MAX_A: int = N_COINS**N_COINS * A_MULTIPLIER * 100000
NOISE_FEE: int = 10**5  # 0.1 bps

EXP_PRECISION: int = 10**10
